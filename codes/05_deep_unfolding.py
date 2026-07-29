"""
Deep-unfolding network: alternates a data-fidelity (gradient) step with
a learned denoiser, mimicking K iterations of a proximal-gradient solver.

=== Color-collapse fix (previous update) ===
UnfoldBlock's denoiser predicts a RESIDUAL correction on top of the
gradient step, not a replacement, and its last layer is zero-initialized
so training starts from an exact, colour-correct identity mapping. See
the README for the full story - this is what fixed the earlier
grayscale-looking output.

=== This update: higher PSNR, tuned for a GPU with 8 GB VRAM ===

  - MORE CAPACITY: 3 -> 5 stages, 64 -> 96 features per conv layer. More
    proximal-gradient iterations and a wider denoiser give the model
    more room to actually remove low-light degradation instead of just
    a mild correction. This was previously kept small assuming CPU-only
    training; a 4060 has plenty of headroom for it.

  - BEST-CHECKPOINT SAVING (this is the single biggest lever for a
    higher reported PSNR): training now holds out a small validation
    split from our485 and, after every epoch, measures validation PSNR
    and saves the checkpoint ONLY if it's the best one seen so far.
    Previously the checkpoint was just whatever the model looked like
    after the last epoch - which is not necessarily the best one, and
    with more capacity + more epochs the model can start overfitting
    the training crops in later epochs (loss keeps dropping on training
    data, but validation PSNR gets worse). Now you always keep the
    checkpoint that generalizes best.

  - LOSS RE-WEIGHTED TOWARD MSE: PSNR = 10*log10(1 / MSE) - it is a
    direct, monotonic function of mean-squared-error. The previous pure
    L1 loss optimizes for something related but not identical (L1 is
    more robust to outliers and tends to keep edges sharp, but doesn't
    directly minimize the quantity PSNR is computed from). Loss is now
    a weighted combination of L1 (kept small, for edge stability) and
    MSE (dominant, since that's what you asked to improve):
    loss = 0.2 * L1 + 1.0 * MSE. If outputs get slightly blurrier than
    you'd like, lower MSE_WEIGHT / raise L1_WEIGHT below.

  - DATA AUGMENTATION: each training crop is randomly flipped
    horizontally/vertically and rotated in 90-degree steps (applied
    identically to the low/high pair). This multiplies the effective
    variety seen from the same 485 training pairs at ~zero cost, which
    matters more now that the model has more capacity to (over)fit.

  - GPU-SCALE TRAINING: crop size 128 -> 192, batch size 8 -> 16,
    default epochs 20 -> 60, and automatic mixed-precision (torch.cuda
    .amp) when running on a CUDA GPU - roughly 2x faster / lets you fit
    the larger crops+batches in the same VRAM budget as before. Mixed
    precision is skipped automatically on CPU (it isn't supported
    there), so this remains safe to run without a GPU too, just slower.

  - CHECKPOINT-ARCHITECTURE MISMATCH HANDLING: since the default
    n_stages/n_features changed above, a checkpoint trained with the
    OLD architecture will not load into the new one (PyTorch raises a
    size-mismatch error). load_trained_model() now catches that and
    returns None with a clear message telling you to retrain, instead
    of crashing main.py / the Streamlit app.

  - RETRAIN-AVOIDANCE (unchanged from before): train() checks
    CHECKPOINT_PATH first and reuses an existing (compatible) checkpoint
    unless you pass --retrain.
"""

import os

# Reduce CUDA memory fragmentation before any torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config import TRAIN_LOW_PATH, TRAIN_HIGH_PATH, LOW_IMAGE, RESULTS_DIR
from utils import load_rgb, ensure_dir

# ---- Training knobs ----
# Tuned for an 8 GB VRAM GPU. If you still hit OOM, lower BATCH_SIZE
# first (try 4), then CROP_SIZE (try 96).
N_EPOCHS = 60
BATCH_SIZE = 8       # was 16 — halved to free ~2 GB of activation memory
LEARNING_RATE = 2e-4
CROP_SIZE = 128      # was 192 — smaller crops cut VRAM by ~(128/192)^2 ≈ 44%
VAL_FRACTION = 0.08  # held out from our485 for best-checkpoint selection
N_FEATURES = 64      # was 96 — narrower convs, still enough capacity
N_STAGES = 4         # was 5  — one fewer unrolled stage
L1_WEIGHT = 0.2
MSE_WEIGHT = 1.0
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "deep_unfolding_model.pt")


class UnfoldBlock(nn.Module):
    """One proximal-gradient iteration: gradient step + residual denoiser."""

    def __init__(self, n_features=N_FEATURES):
        super().__init__()

        self.step = nn.Parameter(torch.tensor(0.1))

        self.denoiser = nn.Sequential(
            nn.Conv2d(3, n_features, 3, padding=1),
            nn.GroupNorm(8, n_features),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.GroupNorm(8, n_features),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_features, 3, 3, padding=1),
        )

        # Zero-init the final layer: at the start of training the
        # denoiser outputs exactly 0, so this block starts out as pure
        # identity-on-the-gradient-step (colour-correct by construction)
        # instead of a random, potentially colour-destroying mapping.
        nn.init.zeros_(self.denoiser[-1].weight)
        nn.init.zeros_(self.denoiser[-1].bias)

    def forward(self, x, y):
        # Gradient step on the data-fidelity term ||x - y||^2: moves x
        # towards the observed low-light image y, not towards zero.
        x_grad = x - self.step * (x - y)

        # Residual correction, not a replacement.
        residual = self.denoiser(x_grad)
        x = x_grad + residual

        return torch.clamp(x, 0.0, 1.0)


class DeepUnfoldingNet(nn.Module):
    """Stacks K UnfoldBlocks, each seeing the original observation y."""

    def __init__(self, n_stages=N_STAGES, n_features=N_FEATURES):
        super().__init__()
        self.blocks = nn.ModuleList(
            [UnfoldBlock(n_features=n_features) for _ in range(n_stages)]
        )

    def forward(self, y):
        x = y  # initialize the estimate with the observed low-light image
        for block in self.blocks:
            x = block(x, y)
        return x


def _augment_pair(low_crop, high_crop):
    """
    Randomly flip/rotate a (low, high) crop pair identically, so the
    model sees ~8x the effective variety from the same training pairs.
    Both arrays are HWC float32 in [0, 1].
    """
    if random.random() < 0.5:
        low_crop = np.fliplr(low_crop)
        high_crop = np.fliplr(high_crop)
    if random.random() < 0.5:
        low_crop = np.flipud(low_crop)
        high_crop = np.flipud(high_crop)
    k = random.randint(0, 3)
    if k:
        low_crop = np.rot90(low_crop, k)
        high_crop = np.rot90(high_crop, k)
    return np.ascontiguousarray(low_crop), np.ascontiguousarray(high_crop)


class LOLDataset(Dataset):
    """
    Loads matching low/high pairs from two folders and returns random
    CROP_SIZE x CROP_SIZE crops (same crop location for both images),
    with optional flip/rotation augmentation.
    """

    def __init__(self, low_dir, high_dir, crop_size=CROP_SIZE, augment=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.crop_size = crop_size
        self.augment = augment

        if not os.path.isdir(low_dir):
            raise FileNotFoundError(f"Training low/ folder not found: {low_dir}")
        if not os.path.isdir(high_dir):
            raise FileNotFoundError(f"Training high/ folder not found: {high_dir}")

        low_files = set(os.listdir(low_dir))
        high_files = set(os.listdir(high_dir))
        self.filenames = sorted(low_files & high_files)

        if not self.filenames:
            raise FileNotFoundError(
                f"No matching filenames between {low_dir} and {high_dir}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        low = load_rgb(os.path.join(self.low_dir, name)).astype("float32") / 255.0
        high = load_rgb(os.path.join(self.high_dir, name)).astype("float32") / 255.0

        h, w = low.shape[:2]
        c = self.crop_size
        if h < c or w < c:
            raise ValueError(
                f"{name} is {w}x{h}, smaller than crop_size={c}. "
                f"Lower CROP_SIZE in 05_deep_unfolding.py."
            )

        top = random.randint(0, h - c)
        left = random.randint(0, w - c)
        low_crop = low[top:top + c, left:left + c, :]
        high_crop = high[top:top + c, left:left + c, :]

        if self.augment:
            low_crop, high_crop = _augment_pair(low_crop, high_crop)

        low_t = torch.from_numpy(low_crop.copy()).permute(2, 0, 1)
        high_t = torch.from_numpy(high_crop.copy()).permute(2, 0, 1)
        return low_t, high_t


def _psnr_from_mse(mse, eps=1e-10):
    """Images are normalized to [0,1], so data_range=1 -> PSNR = 10*log10(1/mse)."""
    return 10.0 * torch.log10(1.0 / (mse + eps))


@torch.no_grad()
def evaluate(model, loader, device):
    """Average validation loss/PSNR over a data loader."""
    model.eval()
    mse_fn = nn.MSELoss()
    total_mse = 0.0
    n_batches = 0
    for low_batch, high_batch in loader:
        low_batch = low_batch.to(device)
        high_batch = high_batch.to(device)
        output = model(low_batch)
        total_mse += mse_fn(output, high_batch).item()
        n_batches += 1
    avg_mse = total_mse / max(n_batches, 1)
    avg_psnr = 10.0 * np.log10(1.0 / max(avg_mse, 1e-10))
    return avg_mse, avg_psnr


def train(
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    crop_size=CROP_SIZE,
    checkpoint_path=CHECKPOINT_PATH,
    device=None,
    force_retrain=False,
):
    """
    Trains DeepUnfoldingNet on the our485 split (with a small held-out
    validation slice) and saves the BEST checkpoint by validation PSNR -
    not just whatever the model looks like after the last epoch.

    If a compatible checkpoint already exists at checkpoint_path and
    force_retrain is False (the default), training is skipped entirely
    and the saved model is loaded and returned instead.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    if not force_retrain:
        existing = load_trained_model(checkpoint_path, device=device)
        if existing is not None:
            print(
                f"Found existing checkpoint at {checkpoint_path} - reusing it "
                f"and skipping training. Pass --retrain to force retraining."
            )
            return existing

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed crop size -> faster conv algo selection

    full_dataset = LOLDataset(TRAIN_LOW_PATH, TRAIN_HIGH_PATH, crop_size=crop_size, augment=True)
    n_val = max(1, int(len(full_dataset) * VAL_FRACTION))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    print(
        f"Training on {n_train} pairs, validating on {n_val} pairs "
        f"(from {TRAIN_LOW_PATH}, device={device}, amp={use_amp})"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeepUnfoldingNet(n_stages=N_STAGES, n_features=N_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    l1_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_psnr = -float("inf")
    ensure_dir(os.path.dirname(checkpoint_path))

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for low_batch, high_batch in train_loader:
            low_batch = low_batch.to(device)
            high_batch = high_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                output = model(low_batch)
                loss = L1_WEIGHT * l1_fn(output, high_batch) + MSE_WEIGHT * mse_fn(output, high_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_mse, val_psnr = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{n_epochs}  train_loss={epoch_loss / n_batches:.4f}  "
            f"val_mse={val_mse:.5f}  val_psnr={val_psnr:.2f} dB"
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best (val_psnr={val_psnr:.2f} dB), saved checkpoint: {checkpoint_path}")

    print(f"Training done. Best validation PSNR: {best_val_psnr:.2f} dB")

    # Reload the BEST checkpoint (not necessarily the state after the
    # final epoch) before returning, so callers always get the model
    # that was actually saved.
    return load_trained_model(checkpoint_path, device=device)


def load_trained_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    """
    Loads a previously trained checkpoint. Returns None if none exists,
    OR if one exists but doesn't match the current architecture (e.g.
    you upgraded N_STAGES/N_FEATURES since it was trained) - in that
    case a message is printed telling you to retrain, instead of
    crashing with a PyTorch size-mismatch error.
    """
    if not os.path.exists(checkpoint_path):
        return None
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepUnfoldingNet(n_stages=N_STAGES, n_features=N_FEATURES).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except RuntimeError as e:
        print(
            f"Checkpoint at {checkpoint_path} doesn't match the current model "
            f"architecture (N_STAGES={N_STAGES}, N_FEATURES={N_FEATURES}) - "
            f"probably trained before an architecture change. Retrain with "
            f"`python 05_deep_unfolding.py --retrain`.\n(Details: {e})"
        )
        return None
    model.eval()
    return model


def run_inference(model, image_path, device=None):
    """Runs the (trained) model on a single image, returns an RGB float array in [0,1]."""
    device = device or next(model.parameters()).device
    img = load_rgb(image_path).astype("float32") / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(output, 0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train (or reuse a checkpoint for) the deep-unfolding low-light model."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if a compatible checkpoint already exists at "
        f"{CHECKPOINT_PATH} (default: reuse it if present).",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "No CUDA GPU detected by PyTorch - training will run on CPU and be "
            "much slower. If you have an NVIDIA GPU, check that you installed a "
            "CUDA-enabled build of torch (see README)."
        )
    ensure_dir(RESULTS_DIR)

    model = train(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        device=device,
        force_retrain=args.retrain,
    )

    print(f"Running trained model on eval image: {LOW_IMAGE}")
    output = run_inference(model, LOW_IMAGE, device=device)

    output_bgr = cv2.cvtColor((output * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    out_path = os.path.join(RESULTS_DIR, "deep_unfolding.png")
    cv2.imwrite(out_path, output_bgr)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
