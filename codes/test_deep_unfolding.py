"""
Small standalone script to load a trained Deep Unfolding checkpoint and
run it once on the eval image, without going through main.py.

Fixes vs. original:
  - DeepUnfoldingNet(n_stages=3) is now instantiated the same way
    load_trained_model() in 05_deep_unfolding.py does (n_stages and
    n_features must match what the checkpoint was trained with, or
    load_state_dict raises a size-mismatch error).
  - Uses SourceFileLoader to import the sibling "05_..." module (same
    trick main.py uses), but now resolves it relative to this file's
    own folder via __file__, instead of a bare relative filename that
    only worked if you happened to run this from the codes/ folder.
"""

import os
import cv2
import torch
import numpy as np
from importlib.machinery import SourceFileLoader

from config import LOW_IMAGE, RESULTS_DIR
from utils import load_rgb, ensure_dir

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
module = SourceFileLoader(
    "deep_unfolding",
    os.path.join(THIS_DIR, "05_deep_unfolding.py"),
).load_module()

DeepUnfoldingNet = module.DeepUnfoldingNet
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "deep_unfolding_model.pt")


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. "
            "Run `python 05_deep_unfolding.py` once to train and save it."
        )

    model = DeepUnfoldingNet(n_stages=module.N_STAGES, n_features=module.N_FEATURES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    return model, device


def enhance_image(model, device, image_path):
    img = load_rgb(image_path).astype("float32") / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(output, 0, 1)


def main():
    ensure_dir(RESULTS_DIR)
    model, device = load_model()

    input_image = LOW_IMAGE
    result = enhance_image(model, device, input_image)

    output = (result * 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    save_path = os.path.join(RESULTS_DIR, "enhanced_output.png")
    cv2.imwrite(save_path, output_bgr)
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
