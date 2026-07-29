"""
Runs every reconstruction method on LOW_IMAGE and scores each against
HIGH_IMAGE.

Fixes vs. original:
  - The original main.py didn't call any of the method implementations
    at all - it just printed the method *names* in a loop and did no
    actual work. This version imports each method's function, runs it,
    saves the result, and reports PSNR/SSIM against the ground truth.
  - Method modules are named starting with digits (e.g. "02_map_tv.py"),
    which Python can't import via a normal `import` statement, so they
    are loaded explicitly with importlib.
  - _load_module() used bare relative filenames ("02_map_tv.py"), which
    get resolved against the process's *current working directory* -
    wherever the shell happened to be when you ran the script - not the
    folder main.py itself lives in. Running from anywhere except
    codes/ (e.g. from your home directory, or a cygwin shell) failed
    with FileNotFoundError even though the files were right there next
    to main.py. Paths are now built from this file's own location via
    __file__, so it works regardless of the current working directory.
  - LOW_IMAGE and HIGH_IMAGE were used but never imported from config -
    this would have crashed with NameError as soon as the path bug
    above was fixed. Now imported alongside RESULTS_DIR.
  - Metrics were computed in-memory on grayscale results (histogram,
    sparse) with shape != color ground truth, so they were silently
    "skipped" instead of scored. Now reuses 06_metrics.compute_metrics
    on the saved PNGs, which handles grayscale-vs-color comparisons
    instead of giving up on them.
  - Deep Unfolding (05_deep_unfolding.py) is now a real trainable model
    with a saved checkpoint. If a checkpoint exists, main.py loads it
    and includes it in the results/metrics table; otherwise it's
    skipped with a message telling you to train it first (training
    takes minutes, so it isn't triggered automatically on every run).
"""

import os
import importlib.util
import cv2

from config import RESULTS_DIR, LOW_IMAGE, HIGH_IMAGE
from utils import ensure_dir, load_rgb, load_gray_float, safe_imread

# Directory this file lives in (codes/), regardless of the shell's cwd.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(filename, module_name):
    # Resolve sibling method scripts relative to THIS file, not the cwd.
    filepath = os.path.join(THIS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Expected method script not found: {filepath}\n"
            f"(main.py is at {THIS_DIR}; make sure {filename} lives in the same folder.)"
        )
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_all():
    ensure_dir(RESULTS_DIR)

    map_tv_mod = _load_module("02_map_tv.py", "map_tv_mod")
    poisson_mod = _load_module("03_poisson.py", "poisson_mod")
    sparse_mod = _load_module("04_sparse.py", "sparse_mod")
    metrics_mod = _load_module("06_metrics.py", "metrics_mod")

    rgb = load_rgb(LOW_IMAGE)
    gray = safe_imread(LOW_IMAGE, cv2.IMREAD_GRAYSCALE)
    gray_f = load_gray_float(LOW_IMAGE)

    results = {}

    print("Running: Histogram Equalization (baseline)")
    results["histogram"] = cv2.equalizeHist(gray)

    print("Running: MAP-TV")
    out = map_tv_mod.map_tv(rgb)
    results["map_tv"] = cv2.cvtColor((out * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

    print("Running: Poisson")
    out = poisson_mod.poisson_denoise(rgb)
    results["poisson"] = cv2.cvtColor((out * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

    print("Running: Sparse Coding")
    out = sparse_mod.sparse_denoise(gray_f)
    results["sparse"] = (out * 255).astype("uint8")

    # Deep unfolding is a trained model, not a one-shot filter - only run
    # it here if it's already been trained (python 05_deep_unfolding.py).
    # If there's no checkpoint yet, skip it instead of training inline
    # every time main.py runs (training takes minutes, not seconds).
    deep_unfolding_mod = _load_module("05_deep_unfolding.py", "deep_unfolding_mod")
    trained_model = deep_unfolding_mod.load_trained_model()
    if trained_model is not None:
        print("Running: Deep Unfolding (using saved checkpoint)")
        out = deep_unfolding_mod.run_inference(trained_model, LOW_IMAGE)
        results["deep_unfolding"] = cv2.cvtColor(
            (out * 255).astype("uint8"), cv2.COLOR_RGB2BGR
        )
    else:
        print(
            "Skipping Deep Unfolding: no trained checkpoint found at "
            f"{deep_unfolding_mod.CHECKPOINT_PATH}. Run "
            "`python 05_deep_unfolding.py` once to train and save it, "
            "then re-run main.py to include it here."
        )

    for name, img in results.items():
        out_path = f"{RESULTS_DIR}/{name}.png"
        cv2.imwrite(out_path, img)
        print(f"  saved {out_path}")

    print("\n--- Metrics (vs. high.png) ---")
    for name in results:
        out_path = f"{RESULTS_DIR}/{name}.png"
        # Read back from disk (like a real evaluation would) and let
        # compute_metrics handle any grayscale-vs-color shape mismatch
        # instead of silently skipping it.
        psnr, ssim = metrics_mod.compute_metrics(HIGH_IMAGE, out_path)
        print(f"{name}: PSNR={psnr:.2f}  SSIM={ssim:.4f}")


if __name__ == "__main__":
    run_all()
