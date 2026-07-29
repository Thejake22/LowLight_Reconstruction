"""
Central place for all paths used across the project.

Folder layout on disk (fixed, no other images exist in the project):

    D:\\Projects\\LowLight_Reconstruction\\Dataset\\LOLdataset\\our485\\low
    D:\\Projects\\LowLight_Reconstruction\\Dataset\\LOLdataset\\our485\\high
    D:\\Projects\\LowLight_Reconstruction\\Dataset\\LOLdataset\\eval15\\low
    D:\\Projects\\LowLight_Reconstruction\\Dataset\\LOLdataset\\eval15\\high

our485  -> training split   (used by 01_load_dataset.py and for training
                              the deep-unfolding network in 05_deep_unfolding.py)
eval15  -> evaluation split (used by every one-shot denoising method:
                              02_map_tv.py, 03_poisson.py, 04_sparse.py,
                              06_metrics.py, and main.py)

Everything is override-able via environment variables so the project
still runs on a different machine/OS without editing this file:
    LOWLIGHT_PROJECT_ROOT  -> overrides PROJECT_ROOT
    LOLDATASET_ROOT        -> overrides DATASET_ROOT
"""

import os

# Root of the project.
PROJECT_ROOT = os.environ.get(
    "LOWLIGHT_PROJECT_ROOT",
    r"D:\Projects\LowLight_Reconstruction",
)

# Root of the LOL dataset (contains our485/ and eval15/).
DATASET_ROOT = os.environ.get(
    "LOLDATASET_ROOT",
    os.path.join(PROJECT_ROOT, "Dataset", "LOLdataset"),
)

# ---- Training split (our485) ----
TRAIN_ROOT = os.path.join(DATASET_ROOT, "our485")
TRAIN_LOW_PATH = os.path.join(TRAIN_ROOT, "low")
TRAIN_HIGH_PATH = os.path.join(TRAIN_ROOT, "high")

# ---- Evaluation split (eval15) ----
EVAL_ROOT = os.path.join(DATASET_ROOT, "eval15")
EVAL_LOW_PATH = os.path.join(EVAL_ROOT, "low")
EVAL_HIGH_PATH = os.path.join(EVAL_ROOT, "high")

# Backward-compatible aliases: 01_load_dataset.py previews a training
# pair, so LOW_PATH / HIGH_PATH point at the training split.
LOW_PATH = TRAIN_LOW_PATH
HIGH_PATH = TRAIN_HIGH_PATH


def _first_filename(dirpath):
    """Return the first filename in dirpath, or None if missing/empty."""
    if os.path.isdir(dirpath):
        files = sorted(os.listdir(dirpath))
        if files:
            return files[0]
    return None


# LOW_IMAGE / HIGH_IMAGE: the single evaluation pair used by every
# one-shot method script (02, 03, 04, 06, main.py). Pulled from eval15
# (held-out data), never from our485 (training data), so method scripts
# are never accidentally scored on an image they were "trained" on.
_eval_filename = _first_filename(EVAL_LOW_PATH)

if _eval_filename is not None:
    LOW_IMAGE = os.path.join(EVAL_LOW_PATH, _eval_filename)
    HIGH_IMAGE = os.path.join(EVAL_HIGH_PATH, _eval_filename)
else:
    # No images found under eval15/low. Keep a path (not None) so
    # downstream safe_imread() raises a clear FileNotFoundError that
    # points at the right folder, instead of crashing on None.
    LOW_IMAGE = os.path.join(EVAL_LOW_PATH, "<no images found in eval15/low>")
    HIGH_IMAGE = os.path.join(EVAL_HIGH_PATH, "<no images found in eval15/low>")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")