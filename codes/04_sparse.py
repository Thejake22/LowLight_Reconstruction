"""
Sparse-coding based denoising via learned dictionary of 8x8 patches.

Fixes vs. original:
  - BIGGEST bug: the original saved `reconstructed[0]*255` - i.e. only
    the *first* 8x8 patch, reshaped and written out as if it were the
    whole result. The other 4999 reconstructed patches were computed
    and thrown away. Now every patch is stitched back together with
    reconstruct_from_patches_2d and overlaps are averaged.
  - No random_state was set on DictionaryLearning, so results (and
    dictionary quality) were not reproducible between runs.
  - No safe_imread / missing results/ dir handling.
  - Patch extraction and reconstruction now go through the same image
    shape, so reconstruct_from_patches_2d knows the correct output size.
  - "Orthogonal matching pursuit ended prematurely due to linear
    dependence in the dictionary": low-light patches are mostly
    near-flat/near-black, so a large fraction of the 5000 training
    patches are nearly identical up to a scale factor. Learning a
    dictionary directly on raw intensities lets atoms collapse onto
    each other (rank-deficient dictionary) - textbook sparse-coding
    practice is to remove each patch's own mean (its DC / average
    brightness) before learning and coding, then add it back after
    reconstruction, so the dictionary only has to represent *texture*,
    not brightness level. That alone removes most of the degeneracy.
    Also switched the coding step from the default OMP - which is the
    algorithm actually emitting the warning - to lasso_lars, which is
    numerically stable even when dictionary atoms are highly
    correlated (the same OMP failure mode; you'd still see the warning
    with pure mean-centering alone if it's kept).
  - No output brightness/color correction (see auto_gamma_correct in
    utils.py) - same illumination-gap issue as the other classical
    methods; a denoiser alone can't fix "dark vs. bright".
"""

import cv2
import numpy as np

from sklearn.decomposition import DictionaryLearning
from sklearn.feature_extraction.image import (
    extract_patches_2d,
    reconstruct_from_patches_2d,
)

from config import LOW_IMAGE, RESULTS_DIR
from utils import load_gray_float, ensure_dir, auto_gamma_correct

PATCH_SIZE = (8, 8)
N_COMPONENTS = 64
N_TRAIN_PATCHES = 5000


def sparse_denoise(img):
    # Sample a subset of patches to *train* the dictionary (cheap).
    train_patches = extract_patches_2d(
        img, PATCH_SIZE, max_patches=N_TRAIN_PATCHES, random_state=0
    )
    train_patches_flat = train_patches.reshape(len(train_patches), -1)

    # Remove each patch's own mean before learning. Without this, dark
    # low-light patches (which dominate this dataset) are nearly all
    # constant-valued, so the dictionary atoms collapse into each other
    # and become linearly dependent - the root cause of the OMP warning.
    train_means = train_patches_flat.mean(axis=1, keepdims=True)
    train_patches_centered = train_patches_flat - train_means

    dictionary = DictionaryLearning(
        n_components=N_COMPONENTS,
        alpha=1,
        max_iter=100,
        random_state=0,
        # lasso_lars is numerically stable with correlated/near-degenerate
        # atoms; the default 'omp' is what produced the warning.
        transform_algorithm="lasso_lars",
    )
    dictionary.fit(train_patches_centered)

    # Now extract *every* patch (no sampling) so we can reconstruct
    # the full image, not just one patch.
    all_patches = extract_patches_2d(img, PATCH_SIZE)
    all_patches_flat = all_patches.reshape(len(all_patches), -1)

    all_means = all_patches_flat.mean(axis=1, keepdims=True)
    all_patches_centered = all_patches_flat - all_means

    code = dictionary.transform(all_patches_centered)
    reconstructed_flat = code @ dictionary.components_

    # Add each patch's own mean back - the dictionary only modeled the
    # zero-mean texture, not the brightness level.
    reconstructed_flat = reconstructed_flat + all_means

    reconstructed_patches = reconstructed_flat.reshape(
        len(all_patches), PATCH_SIZE[0], PATCH_SIZE[1]
    )

    # Stitch overlapping patches back into a full image (averages overlaps).
    full_image = reconstruct_from_patches_2d(reconstructed_patches, img.shape)
    full_image = np.clip(full_image, 0, 1)

    # Same illumination-gap issue as the other classical methods: the
    # dictionary reconstructs (denoised) low-light patches, it doesn't
    # know anything about "bright". Lift brightness globally afterwards.
    full_image = auto_gamma_correct(full_image, target_mean=0.5)

    return np.clip(full_image, 0, 1)


def main():
    ensure_dir(RESULTS_DIR)

    img = load_gray_float(LOW_IMAGE)
    result = sparse_denoise(img)

    out_path = f"{RESULTS_DIR}/sparse.png"
    cv2.imwrite(out_path, (result * 255).astype("uint8"))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

