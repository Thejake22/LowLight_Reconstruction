"""
Computes PSNR / SSIM between a result image and its ground truth.

Fixes vs. original:
  - No check that either file loaded successfully (cv2.imread -> None
    on a bad path), which crashed with a confusing error inside the
    metric functions instead of a clear "file not found".
  - No check that original/result have the same shape - skimage
    raises a hard-to-read error if they mismatch; now checked upfront.
  - data_range was not specified. Since both images are pre-divided
    by 255 into [0,1], skimage should be told data_range=1.0 -
    otherwise (depending on skimage version) it infers the range from
    the arrays themselves, which is fragile and can silently skew
    PSNR/SSIM values.
"""

import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import safe_imread


def compute_metrics(original_path, result_path):
    """
    Compare a result image against its ground truth.

    Some methods (histogram equalization, sparse coding) intentionally
    produce single-channel grayscale output while the ground truth is
    loaded as color. The original version of this function treated any
    shape difference as an error, which meant those methods' results
    were skipped entirely instead of scored. If one side is grayscale
    and the other is color, convert the color side to grayscale and
    compare there - a real, if less complete, number beats no number.
    A genuine mismatch (different height/width) still raises, since
    that indicates a real problem rather than a channel-count
    difference.
    """
    original = safe_imread(original_path) / 255.0
    result = safe_imread(result_path) / 255.0

    if original.shape != result.shape:
        original_is_gray = original.ndim == 2
        result_is_gray = result.ndim == 2

        if original_is_gray and not result_is_gray:
            result = cv2.cvtColor((result * 255).astype("uint8"), cv2.COLOR_BGR2GRAY) / 255.0
        elif result_is_gray and not original_is_gray:
            original = cv2.cvtColor((original * 255).astype("uint8"), cv2.COLOR_BGR2GRAY) / 255.0
        else:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs result {result.shape}"
            )

    if original.ndim == 2:
        psnr = peak_signal_noise_ratio(original, result, data_range=1.0)
        ssim = structural_similarity(original, result, data_range=1.0)
    else:
        psnr = peak_signal_noise_ratio(original, result, data_range=1.0)
        ssim = structural_similarity(original, result, channel_axis=2, data_range=1.0)

    return psnr, ssim


def main():
    from config import HIGH_IMAGE, RESULTS_DIR

    result_path = f"{RESULTS_DIR}/map_tv.png"  # change to whichever result to score
    psnr, ssim = compute_metrics(HIGH_IMAGE, result_path)

    print("PSNR:", psnr)
    print("SSIM:", ssim)


if __name__ == "__main__":
    main()
