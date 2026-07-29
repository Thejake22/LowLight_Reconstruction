"""
MAP estimate with Total-Variation prior.

Fixes vs. original:
  - Original read the image with cv2.imread (BGR) but never converted
    to RGB, so saved output had swapped R/B channels.
  - denoise_tv_chambolle was called without `channel_axis`, so on a
    color (H,W,3) image skimage treated it as a 3D *grayscale* volume
    and denoised across channels as if they were depth slices -
    producing color bleeding instead of per-pixel TV smoothing.
  - No check that the input file exists, and results/ was never
    created, so cv2.imwrite silently failed if the folder was missing.
"""

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle

from config import LOW_IMAGE, RESULTS_DIR
from utils import load_rgb, ensure_dir, auto_gamma_correct, gray_world_white_balance


def map_tv(image_rgb, weight=0.15, target_mean=0.5):
    image = image_rgb / 255.0

    result = denoise_tv_chambolle(
        image,
        weight=weight,
        channel_axis=-1,  # treat last dim as color channels, not depth
    )

    # TV denoising removes noise but does nothing about brightness or
    # color cast. On a low-light input the result is still dark and
    # off-color, so it scores badly against a bright, neutral ground
    # truth even though the denoising itself worked. Correct color cast
    # first, then lift overall brightness (see utils.py for both).
    result = gray_world_white_balance(result)
    result = auto_gamma_correct(result, target_mean=target_mean)

    return np.clip(result, 0, 1)


def main():
    ensure_dir(RESULTS_DIR)

    img_rgb = load_rgb(LOW_IMAGE)
    output = map_tv(img_rgb)

    # convert back to BGR for cv2.imwrite
    output_bgr = cv2.cvtColor((output * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    out_path = f"{RESULTS_DIR}/map_tv.png"
    cv2.imwrite(out_path, output_bgr)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
