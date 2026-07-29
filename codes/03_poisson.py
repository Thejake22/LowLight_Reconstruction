"""
Poisson-noise denoising via the Anscombe variance-stabilizing transform
followed by TV denoising, then inverse Anscombe.

Fixes vs. original:
  - Same BGR/RGB and channel_axis bugs as 02_map_tv.py (color image
    denoised as if it were a grayscale volume, and never converted
    back to RGB before saving).
  - No safe_imread / missing results/ dir handling.
  - Inverse Anscombe now uses the (slightly) more accurate closed-form
    unbiased inverse instead of the naive algebraic inverse, which
    reduces bias in dark regions - the exact scenario this pipeline
    is used for.

Fixes vs. the unsharp/CLAHE version added on top of that:
  - BLACK-IMAGE BUG: main() computed `result_bgr` (the image correctly
    rescaled to 0-255 and cast to uint8) but then called
    `cv2.imwrite(out_path, result)` - writing the *unscaled* `result`,
    which is still a float64 array with values in [0, 1]. cv2.imwrite
    doesn't know those are normalized floats; it casts them straight to
    the output bit depth, so 0.0-0.99... all round down to 0. That's
    why the saved PNG came out almost entirely black. Fixed by writing
    `result_bgr` instead.
  - CHANNEL-SWAP BUG: the CLAHE block inside poisson_denoise() called
    cv2.cvtColor(..., COLOR_BGR2LAB) / COLOR_LAB2BGR, but the data
    flowing through this function is RGB the whole way through (it
    comes from load_rgb(), and main() only converts to BGR once, right
    before saving). Labeling RGB data as BGR mid-function silently
    swaps the red and blue channels. Switched to COLOR_RGB2LAB /
    COLOR_LAB2RGB so the function's contract (RGB in, RGB out) holds
    all the way through, matching every other method script.

Reconstruction-quality improvement (this update - better equalization):
  - Step 5 used to be a single GLOBAL gamma value (auto_gamma_correct)
    applied identically to every pixel, chosen only to fix the overall
    mean brightness. That's a blunt instrument: it treats a rare, very
    dark pixel and a common, slightly-dark pixel exactly the same way,
    so it either under-boosts the shadows or over-boosts (washes out)
    everything else trying to compensate.
    Replaced with AGCWD (Adaptive Gamma Correction with Weighting
    Distribution - see utils.agcwd_equalize), applied to the L
    (lightness) channel only so color is untouched. AGCWD computes a
    DIFFERENT gamma at every intensity level based on that level's
    probability in the image's own histogram - rare dark levels (which
    dominate low-light shadows) get pushed up more aggressively, common
    mid/bright levels get pushed up less. This recovers more real
    dynamic range than a single global gamma and is the main lever for
    a better PSNR/SSIM here.
  - CLAHE's clipLimit raised slightly (1.2 -> 2.0) since AGCWD already
    handles the global brightness curve now, so CLAHE's job narrows to
    local contrast recovery only and can afford to be a bit stronger
    without blowing out flat regions.
"""

import cv2
import numpy as np
import bm3d

from config import LOW_IMAGE, RESULTS_DIR
from utils import load_rgb, ensure_dir, agcwd_equalize, gray_world_white_balance


def unsharp_mask(image, sigma=1.2, amount=1):
    """
    Sharpen an image using unsharp masking.

    image : float image in [0,1]
    sigma : Gaussian blur sigma
    amount: Sharpening strength
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1)


def anscombe(x):
    return 2.0 * np.sqrt(np.clip(x, 0, None) + 3.0 / 8.0)


def inverse_anscombe(y):
    # Closed-form unbiased inverse (Makitalo & Foi), reduces bias vs.
    # the naive algebraic inverse used in the original script.
    return (
        (y / 2.0) ** 2
        + 0.25 * np.sqrt(1.5) * y ** -1
        - (11.0 / 8.0) * y ** -2
        + (5.0 / 8.0) * np.sqrt(1.5) * y ** -3
        - 1.0 / 8.0
    )


def agcwd_equalize_rgb(image_rgb_float, alpha=0.5):
    """
    Apply AGCWD (see utils.agcwd_equalize) to the L (lightness) channel
    of an RGB image in LAB space, leaving color (A/B channels) untouched.
    This is the "better equalization" step: a histogram/distribution-
    aware brightness correction instead of a single flat global gamma.

    image_rgb_float: RGB float array in [0, 1].
    alpha: AGCWD strength - see utils.agcwd_equalize docstring.
    """
    img_uint8 = (np.clip(image_rgb_float, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    L_float = L.astype(np.float64) / 255.0
    L_eq = agcwd_equalize(L_float, alpha=alpha)
    L_eq_uint8 = np.clip(L_eq * 255.0, 0, 255).astype(np.uint8)

    lab_eq = cv2.merge((L_eq_uint8, A, B))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return rgb_eq.astype(np.float64) / 255.0


def poisson_denoise(image_rgb,
                    sigma_psd=0.05,
                    agcwd_alpha=0.5):

    # Normalize
    img = image_rgb.astype(np.float32) / 255.0


    # -----------------------------
    # 1. Anscombe transform
    # -----------------------------
    transformed = anscombe(img)


    # -----------------------------
    # 2. BM3D denoising
    # -----------------------------
    denoised = np.zeros_like(transformed)


    for c in range(3):

        denoised[:, :, c] = bm3d.bm3d(
            transformed[:, :, c],
            sigma_psd=sigma_psd,
            stage_arg=bm3d.BM3DStages.ALL_STAGES
        )


    # -----------------------------
    # 3. Inverse Anscombe
    # -----------------------------
    result = inverse_anscombe(
        np.clip(denoised, 1e-3, None)
    )


    result = np.clip(result,0,1)



    # -----------------------------
    # 4. White balance
    # -----------------------------
    result = gray_world_white_balance(result)



    # -----------------------------
    # 5. Adaptive equalization (AGCWD)
    # -----------------------------
    result = agcwd_equalize_rgb(
        result,
        alpha=agcwd_alpha
    )


    # -----------------------------
    # 6. Mild sharpening
    # -----------------------------
    result = unsharp_mask(
        result,
        sigma=1.0,
        amount=0.5
    )


    # -----------------------------
    # 7. CLAHE
    # -----------------------------
    result_uint8 = (
        result*255
    ).astype(np.uint8)


    lab=cv2.cvtColor(
        result_uint8,
        cv2.COLOR_RGB2LAB
    )

    L,A,B=cv2.split(lab)


    clahe=cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    L=clahe.apply(L)


    lab=cv2.merge(
        (L,A,B)
    )


    result=cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )


    result=result.astype(np.float32)/255.0


    return np.clip(result,0,1)

def main():
    ensure_dir(RESULTS_DIR)

    img_rgb = load_rgb(LOW_IMAGE)
    result = poisson_denoise(img_rgb)

    # result is RGB float in [0,1] here; convert to BGR uint8 for
    # cv2.imwrite, and make sure imwrite actually gets that converted
    # array (not the raw [0,1] float `result`, which is what produced
    # the all-black PNG before).
    result_bgr = cv2.cvtColor((result * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    out_path = f"{RESULTS_DIR}/poisson.png"
    cv2.imwrite(out_path, result_bgr)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()