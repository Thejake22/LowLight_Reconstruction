"""
Small shared helpers used by every method script.
Centralizing these avoids each script silently crashing (or worse,
silently producing garbage) when a file is missing.
"""

import os
import cv2
import numpy as np


def ensure_dir(path):
    """Create a directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def safe_imread(path, flags=cv2.IMREAD_COLOR):
    """
    cv2.imread returns None (instead of raising) when a file is missing
    or unreadable. Every original script skipped this check, which meant
    a typo'd path failed later with a confusing shape/None error deep
    inside numpy. Fail loudly and early instead.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, flags)
    if img is None:
        raise ValueError(f"cv2 failed to decode image: {path}")

    return img


def load_rgb(path):
    """Load an image and convert BGR (OpenCV default) -> RGB."""
    img = safe_imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gray_float(path):
    """Load an image as grayscale, normalized to [0, 1] floats."""
    img = safe_imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype("float64") / 255.0


def auto_gamma_correct(img, target_mean=0.5, eps=1e-6):
    """
    Simple, ground-truth-free brightness fix.

    map_tv / poisson_denoise / sparse_denoise are pure *denoisers* - they
    remove noise but never touch overall brightness. On the LOL dataset
    the low/high gap is mostly an illumination gap (dark vs. bright), not
    a noise gap, so a denoiser alone leaves the output dark and scores
    terribly against the bright ground truth even when the noise removal
    itself worked fine.

    This solves for a single global gamma such that mean(img**gamma) is
    approximately target_mean, then applies it - i.e. "brighten/darken
    the whole image until its average brightness looks like a normally
    lit photo". It only looks at the image being corrected, never at the
    ground truth, so it's a fair thing to apply at inference/evaluation
    time (no leakage). It is NOT a substitute for a real illumination
    model (Retinex-style decomposition, or the learned deep-unfolding
    network in 05_deep_unfolding.py) - just a cheap first step so the
    classical methods are at least in the right brightness ballpark.

    img must be a float array in [0, 1] (any number of channels).
    """
    current_mean = float(np.clip(img, eps, 1.0).mean())
    if current_mean <= eps:
        return img  # fully black frame; nothing sane to solve for

    gamma = np.log(target_mean) / np.log(current_mean)
    gamma = np.clip(gamma, 0.1, 10.0)  # guard against extreme/degenerate gammas

    return np.clip(img ** gamma, 0, 1)


def agcwd_equalize(channel, alpha=0.5, eps=1e-6):
    """
    Adaptive Gamma Correction with Weighting Distribution (Huang, Cheng &
    Chiu, 2013) - a per-intensity-level, histogram-aware alternative to a
    single global gamma value (see auto_gamma_correct above).

    auto_gamma_correct solves for ONE gamma exponent that fixes the
    image's overall mean brightness - every intensity level gets exactly
    the same correction. AGCWD instead computes a *different* gamma at
    every intensity level, based on how much probability mass that level
    has relative to the rest of the histogram: under-represented dark
    levels (which dominate low-light images) get pushed up more
    aggressively, while already-common mid/bright levels are boosted
    less. This recovers more of the true dynamic range and avoids the
    flat, washed-out look a single global gamma can produce, which
    generally shows up as a meaningful PSNR/SSIM improvement over plain
    global gamma correction on low-light data.

    channel: float array in [0, 1], SINGLE channel (e.g. the L channel
             of a LAB image, or a grayscale image). Call this per-channel
             or on a luminance channel - don't pass a 3-channel RGB
             array directly, or you'll wash out the color.
    alpha: controls how aggressively rare intensity levels are boosted.
           Higher = closer to standard histogram equalization (stronger,
           can look harsh). Lower = gentler, closer to a linear mapping.
           0.5 is the default used in the original paper and is a
           reasonable middle ground.

    Returns a float array in [0, 1], same shape as the input.
    """
    img_u8 = np.clip(channel * 255.0, 0, 255).astype(np.uint8)

    hist, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 255))
    pdf = hist / (hist.sum() + eps)

    pdf_max = pdf.max()
    pdf_min = pdf.min()
    # Weighted PDF: compress the influence of whichever intensity level
    # happens to be most common, so the mapping doesn't over-enhance
    # just because one level dominates the histogram.
    pdf_w = pdf_max * np.power(
        np.clip((pdf - pdf_min) / (pdf_max - pdf_min + eps), 0, None), alpha
    )
    cdf_w = np.cumsum(pdf_w) / (pdf_w.sum() + eps)

    # Per-level gamma: levels with more cumulative weighted mass below
    # them (brighter / more common) get a gamma closer to 1 (less
    # boost); rare, dark levels get a smaller gamma (bigger boost).
    gamma = np.clip(1.0 - cdf_w, 0.05, 1.0)
    levels = np.arange(256) / 255.0
    mapped_levels = np.power(levels, gamma)

    lut = np.clip(mapped_levels * 255.0, 0, 255).astype(np.uint8)
    result_u8 = lut[img_u8]

    return result_u8.astype(np.float64) / 255.0


def gray_world_white_balance(img, eps=1e-6):
    """
    Simple gray-world white balance for color casts.

    Low-light sensor images commonly have a color cast (often blue- or
    green-heavy) that neither TV/Poisson denoising nor auto_gamma_correct
    touches, since both operate identically on every channel. This scales
    each channel so all three channel means match the overall mean - the
    standard cheap "assume the scene averages out to gray" auto white
    balance heuristic. Not a substitute for a real color-constancy model,
    but a meaningful, ground-truth-free improvement over doing nothing.

    img: float array, shape (H, W, 3), values in [0, 1]. Grayscale images
    are returned unchanged (nothing to balance across channels).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    channel_means = img.reshape(-1, 3).mean(axis=0)
    overall_mean = channel_means.mean()
    scale = overall_mean / np.clip(channel_means, eps, None)
    balanced = img * scale.reshape(1, 1, 3)

    return np.clip(balanced, 0, 1)
