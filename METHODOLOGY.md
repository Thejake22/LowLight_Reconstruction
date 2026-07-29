# Methodology

Low-light photos suffer from two largely separate problems that get
conflated if you're not careful:

1. **Noise** - sensors are noisier in low light (fewer photons hit each
   pixel per unit time, so shot noise dominates).
2. **Illumination gap** - the low-light image is just globally too dark
   compared to a normally-lit photo, independent of noise.

The three classical methods here (`02_map_tv.py`, `03_poisson.py`,
`04_sparse.py`) are all fundamentally **denoisers** - they don't know
anything about the target brightness. Left alone, a perfect denoiser on
a low-light photo just gives you a *clean, still-dark* photo, which
scores badly against a bright ground truth even when the denoising
worked. So each of them is followed by two small, ground-truth-free
brightness/color corrections (`utils.py`):

- **`auto_gamma_correct`**: solves for a single global gamma so the
  image's mean brightness lands near a target (default 0.5), i.e.
  "brighten until it looks like a normally-lit photo." Only looks at
  the image being corrected - never the ground truth - so it's fair to
  apply at evaluation time.
- **`gray_world_white_balance`**: rescales each color channel so their
  means match, correcting the blue/green colour casts common in
  low-light sensor images, which denoising alone doesn't touch.

Neither is a substitute for a real illumination model - they're a cheap
way to get the classical methods into the right ballpark so a fair
comparison against the learned method is possible.

## 1. Histogram Equalization (baseline)

Redistributes the image's intensity histogram to be closer to uniform,
which tends to spread out concentrated dark values and increase
contrast/brightness. No denoising, no dataset - just a per-image,
parameter-free baseline everything else should beat.

## 2. MAP-TV (`02_map_tv.py`)

A Maximum-A-Posteriori estimate under a Total-Variation (TV) prior:

```
argmin_x  (1/2)||x - y||^2 + weight * TV(x)
```

`TV(x)` penalizes the sum of gradient magnitudes, i.e. it prefers
piecewise-smooth images (flattens noise) while still allowing sharp
edges (unlike Gaussian blur, which penalizes all high frequencies
including real edges). Solved here via `skimage.restoration
.denoise_tv_chambolle` (Chambolle's projection algorithm), applied per
color channel (`channel_axis=-1`) so color information isn't blended
across channels. Followed by white balance + gamma correction as above.

## 3. Poisson denoising (`03_poisson.py`)

Sensor (shot) noise in low light is closer to Poisson-distributed than
Gaussian - its variance scales with the signal itself, so a naive
Gaussian-noise denoiser under- or over-smooths depending on local
brightness. The standard fix:

1. **Anscombe transform**: `f(x) = 2*sqrt(x + 3/8)`, which approximately
   converts Poisson noise into unit-variance Gaussian noise.
2. **Denoise in the transformed domain** with BM3D (block-matching and
   3D filtering), a strong Gaussian denoiser, per color channel.
3. **Inverse Anscombe transform**: this implementation uses the
   closed-form *unbiased* inverse (Makitalo & Foi), which is less biased
   in dark regions than the naive algebraic inverse `f^-1(y) = (y/2)^2 - 3/8`
   - the darkest regions are exactly where this pipeline is used most.
4. White balance, then **AGCWD** (Adaptive Gamma Correction with
   Weighting Distribution - Huang, Cheng & Chiu, 2013) instead of a
   single flat gamma value: AGCWD computes a *histogram of the image's
   own L (lightness) channel*, then a *different* gamma exponent for
   every intensity level based on how rare/common that level is. Levels
   that are under-represented (the dark shadow tones that dominate
   low-light images) get pushed up aggressively; levels that are
   already common (mid-tones, highlights) get pushed up gently. This
   recovers more real dynamic range than one global gamma can, without
   washing out already-bright regions. Applied to the L channel only
   (LAB space) so color is untouched - see `utils.agcwd_equalize`.
5. A mild unsharp mask, then CLAHE (contrast-limited adaptive histogram
   equalization, in LAB space on the L channel only) for local contrast
   - AGCWD handles the global brightness curve, CLAHE recovers local
   contrast on top of it.

## 4. Sparse Coding (`04_sparse.py`)

Learns a dictionary of small (8x8) image patches such that every patch
in the image can be approximately reconstructed as a *sparse* linear
combination of a handful of "atoms" from that dictionary
(`sklearn.decomposition.DictionaryLearning`, coded with `lasso_lars`).
Noise doesn't have consistent patch structure, so it's expensive to
represent sparsely and gets discarded during reconstruction, while
genuine image structure survives.

Two practical details matter a lot for low-light patches specifically:

- **Mean-centering.** Low-light patches are mostly near-flat/near-black,
  so most training patches are nearly identical up to a scale factor.
  Learning directly on raw intensities lets dictionary atoms collapse
  onto each other (rank-deficient dictionary, manifesting as an "OMP
  ended prematurely" warning). Subtracting each patch's own mean before
  training/coding, and adding it back after reconstruction, means the
  dictionary only has to model *texture*, not brightness level, which
  removes most of the degeneracy.
- **Full reconstruction.** Every patch (not just a training subsample)
  is coded and stitched back with `reconstruct_from_patches_2d`
  (overlaps averaged), then gamma-corrected as above.

## 5. Deep Unfolding network (`05_deep_unfolding.py`)

"Unfolding" a classical iterative solver means turning each iteration
into a neural network layer and training the whole stack end-to-end,
instead of hand-tuning the solver's parameters. Here, the network mimics
`K` iterations of proximal-gradient descent for the same kind of
data-fidelity + prior objective as MAP-TV above, but with a **learned**
prior instead of a fixed TV penalty:

For each stage `k`:

```
x_grad   = x - step_k * (x - y)          # gradient step on ||x - y||^2
residual = Denoiser_k(x_grad)            # learned correction (small conv net)
x        = clamp(x_grad + residual, 0, 1)
```

- `y` is the observed low-light image, fed into *every* stage (not just
  the first) so each stage can re-anchor to the actual observation
  rather than drifting from compounded errors in earlier stages.
- `step_k` is a learned scalar per stage (how big a gradient step to
  take).
- `Denoiser_k` is a small 3-layer conv net (GroupNorm + LeakyReLU) that
  predicts a **residual correction**, not a full replacement image. Its
  final layer is zero-initialized, so at the start of training each
  stage is exactly the gradient step (a safe, colour-correct
  no-op) and training only has to learn an *improvement* on top of that
  - this is what fixes the earlier grayscale/colour-collapse behaviour
  (see the README and the top-of-file comment in `05_deep_unfolding.py`
  for the full story).
- 3 stages are stacked, each seeing `y` again, approximating 3 iterations
  of the solver.

**Training**: `LOLDataset` loads matching low/high pairs from `our485`
and returns random 192x192 crops (same location from both images,
randomly flipped/rotated identically) - cropping avoids distorting
aspect ratio and gives more variety per epoch out of a small (485-image)
dataset while keeping every batch a fixed tensor size; the flip/rotation
augmentation multiplies that variety further at negligible cost. A
small slice of `our485` (~8%) is held out as a validation set.

Trained with Adam + cosine LR schedule, for up to 60 epochs by default,
with a loss weighted toward MSE (`0.2 * L1 + 1.0 * MSE`) rather than
pure L1 - since PSNR is `10*log10(1/MSE)`, more directly optimizing MSE
pushes PSNR up more than L1 alone does (L1 is kept as a smaller term for
edge-stability/robustness). After every epoch the model is scored on the
held-out validation crops, and the checkpoint is only overwritten if
that epoch's validation PSNR is the best one seen so far - so the final
saved checkpoint is the best-generalizing one across the whole run, not
just whatever the last epoch happened to produce. On CUDA GPUs, training
uses automatic mixed precision (`torch.amp`) to keep the larger
crops/batches practical time-wise.

The trained weights are saved to `results/deep_unfolding_model.pt` and
reused on every subsequent run unless you pass `--retrain`. If the
checkpoint's architecture (stage/feature counts) doesn't match the
current code, it's treated as absent (with a message telling you to
retrain) rather than crashing.

**Inference**: fully convolutional, so it runs on the full-resolution
eval (or uploaded) image directly, no cropping needed at test time.

## Evaluation (`06_metrics.py`)

- **PSNR** (Peak Signal-to-Noise Ratio): pixel-level fidelity, in dB,
  higher is better. Sensitive to any pixel-value differences, including
  ones a human wouldn't notice.
- **SSIM** (Structural Similarity Index): compares local luminance,
  contrast and structure rather than raw pixel differences, generally
  correlates better with perceived quality. Range [-1, 1], higher is
  better.

Both are computed with `data_range=1.0` (images are normalized to
`[0, 1]` first) so the numbers aren't accidentally skewed by
version-dependent auto-detected ranges. If one image is grayscale and
the other color (e.g. Histogram Equalization vs. a color ground truth),
the color one is converted to grayscale first so a comparison is still
possible instead of being skipped.
