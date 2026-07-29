"""
Streamlit interface for the Low-Light Image Reconstruction project.

Lets a user upload a low-light image (and optionally a matching
ground-truth/well-lit image), runs every method in codes/ on it, shows
the results side by side, and - if a ground truth was provided - scores
each result against it with PSNR / SSIM (reusing codes/06_metrics.py so
the numbers here match what main.py reports).

Run with:
    streamlit run app.py

Notes:
  - Poisson (BM3D) and Sparse Coding (dictionary learning) can take from
    several seconds to a couple of minutes on a full-resolution image on
    CPU. Uncheck them in the sidebar if you just want a quick look.
  - Deep Unfolding only runs if a trained checkpoint already exists
    (train it once with `python codes/05_deep_unfolding.py`). If none is
    found, the app tells you instead of failing.
  - This app does not read/write anything under Dataset/eval15 - it
    works entirely on whatever you upload, so it does not depend on
    config.LOW_IMAGE / HIGH_IMAGE being set correctly.
"""

import os
import sys
import time
import tempfile
import importlib.util

import cv2
import numpy as np
import streamlit as st

CODES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codes")
sys.path.insert(0, CODES_DIR)


def _load_module(filename, module_name):
    filepath = os.path.join(CODES_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected script not found: {filepath}")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def _get_modules():
    """Import codes/*.py once and cache across reruns (Streamlit reruns
    the whole script on every interaction, so avoid re-importing/re-
    loading torch etc. every time)."""
    map_tv_mod = _load_module("02_map_tv.py", "map_tv_mod")
    poisson_mod = _load_module("03_poisson.py", "poisson_mod")
    sparse_mod = _load_module("04_sparse.py", "sparse_mod")
    deep_unfolding_mod = _load_module("05_deep_unfolding.py", "deep_unfolding_mod")
    metrics_mod = _load_module("06_metrics.py", "metrics_mod")
    return map_tv_mod, poisson_mod, sparse_mod, deep_unfolding_mod, metrics_mod


@st.cache_resource(show_spinner=False)
def _get_deep_model(_deep_unfolding_mod):
    """Load the trained checkpoint once and cache it (None if absent)."""
    return _deep_unfolding_mod.load_trained_model()


def _read_uploaded_image(file):
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode the uploaded image.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(page_title="Low-Light Image Reconstruction", layout="wide")
    st.title("Low-Light Image Reconstruction")
    st.caption(
        "Upload a low-light photo and compare classical (MAP-TV, Poisson/BM3D, "
        "Sparse Coding) and learned (Deep Unfolding) reconstruction methods."
    )

    map_tv_mod, poisson_mod, sparse_mod, deep_unfolding_mod, metrics_mod = _get_modules()

    with st.sidebar:
        st.header("Methods to run")
        run_histogram = st.checkbox("Histogram Equalization (baseline)", value=True)
        run_map_tv = st.checkbox("MAP-TV", value=True)
        run_poisson = st.checkbox("Poisson / BM3D (slower)", value=True)
        run_sparse = st.checkbox("Sparse Coding (slower)", value=True)
        run_deep = st.checkbox("Deep Unfolding (needs a trained checkpoint)", value=True)

        st.divider()
        deep_model = _get_deep_model(deep_unfolding_mod)
        if deep_model is None:
            st.warning(
                "No trained Deep Unfolding checkpoint found.\n\n"
                "Train it once with:\n\n"
                "`python codes/05_deep_unfolding.py`\n\n"
                "then reload this page."
            )
        else:
            st.success(f"Deep Unfolding checkpoint loaded from:\n{deep_unfolding_mod.CHECKPOINT_PATH}")

    low_file = st.file_uploader("Low-light image", type=["png", "jpg", "jpeg", "bmp"])
    high_file = st.file_uploader(
        "Ground-truth / well-lit image (optional, enables PSNR & SSIM)",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    if low_file is None:
        st.info("Upload a low-light image to get started.")
        return

    low_rgb = _read_uploaded_image(low_file)
    st.image(low_rgb, caption="Input (low-light)", width=420)

    high_rgb = _read_uploaded_image(high_file) if high_file is not None else None
    if high_rgb is not None and high_rgb.shape[:2] != low_rgb.shape[:2]:
        st.warning(
            "The ground-truth image is a different size than the low-light image "
            f"({high_rgb.shape[:2]} vs {low_rgb.shape[:2]}). Metrics will error unless "
            "these match - make sure you're uploading the corresponding pair."
        )

    if not st.button("Run reconstruction", type="primary"):
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        results = {}
        timings = {}
        gray = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2GRAY)

        if run_histogram:
            with st.spinner("Running Histogram Equalization..."):
                t0 = time.time()
                results["histogram"] = cv2.equalizeHist(gray)
                timings["histogram"] = time.time() - t0

        if run_map_tv:
            with st.spinner("Running MAP-TV..."):
                t0 = time.time()
                out = map_tv_mod.map_tv(low_rgb)
                results["map_tv"] = (out * 255).astype("uint8")
                timings["map_tv"] = time.time() - t0

        if run_poisson:
            with st.spinner("Running Poisson / BM3D denoising (can take a minute)..."):
                t0 = time.time()
                out = poisson_mod.poisson_denoise(low_rgb)
                results["poisson"] = (out * 255).astype("uint8")
                timings["poisson"] = time.time() - t0

        if run_sparse:
            with st.spinner("Running Sparse Coding (dictionary learning, can take a while)..."):
                t0 = time.time()
                gray_f = gray.astype("float64") / 255.0
                out = sparse_mod.sparse_denoise(gray_f)
                results["sparse"] = (out * 255).astype("uint8")
                timings["sparse"] = time.time() - t0

        if run_deep:
            if deep_model is None:
                st.warning("Skipping Deep Unfolding: no trained checkpoint available.")
            else:
                with st.spinner("Running Deep Unfolding..."):
                    t0 = time.time()
                    import torch

                    device = next(deep_model.parameters()).device
                    tensor = (
                        torch.from_numpy(low_rgb.astype("float32") / 255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                    )
                    with torch.no_grad():
                        out_t = deep_model(tensor)
                    out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    results["deep_unfolding"] = (np.clip(out, 0, 1) * 255).astype("uint8")
                    timings["deep_unfolding"] = time.time() - t0

        if not results:
            st.warning("No methods selected - check at least one box in the sidebar.")
            return

        # Save ground truth once, if provided, so compute_metrics can
        # compare it against each saved result on disk.
        high_path = None
        if high_rgb is not None:
            high_path = os.path.join(tmp_dir, "high.png")
            cv2.imwrite(high_path, cv2.cvtColor(high_rgb, cv2.COLOR_RGB2BGR))

        st.subheader("Results")
        score_rows = []
        cols = st.columns(len(results))

        for col, (name, img) in zip(cols, results.items()):
            out_path = os.path.join(tmp_dir, f"{name}.png")
            if img.ndim == 2:
                cv2.imwrite(out_path, img)
            else:
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            with col:
                st.image(img, caption=name, use_container_width=True)
                st.caption(f"{timings[name]:.2f}s")
                with open(out_path, "rb") as f:
                    st.download_button(
                        f"Download",
                        f,
                        file_name=f"{name}.png",
                        key=f"dl_{name}",
                    )

            if high_path is not None:
                try:
                    psnr, ssim = metrics_mod.compute_metrics(high_path, out_path)
                    score_rows.append({"Method": name, "PSNR (dB)": round(psnr, 2), "SSIM": round(ssim, 4)})
                except Exception as e:
                    score_rows.append({"Method": name, "PSNR (dB)": "error", "SSIM": str(e)})

        if score_rows:
            st.subheader("Scores vs. ground truth")
            st.table(score_rows)
        else:
            st.info("Upload a ground-truth image above and rerun to see PSNR / SSIM scores.")


if __name__ == "__main__":
    main()
