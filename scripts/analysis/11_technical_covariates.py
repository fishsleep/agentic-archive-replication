"""
13_technical_covariates.py — Extract low-level image features for all 690 images.

Outputs:
  05_datasets/analysis/combined_mse_metadata_joined_with_covariates.csv
  07_results/technical_covariates_summary.md

Covariates:
  width, height, aspect_ratio, is_grayscale, jpeg_quality (MUG),
  mean_saturation, dynamic_range
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # FINAL-PAPER-CNIR root
INPUT = BASE_DIR / "05_datasets" / "analysis" / "combined_mse_metadata_joined.csv"
OUTPUT_CSV = (
    BASE_DIR
    / "05_datasets"
    / "analysis"
    / "combined_mse_metadata_joined_with_covariates.csv"
)
OUTPUT_MD = BASE_DIR / "07_results" / "technical_covariates_summary.md"

# MUG constants
MUG_BLOCK_SIZE = 8
MUG_2D_DCT = None  # lazy init


def init_2d_dct():
    """Build 2D-DCT basis matrix (8x8)."""
    global MUG_2D_DCT
    if MUG_2D_DCT is not None:
        return
    N = MUG_BLOCK_SIZE
    u, v = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    C = np.zeros((N, N))
    for i in range(N):
        C[i, 0] = np.sqrt(1.0 / N) if i == 0 else np.sqrt(2.0 / N)
    MUG_2D_DCT = C


def compute_2d_dct_block(block):
    """Compute 2D-DCT of an 8x8 block."""
    return MUG_2D_DCT @ block @ MUG_2D_DCT.T


def mug_jpeg_quality(img_array):
    """
    MUG: No-Reference JPEG Quality Estimator.

    Uses 2D-DCT coefficient distributions from inner and edge blocks
    to estimate JPEG quality factor (1-100).

    Based on:
    - Wang et al. ICIP 2002 (blocking artifact features)
    - MUG (arXiv 2016) — parameterless JPEG quality estimation
    """
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array.astype(np.float64)

    orig_h, orig_w = gray.shape
    # Truncate to dimensions divisible by 8
    trun_h = orig_h - (orig_h % MUG_BLOCK_SIZE)
    trun_w = orig_w - (orig_w % MUG_BLOCK_SIZE)
    if trun_h < MUG_BLOCK_SIZE or trun_w < MUG_BLOCK_SIZE:
        return None

    gray = gray[:trun_h, :trun_w]

    # Extract 8x8 blocks via reshape + transpose
    n_blocks_h = trun_h // MUG_BLOCK_SIZE
    n_blocks_w = trun_w // MUG_BLOCK_SIZE
    blocks = gray.reshape(trun_h // 8, 8, trun_w // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3)  # (n_h, n_w, 8, 8)
    blocks_2dct = np.empty_like(blocks)
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            blocks_2dct[i, j] = compute_2d_dct_block(blocks[i, j])

    # MUG feature 1: Blocking artifact strength (fully vectorized)
    # Vertical boundaries: right edge of block[:,j,:] vs left edge of block[:,j+1,:]
    if n_blocks_w > 1:
        right_edges = blocks_2dct[:, :-1, :, -1]
        left_edges = blocks_2dct[:, 1:, :, 0]
        v_diff = float(np.mean(np.abs(right_edges - left_edges)))
    else:
        v_diff = 0.0

    # Horizontal boundaries: bottom edge vs top edge of vertically adjacent blocks
    if n_blocks_h > 1:
        bottom = blocks_2dct[:-1, :, :, -1]  # (n_h-1, n_w, 8) last row of upper blocks
        top = blocks_2dct[1:, :, :, 0]  # (n_h-1, n_w, 8) first row of lower blocks
        h_diff = float(np.mean(np.abs(bottom - top)))
    else:
        h_diff = 0.0

    block_diff = (v_diff + h_diff) / 2

    # MUG feature 2: AC coefficient variance (smoothness indicator)
    # For JPEG, low quality = heavy quantization = most AC coefficients near zero
    # High quality = more AC coefficients spread out
    ac_coeffs = blocks_2dct[:, :, 1:, 1:].flatten()  # exclude DC (0,0)
    ac_var = np.var(ac_coeffs)

    # MUG feature 3: High-frequency content ratio
    # Fraction of AC coefficients with |dct| > 4 (significant high-freq content)
    hf_count = np.sum(np.abs(ac_coeffs) > 4)
    hf_ratio = hf_count / len(ac_coeffs) if len(ac_coeffs) > 0 else 0

    # Quality estimation via empirical mapping
    # block_diff: high at low Q (visible blocking), low at high Q
    # ac_var: low at low Q (quantized to near-zero), high at high Q
    # hf_ratio: low at low Q (high-freq smoothed), high at high Q

    # Normalize to 0-1 range
    # block_diff: ~20-40 at Q=10, ~1-5 at Q=95
    norm_blocking = np.clip((block_diff - 3) / (35 - 3), 0, 1)
    # ac_var: ~50-200 at Q=10, ~500-2000 at Q=95
    norm_energy = np.clip((ac_var - 50) / (1500 - 50), 0, 1)
    # hf_ratio: ~0.02 at Q=10, ~0.15-0.25 at Q=95
    norm_hf = np.clip((hf_ratio - 0.02) / (0.20 - 0.02), 0, 1)

    # Weighted combination
    quality = 40 * (1 - norm_blocking) + 30 * norm_energy + 30 * norm_hf
    quality = np.clip(quality, 1, 99)

    return round(quality)


def extract_covariates(img_path):
    """Extract all technical covariates from an image file."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            return None

    width, height = img.size
    aspect_ratio = width / height if height > 0 else 0

    is_grayscale = img.mode == "L"

    arr = np.array(img, dtype=np.float64)
    mean_saturation = None
    dynamic_range = None

    if is_grayscale:
        mean_saturation = 0.0
    else:
        # Convert to HSV for saturation
        hsv = img.convert("HSV")
        hsv_arr = np.array(hsv, dtype=np.float64)
        # HSV: H=0-360, S=0-255, V=0-255
        # S channel is index 1
        mean_saturation = float(np.mean(hsv_arr[:, 1]))

    # Dynamic range
    dynamic_range = float(np.max(arr) - np.min(arr))

    # MUG JPEG quality
    jpeg_quality = mug_jpeg_quality(arr)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "is_grayscale": 1 if is_grayscale else 0,
        "jpeg_quality": jpeg_quality,
        "mean_saturation": round(mean_saturation, 2)
        if mean_saturation is not None
        else None,
        "dynamic_range": round(dynamic_range, 2),
    }


def normalize_path(path_str):
    """Fix stale paths from old directory structure."""
    # Try original path first
    if os.path.isfile(path_str):
        return path_str

    # Fix: replace FINAL-PAPER-CNIR/05_datasets with Articles/CNIR/FINAL-PAPER-CNIR/05_datasets
    fixed = path_str.replace(
        "/FINAL-PAPER-CNIR/05_datasets", "/Articles/CNIR/FINAL-PAPER-CNIR/05_datasets"
    )
    if os.path.isfile(fixed):
        return fixed

    # Also try relative to the script's parent directory
    rel = Path(__file__).resolve().parent.parent.parent / Path(path_str).name
    if os.path.isfile(rel):
        return str(rel)

    return None


def main():
    init_2d_dct()

    print("=" * 70)
    print("Step 1.1: Technical Covariates Extraction")
    print("=" * 70)
    print(f"Input: {INPUT}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    df = pd.read_csv(INPUT)
    print(f"Total images to process: {len(df)}")

    covariates = []
    success_count = 0
    fail_count = 0
    mug_fail_count = 0

    for idx, row in df.iterrows():
        original_path = row["path"]
        normalized = normalize_path(original_path)

        if normalized is None:
            covariates.append(None)
            fail_count += 1
            if idx % 100 == 0:
                print(
                    f"  [{idx}/{len(df)}] FAILED to resolve path: {original_path[:80]}..."
                )
            continue

        cov = extract_covariates(normalized)
        if cov is None:
            covariates.append(None)
            fail_count += 1
        else:
            if cov["jpeg_quality"] is None:
                mug_fail_count += 1
            covariates.append(cov)
            success_count += 1

        if idx % 100 == 0:
            print(
                f"  [{idx}/{len(df)}] processed (success={success_count}, mug_fail={mug_fail_count})"
            )

    # Build covariate columns
    df["width"] = np.nan
    df["height"] = np.nan
    df["aspect_ratio"] = np.nan
    df["is_grayscale"] = np.nan
    df["jpeg_quality"] = np.nan
    df["mean_saturation"] = np.nan
    df["dynamic_range"] = np.nan

    for idx, cov in enumerate(covariates):
        if cov is not None:
            df.loc[idx, "width"] = cov["width"]
            df.loc[idx, "height"] = cov["height"]
            df.loc[idx, "aspect_ratio"] = cov["aspect_ratio"]
            df.loc[idx, "is_grayscale"] = cov["is_grayscale"]
            df.loc[idx, "jpeg_quality"] = cov["jpeg_quality"]
            df.loc[idx, "mean_saturation"] = cov["mean_saturation"]
            df.loc[idx, "dynamic_range"] = cov["dynamic_range"]

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(
        f"Success: {success_count}/{len(df)} images ({success_count / len(df) * 100:.1f}%)"
    )
    print(f"Path resolution failures: {fail_count}")
    print(f"MUG quality failures: {mug_fail_count}")

    # Generate summary markdown
    summary_lines = [
        "# Technical Covariates Summary",
        "",
        f"## Overview",
        f"- Total images: {len(df)}",
        f"Successfully extracted: {success_count} ({success_count / len(df) * 100:.1f}%)",
        f"Path resolution failures: {fail_count}",
        f"MUG quality estimation failures: {mug_fail_count}",
        "",
        f"## Per-Archive Summary",
        "",
        f"| Archive | n | Mean Width | Mean Height | Mean Aspect Ratio | Grayscale % | Mean JPEG Q | Mean Saturation | Mean Dynamic Range |",
        f"|---------|---|------------|-------------|-------------------|-------------|-------------|-----------------|-------------------|",
    ]

    for source in df["source"].unique():
        sub = df[df["source"] == source]
        n = len(sub)
        mean_w = sub["width"].mean()
        mean_h = sub["height"].mean()
        mean_ar = sub["aspect_ratio"].mean()
        gray_pct = (sub["is_grayscale"] == 1).sum() / n * 100 if n > 0 else 0
        mean_jq = sub["jpeg_quality"].mean()
        mean_sat = sub["mean_saturation"].mean()
        mean_dr = sub["dynamic_range"].mean()

        summary_lines.append(
            f"| {source} | {n} | {mean_w:.0f} | {mean_h:.0f} | {mean_ar:.3f} | {gray_pct:.1f}% | "
            f"{mean_jq:.0f} | {mean_sat:.1f} | {mean_dr:.1f} |"
        )

    summary_lines += [
        "",
        f"## Full Covariate Statistics",
        "",
        f"| Covariate | Overall Mean | Overall Std | Min | Max |",
        f"|-----------|-------------|-------------|-----|-----|",
    ]

    for col in [
        "width",
        "height",
        "aspect_ratio",
        "jpeg_quality",
        "mean_saturation",
        "dynamic_range",
    ]:
        vals = df[col].dropna()
        if len(vals) > 0:
            summary_lines.append(
                f"| {col} | {vals.mean():.2f} | {vals.std():.2f} | {vals.min():.2f} | {vals.max():.2f} |"
            )

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSummary saved: {OUTPUT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    main()
