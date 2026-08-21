import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import os

BASE_DIR = os.path.expanduser("~/Documents/My Projects/Things to Do/Articles/CNIR/FINAL-PAPER-CNIR")
INPUT_COVARIATES = os.path.join(
    BASE_DIR, "05_datasets/analysis/combined_mse_metadata_joined_with_covariates.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "07_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fit_mixed_model_api(exog, endog, groups, const=True):
    """Fit mixed-effects model using API (avoids patsy formula parsing issues)."""
    if const:
        exog = sm.add_constant(exog)
    model = MixedLM(endog, exog, groups)
    try:
        return model.fit(re_stratified=False, maxiter=2000)
    except Exception:
        return model.fit()


def main():
    df = pd.read_csv(INPUT_COVARIATES)

    covariates = [
        "completeness",
        "jpeg_quality",
        "width",
        "height",
        "aspect_ratio",
        "mean_saturation",
        "dynamic_range",
    ]

    # ============================================================
    # Baseline model (completeness only)
    # ============================================================
    print("=" * 70)
    print("P1 #2a: Baseline Mixed-Effects Model (completeness only)")
    print("=" * 70)
    print(f"Total observations: {len(df)}")
    for s in ["europeana", "tzigara", "wikimedia"]:
        sub = df[df["source"] == s]
        print(
            f"  {s}: n={len(sub)}, MSE={sub.mse.mean():.4f}, "
            f"completeness={sub.completeness.mean():.4f}"
        )
    print()

    model_base = fit_mixed_model_api(df[["completeness"]], df["mse"], df["source"])
    comp_base = model_base.fe_params["completeness"]
    p_base = model_base.pvalues["completeness"]
    se_base = model_base.bse["completeness"]
    z_base = comp_base / se_base
    icc_base = model_base.cov_re.iloc[0, 0] / (
        model_base.cov_re.iloc[0, 0] + model_base.scale
    )
    between_base = model_base.cov_re.iloc[0, 0]
    within_base = model_base.scale

    print(
        f"  completeness: coef={comp_base:.4f} (SE={se_base:.4f}, z={z_base:.2f}, p={p_base:.4f})"
    )
    print(f"  ICC: {icc_base:.1%}")
    print(f"  Between-archive variance: {between_base:.6f}")
    print(f"  Residual variance: {within_base:.6f}")
    print()

    # ============================================================
    # Extended model (completeness + technical covariates)
    # ============================================================
    print("=" * 70)
    print("P1 #2b: Extended Model (completeness + technical covariates)")
    print("=" * 70)

    exog_ext = df[covariates]
    model_ext = fit_mixed_model_api(exog_ext, df["mse"], df["source"])

    print("Fixed effects:")
    print(f"| Parameter | Estimate | Std.Err. | z | P>|z| |")
    print(f"|-----------|----------|----------|-----|-------|")
    param_names = [
        "const",
        "completeness",
        "jpeg_quality",
        "width",
        "height",
        "aspect_ratio",
        "mean_saturation",
        "dynamic_range",
    ]
    for name in param_names:
        if name in model_ext.fe_params:
            est = model_ext.fe_params[name]
            se = model_ext.bse[name]
            pval = model_ext.pvalues[name]
            z = est / se if se > 0 else 0
            print(f"| {name} | {est:.4f} | {se:.4f} | {z:.2f} | {pval:.4f} |")
    print()

    icc_ext = model_ext.cov_re.iloc[0, 0] / (
        model_ext.cov_re.iloc[0, 0] + model_ext.scale
    )
    between_ext = model_ext.cov_re.iloc[0, 0]
    within_ext = model_ext.scale

    print("Random Effects (Extended)")
    print(f"| Component | Variance | % of Total |")
    print(f"|-----------|----------|------------|")
    print(f"| Between-archive | {between_ext:.6f} | {icc_ext:.1%} |")
    print(f"| Residual | {within_ext:.6f} | {(1 - icc_ext):.1%} |")
    print(f"| **ICC** | | **{icc_ext:.1%}** |")
    print()

    # ============================================================
    # Completeness effect comparison
    # ============================================================
    comp_ext = model_ext.fe_params.get("completeness", 0)
    se_ext = model_ext.bse.get("completeness", 0)
    p_ext = model_ext.pvalues.get("completeness", 1.0)
    z_ext = comp_ext / se_ext if se_ext > 0 else 0

    print("## Completeness Effect: Baseline vs Extended")
    print(f"| Metric | Baseline | Extended |")
    print(f"|--------|----------|----------|")
    print(f"| Coefficient | {comp_base:.4f} | {comp_ext:.4f} |")
    print(f"| Std.Err. | {se_base:.4f} | {se_ext:.4f} |")
    print(f"| z | {z_base:.2f} | {z_ext:.2f} |")
    print(f"| p-value | {p_base:.4f} | {p_ext:.4f} |")
    print(f"| ICC | {icc_base:.1%} | {icc_ext:.1%} |")
    print()

    # Interpretation
    if p_ext < 0.05:
        interp = (
            f"**Significant:** Completeness is a significant predictor of MSE "
            f"after adjusting for technical covariates (p={p_ext:.4f})."
        )
    elif p_ext < 0.10:
        interp = (
            f"Marginally significant: Completeness shows a marginal predictive "
            f"relationship with MSE after adjusting for technical covariates "
            f"(p={p_ext:.4f})."
        )
    else:
        interp = (
            f"**Not significant:** Completeness remains a non-significant "
            f"predictor of MSE after adjusting for technical covariates "
            f"(p={p_ext:.4f})."
        )
    icc_delta = (icc_ext - icc_base) * 100

    print(f"> **Interpretation:** {interp}")
    print(
        f" Technical covariates explain {icc_delta:+.1f} percentage points of "
        f"between-archive variance relative to the baseline model.\n"
    )

    # ============================================================
    # Covariate significance summary
    # ============================================================
    print("## Covariate Significance")
    sig_cols = []
    non_sig_cols = []
    for name in [
        "jpeg_quality",
        "width",
        "height",
        "aspect_ratio",
        "mean_saturation",
        "dynamic_range",
    ]:
        if name in model_ext.pvalues:
            pval = model_ext.pvalues[name]
            est = model_ext.fe_params[name]
            if pval < 0.05:
                sig_cols.append(f"**{name}**: {est:.4f} (p={pval:.4e})")
            elif pval < 0.10:
                sig_cols.append(f"marginal {name}: {est:.4f} (p={pval:.4f})")
            else:
                non_sig_cols.append(f"{name}: {est:.4f} (p={pval:.4f})")

    if sig_cols:
        print("Significant predictors (p<0.05):")
        for s in sig_cols:
            print(f"  - {s}")
        print()
    if non_sig_cols:
        print("Non-significant predictors:")
        for n in non_sig_cols:
            print(f"  - {n}")
        print()

    # ============================================================
    # Per-archive residual analysis
    # ============================================================
    print("## Per-Archive Residual Analysis (Extended Model)")
    fitted = model_ext.fittedvalues
    residuals = df["mse"] - fitted
    for s in ["europeana", "tzigara", "wikimedia"]:
        sub = df[df["source"] == s]
        sub_res = residuals[sub.index]
        print(
            f"  {s}: mean_res={sub_res.mean():.4f}, "
            f"std_res={sub_res.std():.4f}, n={len(sub)}"
        )
    print()

    # ============================================================
    # Write report
    # ============================================================
    report = []
    report.append("# Mixed-Effects Model with Technical Covariates\n")

    report.append("## Models Compared\n")
    report.append(f"| Model | Formula | Observations |")
    report.append(f"|-------|---------|-------------|")
    report.append(f"| Baseline | MSE ~ completeness + (1 | source) | {len(df)} |")
    report.append(
        f"| Extended | MSE ~ completeness + jpeg_quality + width + height + "
        f"aspect_ratio + mean_saturation + dynamic_range + (1 | source) | {len(df)} |"
    )
    report.append("")

    report.append("## Extended Model Fixed Effects\n")
    report.append("| Parameter | Estimate | Std.Err. | z | P>|z| |")
    report.append("|-----------|----------|----------|-----|-------|")
    for name in param_names:
        if name in model_ext.fe_params:
            est = model_ext.fe_params[name]
            se = model_ext.bse[name]
            pval = model_ext.pvalues[name]
            z = est / se if se > 0 else 0
            report.append(f"| {name} | {est:.4f} | {se:.4f} | {z:.2f} | {pval:.4f} |")
    report.append("")

    report.append("## Random Effects\n")
    report.append("| Component | Variance | % of Total |")
    report.append("|-----------|----------|------------|")
    report.append(f"| Between-archive | {between_ext:.6f} | {icc_ext:.1%} |")
    report.append(f"| Residual | {within_ext:.6f} | {(1 - icc_ext):.1%} |")
    report.append(f"| **ICC** | | **{icc_ext:.1%}** |")
    report.append("")

    report.append("## Completeness Effect: Baseline vs Extended\n")
    report.append("| Metric | Baseline | Extended |")
    report.append("|--------|----------|----------|")
    report.append(f"| Coefficient | {comp_base:.4f} | {comp_ext:.4f} |")
    report.append(f"| Std.Err. | {se_base:.4f} | {se_ext:.4f} |")
    report.append(f"| z | {z_base:.2f} | {z_ext:.2f} |")
    report.append(f"| p-value | {p_base:.4f} | {p_ext:.4f} |")
    report.append(f"| ICC | {icc_base:.1%} | {icc_ext:.1%} |")
    report.append("")
    report.append(f"> **Interpretation:** {interp}")
    report.append(
        f" Technical covariates explain {icc_delta:+.1f} percentage points of "
        f"between-archive variance relative to the baseline model.\n"
    )

    if sig_cols:
        report.append("### Significant Predictors (p<0.05)\n")
        for s in sig_cols:
            report.append(f"- {s}")
        report.append("")
    if non_sig_cols:
        report.append("### Non-Significant Predictors\n")
        for n in non_sig_cols:
            report.append(f"- {n}")
        report.append("")

    report.append("## Per-Archive Residuals (Extended Model)\n")
    report.append("| Archive | Mean Residual | Std Residual | n |")
    report.append("|---------|-------------|-------------|---|")
    for s in ["europeana", "tzigara", "wikimedia"]:
        sub = df[df["source"] == s]
        sub_res = residuals[sub.index]
        report.append(
            f"| {s} | {sub_res.mean():.4f} | {sub_res.std():.4f} | {len(sub)} |"
        )
    report.append("")

    with open(os.path.join(OUTPUT_DIR, "mixed_effects_with_covariates.md"), "w") as f:
        f.write("\n".join(report))

    print(
        f"Report saved to: {os.path.join(OUTPUT_DIR, 'mixed_effects_with_covariates.md')}"
    )


if __name__ == "__main__":
    main()
