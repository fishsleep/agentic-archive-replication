# Agentic Archive Replication Package

**Version:** v17c (post-full-panel-review)

Replication package for the CNIR 2026 paper:  
"The Agentic Archive: Mapping Archival Silence in Romanian Visual Heritage"

## Purpose

This repository contains all data, scripts, and pre-trained models needed to replicate the experiments in the paper. It has been intentionally stripped of non-essential documentation (session logs, peer review reports, paper drafts) — only replication-critical files remain.

## Directory Structure

```
agentic-archive-replication/
├── scripts/
│   ├── analysis/                        # 13 numbered experiment scripts (run in order)
│   │   ├── 01_h1_epistemic_void_analysis.py
│   │   ├── 02_mae_inference.py          # MAE forward pass → reconstruction MSE
│   │   ├── 03_correlate_mse_metadata.py   # Spearman ρ, Mann-Whitney, Welch t
│   │   ├── 04_lora_mae_training.py
│   │   ├── 05_visualize_and_refine_metrics.py  # Plot generation
│   │   ├── 06_statistical_tests.py      # Bootstrap, Bonferroni correction
│   │   ├── 07_comprehensive_analysis.py
│   │   ├── 08_lora_adaptation.py        # Conservative LoRA (r=8, 5 epochs)
│   │   ├── 09_lora_adaptation_extended.py  # Extended LoRA (r=64, 50 epochs)
│   │   ├── 10_mixed_effects_model.py    # ICC, random-intercept models
│   │   ├── 11_technical_covariates.py   # JPEG quality, saturation, dynamic range
│   │   ├── 12_mixed_effects_with_covariates.py  # Mediation analysis
│   │   └── 13_lora_analytical_cis.py    # Analytical CIs for ΔMSE
│   └── deterministic_checks.py          # Integrity verification script
├── data/
│   ├── raw_images/
│   │   ├── europeana_metadata.csv
│   │   ├── tzigara_metadata.csv
│   │   └── wikimedia_metadata.csv
│   ├── scores/
│   │   ├── mae_reconstruction_mse.csv
│   │   ├── combined_mse_metadata_joined.csv
│   │   ├── lora_results.json
│   │   ├── lora_results_extended.json
│   │   ├── mse_boxplot.png
│   │   ├── metrics_correlation_plot.png
│   │   └── summary_report.md
│   └── loras/
│       ├── conservative/                # r=8, α=16, 5 epochs
│       │   ├── europeana/
│       │   ├── tzigara/
│       │   └── wikimedia/
│       └── extended/                    # r=64, α=128, 50 epochs, augmentation
│           ├── europeana/
│           ├── tzigara/
│           └── wikimedia/
├── models/
│   └── model_urls.md
├── environment.yml
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── LICENSE_DATA
└── README.md
```

## Quick Start

```bash
git clone https://github.com/fishsleep/agentic-archive-replication.git
cd agentic-archive-replication

# Environment (conda)
conda env create -f environment.yml
conda activate agentic-archive

# Or via pip
pip install -r requirements.txt

# Run analysis (scripts must be run in order 01→13)
python scripts/analysis/02_mae_inference.py
python scripts/analysis/06_statistical_tests.py
python scripts/analysis/08_lora_adaptation.py

# Verify integrity
python scripts/deterministic_checks.py
```

## Reproduction Workflow

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `01_h1_epistemic_void_analysis.py` | H1 epistemic void analysis | `summary_report.md` |
| 2 | `02_mae_inference.py` | MAE forward pass → reconstruction MSE | `*_mae_mse.csv` |
| 3 | `03_correlate_mse_metadata.py` | Spearman, Mann-Whitney, Welch t | `mse_metadata_joined.csv` |
| 4 | `04_lora_mae_training.py` | LoRA training pipeline | Trained adapters |
| 5 | `05_visualize_and_refine_metrics.py` | Plot generation | `mse_boxplot.png` |
| 6 | `06_statistical_tests.py` | Bootstrap, Bonferroni | `metrics_correlation_plot.png` |
| 7 | `07_comprehensive_analysis.py` | Combined cross-archive analysis | `combined_mse_metadata_joined.csv` |
| 8 | `08_lora_adaptation.py` | Conservative LoRA (r=8, 5 epochs) | `lora_results.json` |
| 9 | `09_lora_adaptation_extended.py` | Extended LoRA (r=64, 50 epochs) | `lora_results_extended.json` |
| 10 | `10_mixed_effects_model.py` | Random-intercept models | `mixed_effects_summary.md` |
| 11 | `11_technical_covariates.py` | JPEG quality, saturation, DR | `technical_covariates_summary.md` |
| 12 | `12_mixed_effects_with_covariates.py` | Mediation analysis | `mixed_effects_with_covariates.md` |
| 13 | `13_lora_analytical_cis.py` | Analytical CIs for ΔMSE | `lora_analytical_cis.md` |

**Expected results:**
- Europeana MSE (mean 0.572) decisively lower than Tzigara (0.775) and Wikimedia (0.763)
- Tzigara vs. Wikimedia: p = 0.76 (statistically indistinguishable)
- LoRA ΔMSE ≈ 0 (all 9 cells, both experiments)

## Image Acquisition (User Responsibility)

**Images are NOT included in this repository.** Users are responsible for obtaining their own image datasets and verifying licensing requirements with source institutions.

- **Europeana images**: Obtain via Europeana API (mixed licenses; CC-BY where applicable)
- **Tzigara-Samurcaș images**: Requires institutional access to INP/UAIM
- **Wikimedia Commons images**: Obtain via Wikimedia API (varies by individual file license)

Expected input format: directory of JPEG images with matching metadata CSV containing columns: `image_id`, `archive`, `mse`, `completeness`, `semantic_density`, `metadata_length`.

## Citation

```bibtex
@article{agentic-archive-2026,
  title={The Agentic Archive: Mapping Archival Silence in Romanian Visual Heritage},
  journal={Proceedings of CNIR 2026},
  year={2026}
}
```

## License

- Code: MIT License
- Metadata & analysis results: CC-BY 4.0 (see LICENSE_DATA)
- Pre-trained LoRA adapters: MIT
- Model checkpoint: Apache 2.0 (facebook/vit-mae-base)

**Important:** No images are redistributed with this repository. Users must obtain their own image datasets and verify licensing requirements with source institutions.