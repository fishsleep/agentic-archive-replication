# Research Findings: Visual Ambiguity vs. Epistemic Void

## 1. Global Correlation Analysis
- **Spearman Correlation (MSE vs Completeness):** -0.0959
- **Spearman Correlation (MSE vs Semantic Density):** 0.0537

## 2. Segmented Correlation (Within Datasets)
- **Europeana:** Insufficient variation for correlation
- **Tzigara:** 0.0697
- **Wikimedia:** 0.3227

## 3. Statistical Significance (Pairwise)
- **Europeana vs Tzigara:** MWU p=8.3164e-08, Welch's t p=1.2697e-07
- **Europeana vs Wikimedia:** MWU p=5.4854e-04, Welch's t p=3.9926e-04
- **Tzigara vs Wikimedia:** MWU p=7.6340e-01, Welch's t p=7.7588e-01

## 4. MSE Summary by Archive
| Source | n | Mean MSE | Std MSE | Mean Completeness |
| :--- | :--- | :--- | :--- | :--- |
| Europeana | 72 | 0.5723 | 0.2823 | 0.264 |
| Tzigara | 569 | 0.7753 | 0.2890 | 0.018 |
| Wikimedia | 49 | 0.7632 | 0.2814 | 0.400 |

## 5. Outlier Analysis (Tzigara Highest MSE)
| Path | MSE | Title | Density |
| :--- | :--- | :--- | :--- |
| 14-3.jpg | 1.9435 | (no title) | 0 |
| 13-2.jpg | 1.7815 | (no title) | 0 |
| 9-26.jpg | 1.7782 | (no title) | 0 |
| 9-9.jpg | 1.7351 | (no title) | 0 |
| 14-2.jpg | 1.6871 | (no title) | 0 |