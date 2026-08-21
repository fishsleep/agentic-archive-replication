import pandas as pd
import numpy as np
from scipy import stats
import os

def main():
    input_path = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets/analysis/mse_metadata_refined.csv")
    report_path = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets/analysis/summary_report.md")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    report_lines = [
        "# Research Findings: Visual Ambiguity vs. Epistemic Void\n",
        "## 1. Global Correlation Analysis",
        f"- **Spearman Correlation (MSE vs Completeness):** {df['mse'].corr(df['completeness'], method='spearman'):.4f}",
        f"- **Spearman Correlation (MSE vs Semantic Density):** {df['mse'].corr(df['semantic_density'], method='spearman'):.4f}\n",
        "## 2. Segmented Correlation (Within Datasets)",
    ]

    for source in ['europeana', 'tzigara', 'wikimedia']:
        sub_df = df[df['source'] == source]
        if len(sub_df) > 1 and sub_df['semantic_density'].nunique() > 1:
            corr = sub_df['mse'].corr(sub_df['semantic_density'], method='spearman')
            report_lines.append(f"- **{source.capitalize()}:** {corr:.4f}")
        else:
            report_lines.append(f"- **{source.capitalize()}:** Insufficient variation for correlation")
    
    report_lines.append("\n## 3. Statistical Significance (Pairwise)")
    sources = {s: df[df['source'] == s]['mse'] for s in ['europeana', 'tzigara', 'wikimedia']}
    for a, b in [('europeana', 'tzigara'), ('europeana', 'wikimedia'), ('tzigara', 'wikimedia')]:
        u_stat, p_mw = stats.mannwhitneyu(sources[a], sources[b], alternative='two-sided')
        t_stat, p_t = stats.ttest_ind(sources[a], sources[b], equal_var=False)
        report_lines.append(f"- **{a.capitalize()} vs {b.capitalize()}:** MWU p={p_mw:.4e}, Welch's t p={p_t:.4e}")

    report_lines.append("\n## 4. MSE Summary by Archive")
    report_lines.append("| Source | n | Mean MSE | Std MSE | Mean Completeness |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for s in ['europeana', 'tzigara', 'wikimedia']:
        sub = df[df['source'] == s]
        report_lines.append(f"| {s.capitalize()} | {len(sub)} | {sub['mse'].mean():.4f} | {sub['mse'].std():.4f} | {sub['completeness'].mean():.3f} |")

    report_lines.append("\n## 5. Outlier Analysis (Tzigara Highest MSE)")
    top_outliers = df[df['source'] == 'tzigara'].nlargest(5, 'mse')
    report_lines.append("| Path | MSE | Title | Density |")
    report_lines.append("| :--- | :--- | :--- | :--- |")
    for _, row in top_outliers.iterrows():
        path_simple = os.path.basename(row['path'])
        title_val = row['title'] if pd.notna(row['title']) else '(no title)'
        report_lines.append(f"| {path_simple} | {row['mse']:.4f} | {title_val} | {row['semantic_density']} |")

    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"Comprehensive analysis complete.")
    print(f"Report saved to: {report_path}")
    print("\n--- MSE by Archive ---")
    for s in ['europeana', 'tzigara', 'wikimedia']:
        sub = df[df['source'] == s]
        print(f"{s}: n={len(sub)}, MSE={sub['mse'].mean():.4f}±{sub['mse'].std():.4f}, completeness={sub['completeness'].mean():.3f}")
    print("\n--- Top 5 Tzigara Outliers ---")
    print(top_outliers[['path', 'mse', 'title', 'semantic_density']].to_string(index=False))

if __name__ == "__main__":
    main()
