import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    input_path = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets/analysis/combined_mse_metadata_joined.csv")
    output_csv_path = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets/analysis/mse_metadata_refined.csv")
    output_plot_path = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets/analysis/metrics_correlation_plot.png")

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)

    # 1. Metric Refinement: Semantic Density (Total word count) and Metadata Length (Total character count)
    TEXT_COLS = [c for c in ['title', 'description'] if c in df.columns]
    def calculate_metrics(row):
        words = 0
        chars = 0
        for col in TEXT_COLS:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                s_val = str(val).strip()
                words += len(s_val.split())
                chars += len(s_val)
        return pd.Series({'semantic_density': words, 'metadata_length': chars})

    df[['semantic_density', 'metadata_length']] = df.apply(calculate_metrics, axis=1)

    # 2. Statistical Analysis
    corr_comp = df['mse'].corr(df['completeness'], method='spearman')
    corr_dens = df['mse'].corr(df['semantic_density'], method='spearman')
    corr_len = df['mse'].corr(df['metadata_length'], method='spearman')

    print("--- Correlation Analysis (Refined) ---")
    print(f"Spearman Correlation (MSE vs Completeness): {corr_comp:.4f}")
    print(f"Spearman Correlation (MSE vs Semantic Density): {corr_dens:.4f}")
    print(f"Spearman Correlation (MSE vs Metadata Length): {corr_len:.4f}")
    print("--------------------------------------")

    # 3. Visualization
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Plot 1: MSE vs Completeness
    sns.scatterplot(data=df, x='completeness', y='mse', hue='source', ax=axes[0])
    axes[0].set_title("MSE vs Metadata Completeness")

    # Plot 2: MSE vs Semantic Density
    sns.scatterplot(data=df, x='semantic_density', y='mse', hue='source', ax=axes[1])
    axes[1].set_title("MSE vs Semantic Density (Word Count)")

    # Plot 3: MSE vs Metadata Length
    sns.scatterplot(data=df, x='metadata_length', y='mse', hue='source', ax=axes[2])
    axes[2].set_title("MSE vs Metadata Length (Char Count)")

    # Plot 4: MSE Distribution by Source
    sns.boxplot(data=df, x='source', y='mse', ax=axes[3])
    axes[3].set_title("MSE Distribution by Dataset")

    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Plot saved to: {output_plot_path}")

    # 4. Save Refined Data
    df.to_csv(output_csv_path, index=False)
    print(f"Refined dataset saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
