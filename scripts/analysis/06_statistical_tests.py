import pandas as pd
from scipy import stats
import os

def main():
    input_path = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets/analysis/mse_metadata_refined.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)

    # Separate the MSE values by source
    sources = {}
    for s in ['europeana', 'tzigara', 'wikimedia']:
        sources[s] = df[df['source'] == s]['mse']

    print("--- MSE Summary Statistics ---")
    for s, vals in sources.items():
        print(f"{s}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}")
    print("-----------------------------")

    # Pairwise comparisons (Europeana vs Tzigara, Europeana vs Wikimedia, Tzigara vs Wikimedia)
    pairs = [('europeana', 'tzigara'), ('europeana', 'wikimedia'), ('tzigara', 'wikimedia')]
    for a, b in pairs:
        print(f"\n--- {a.capitalize()} vs {b.capitalize()} ---")
        u_stat, p_val_mw = stats.mannwhitneyu(sources[a], sources[b], alternative='two-sided')
        t_stat, p_val_t = stats.ttest_ind(sources[a], sources[b], equal_var=False)
        print(f"Mann-Whitney U:  stat={u_stat:.2f}, p={p_val_mw:.4e} {'SIG' if p_val_mw < 0.05 else 'n.s.'}")
        print(f"Welch's t-test:  stat={t_stat:.4f}, p={p_val_t:.4e} {'SIG' if p_val_t < 0.05 else 'n.s.'}")
    print("-----------------------------")

if __name__ == "__main__":
    main()
