import pandas as pd
import numpy as np
import os

def main():
    base_dir = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets")
    analysis_dir = os.path.join(base_dir, "analysis")
    
    datasets = ["wikimedia", "tzigara", "europeana"]
    all_joined_dfs = []

    for name in datasets:
        print(f"--- Processing {name} ---")
        mse_path = os.path.join(analysis_dir, f"{name}_mae_mse.csv")
        meta_path = os.path.join(base_dir, name, "metadata.csv")
        
        if not os.path.exists(mse_path):
            print(f"Warning: MSE file not found: {mse_path}")
            continue
        
        if not os.path.exists(meta_path):
            print(f"Warning: Metadata file not found: {meta_path}")
            continue

        # Load datasets
        mse_df = pd.read_csv(mse_path)
        meta_df = pd.read_csv(meta_path)

        # Fix path mismatch for Wikimedia (thumbnails -> images)
        if name == "wikimedia":
            meta_df['local_path'] = meta_df['local_path'].str.replace("/thumbnails/", "/images/", regex=False)

        # 1. Calculate Completeness for Metadata
        cols_to_check = ['title', 'description', 'creator', 'date', 'license']
        cols_to_check = [c for c in cols_to_check if c in meta_df.columns]
        
        temp_meta = meta_df[cols_to_check].replace(r'^\s*$', np.nan, regex=True)
        meta_df['completeness'] = temp_meta.notna().sum(axis=1) / len(cols_to_check)

        # 2. Join on local_path
        # mse_df has 'path', meta_df has 'local_path'
        joined_df = pd.merge(mse_df, meta_df, left_on='path', right_on='local_path', how='inner')

        if joined_df.empty:
            print(f"Warning: Joined dataframe for {name} is empty. Check if paths match.")
            print(f"MSE path sample: {mse_df['path'].iloc[0] if not mse_df.empty else 'Empty'}")
            print(f"Meta local_path sample: {meta_df['local_path'].iloc[0] if not meta_df.empty else 'Empty'}")
            continue

        # Add source column for aggregation
        joined_df['source'] = name
        all_joined_dfs.append(joined_df)

        # 3. Statistical Analysis for this dataset
        correlation = joined_df['mse'].corr(joined_df['completeness'], method='spearman')
        print(f"Samples matched: {len(joined_df)}")
        print(f"Spearman Correlation (MSE vs Completeness): {correlation:.4f}")

    if all_joined_dfs:
        combined_df = pd.concat(all_joined_dfs, ignore_index=True)
        output_path = os.path.join(analysis_dir, "combined_mse_metadata_joined.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"\nCombined dataset saved to: {output_path}")
        
        # Final combined correlation
        total_corr = combined_df['mse'].corr(combined_df['completeness'], method='spearman')
        print(f"Total Spearman Correlation (MSE vs Completeness): {total_corr:.4f}")
    else:
        print("No datasets could be joined. Check MSE and metadata files.")

if __name__ == "__main__":
    main()
