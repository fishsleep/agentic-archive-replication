import pandas as pd
import os

def analyze_h1():
    base_dir = os.path.expanduser("~/Documents/My Projects/Things to Do/project 6/new dataset/data")
    output_dir = os.path.join(base_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Define common schema
    schema = ['source', 'id', 'title', 'original_title', 'description', 'creator', 'date', 'condition', 'license']

    def load_and_map(file_path, source_name, mapping):
        df = pd.read_csv(file_path)
        # Create a new dataframe with the common schema
        mapped_df = pd.DataFrame(columns=schema)
        
        # Populate the mapped columns first
        for target_col, source_col in mapping.items():
            if source_col in df.columns:
                mapped_df[target_col] = df[source_col]
            elif source_col is not None:
                # If the source_col is not in df, but we expect it, it's a void
                mapped_df[target_col] = None
        
        # Add the source column after rows are established via assignment
        mapped_df['source'] = source_name
        
        return mapped_df[schema]

    # Mappings
    # Europeana
    # id,title,description,creator,year,completeness,country,dataProvider,edmPreview,imageWidth,imageHeight,type
    eur_mapping = {
        'id': 'id',
        'title': 'title',
        'description': 'description',
        'creator': 'creator',
        'date': 'year',
        'original_title': None,
        'condition': None,
        'license': None
    }

    # Tzigara
    # box_no,title,original_title,condition,dimensions,image_url,local_path
    tzi_mapping = {
        'id': 'box_no',
        'title': 'title',
        'original_title': 'original_title',
        'condition': 'condition',
        'description': None,
        'creator': None,
        'date': None,
        'license': None
    }

    # Wikimedia
    # id,title,description,creator,date,license,image_url,local_path
    wik_mapping = {
        'id': 'id',
        'title': 'title',
        'description': 'description',
        'creator': 'creator',
        'date': 'date',
        'original_title': None,
        'condition': None,
        'license': 'license'
    }

    # Load them
    try:
        eur_df = load_and_map(f"{base_dir}/europeana/metadata.csv", "europeana", eur_mapping)
        tzi_df = load_and_map(f"{base_dir}/tzigara/metadata.csv", "tzigara", tzi_mapping)
        wik_df = load_and_map(f"{base_dir}/wikimedia/metadata.csv", "wikimedia", wik_mapping)
        
        combined_df = pd.concat([eur_df, tzi_df, wik_df], ignore_index=True)
        combined_df.to_csv(f"{output_dir}/combined_metadata_h1.csv", index=False)
        print(f"[*] Combined metadata saved to {output_dir}/combined_metadata_h1.csv")

        # H1 Analysis: Calculate "Void" (percentage of missing values)
        print("\n--- H1: Epistemic Void Analysis (Percentage of Nulls) ---")
        
        summary_stats = []
        for source in combined_df['source'].unique():
            source_df = combined_df[combined_df['source'] == source]
            source_stats = {'source': source}
            
            # We only care about the semantic/contextual fields for the "void"
            semantic_fields = ['original_title', 'description', 'creator', 'date', 'condition', 'license']
            
            for field in semantic_fields:
                null_count = source_df[field].isna().sum()
                total = len(source_df)
                void_pct = (null_count / total * 100) if total > 0 else 0
                source_stats[field] = f"{void_pct:.1f}%"
                
            summary_stats.append(source_stats)
            
        summary_df = pd.DataFrame(summary_stats)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(f"{output_dir}/h1_void_summary.csv", index=False)
        print(f"\n[*] Summary stats saved to {output_dir}/h1_void_summary.csv")

        # Calculate Global Void (average of voids)
        print("\n--- Global Void Average ---")
        semantic_fields = ['original_title', 'description', 'creator', 'date', 'condition', 'license']
        global_voids = {}
        for field in semantic_fields:
            global_voids[field] = f"{(combined_df[field].isna().mean() * 100):.1f}%"
        
        print(pd.DataFrame([global_voids]).to_string(index=False))

    except Exception as e:
        print(f"[!] Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_h1()
