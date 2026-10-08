#!/usr/bin/env python3
"""
Step 17: Merge CSV files inside each folder in embeddings/splits/
For each subfolder:
  - embeddings/splits/train_data_face/ -> embeddings/splits/train_data_face.csv
  - embeddings/splits/train_voice/     -> embeddings/splits/train_voice.csv
  - embeddings/splits/test_face/       -> embeddings/splits/test_face.csv
  - embeddings/splits/test_voice/      -> embeddings/splits/test_voice.csv

Usage: python scripts/17_merge_splits_folders.py
"""

import os
import glob
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SPLITS_DIR = os.path.join(BASE_DIR, 'embeddings', 'splits')


def merge_folder_csvs(folder_name):
    folder_path = os.path.join(SPLITS_DIR, folder_name)
    if not os.path.isdir(folder_path):
        return

    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
    print(f"\n📂 Processing folder: embeddings/splits/{folder_name}/ ({len(csv_files)} CSV files)")

    if not csv_files:
        print("  ⚠️ No CSV files found in folder.")
        return

    # Group CSV files by column count to prevent column mismatch
    dfs_by_cols = {}
    for cpath in csv_files:
        df = pd.read_csv(cpath)
        col_count = df.shape[1]
        if col_count not in dfs_by_cols:
            dfs_by_cols[col_count] = []

        # Standardize column headers for feature concatenation
        if 'face' in folder_name or 'voice' in folder_name:
            label_col = df.columns[-1]
            feat_count = col_count - 1
            feat_cols = [f"feat_{i}" for i in range(feat_count)] + [label_col]
            df.columns = feat_cols

        df['source_file'] = os.path.basename(cpath)
        dfs_by_cols[col_count].append(df)
        print(f"  • Read {os.path.basename(cpath)}: {df.shape[0]:,} rows × {col_count} cols")

    # Concatenate per feature dimension group
    for col_count, df_list in dfs_by_cols.items():
        merged = pd.concat(df_list, ignore_index=True)

        if len(dfs_by_cols) == 1:
            out_name = f"{folder_name}.csv"
        else:
            out_name = f"{folder_name}_{col_count-1}dim.csv"

        out_path = os.path.join(SPLITS_DIR, out_name)
        merged.to_csv(out_path, index=False)

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  ✅ Saved Merged CSV: {out_path}")
        print(f"     Shape: {merged.shape[0]:,} rows × {merged.shape[1]} cols | Size: {size_mb:.2f} MB")


def main():
    print("=" * 60)
    print("🗂️  Merging CSV files per folder in embeddings/splits/")
    print("=" * 60)

    folders = sorted([
        d for d in os.listdir(SPLITS_DIR)
        if os.path.isdir(os.path.join(SPLITS_DIR, d))
    ])

    for folder in folders:
        merge_folder_csvs(folder)

    print(f"\n{'='*60}")
    print("🎉 FOLDER CSV MERGE COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
