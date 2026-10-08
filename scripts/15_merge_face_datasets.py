#!/usr/bin/env python3
"""
Step 15: Merge FACE datasets ONLY (row-wise concatenation).
Combines Face Train, Face Test, and Dev Face embeddings into a single Face-only dataset.
Face and Voice remain 100% SEPARATE.

Outputs:
  - embeddings/face_merged_dataset.csv (Face ONLY)

Usage: python scripts/15_merge_face_datasets.py
"""

import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings')
SPLIT_DIR = os.path.join(EMB_DIR, 'splits')


def main():
    print("=" * 60)
    print("👤 Merging FACE Datasets ONLY (100% Separate from Voice)")
    print("=" * 60)

    # 1. Primary 512-d FaceNet embeddings (Train + Dev)
    f_main = os.path.join(EMB_DIR, 'face_embeddings.csv')
    f_dev = os.path.join(EMB_DIR, 'dev_English_face_embeddings.csv')

    dfs = []
    if os.path.exists(f_main):
        df1 = pd.read_csv(f_main)
        df1['source_split'] = 'train'
        dfs.append(df1)
        print(f"  • Face Train (`face_embeddings.csv`): {df1.shape[0]:,} rows × {df1.shape[1]} cols")

    if os.path.exists(f_dev):
        df2 = pd.read_csv(f_dev)
        df2['source_split'] = 'dev'
        dfs.append(df2)
        print(f"  • Face Dev (`dev_English_face_embeddings.csv`): {df2.shape[0]:,} rows × {df2.shape[1]} cols")

    if not dfs:
        print("❌ No face embedding files found!")
        return

    merged_face_df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(EMB_DIR, 'face_merged_dataset.csv')
    merged_face_df.to_csv(out_path, index=False)

    print(f"\n  ✅ Merged Face Dataset Saved: {out_path}")
    print(f"  • Total Face Samples: {merged_face_df.shape[0]:,}")
    print(f"  • Total Columns: {merged_face_df.shape[1]} (512 Face Features + Metadata)")
    print(f"  • Missing Values (NaNs): {merged_face_df.isna().sum().sum()}")
    print(f"\n{'='*60}")
    print("✅ FACE MERGE COMPLETE (FACE ONLY)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
