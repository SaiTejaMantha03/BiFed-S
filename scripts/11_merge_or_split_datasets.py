#!/usr/bin/env python3
"""
Step 11: Dataset Split vs. Merge Utility.
Demonstrates:
  1. Keeping datasets separate (DO NOT MERGE) for train vs dev evaluation.
  2. Merging datasets into unified files with a 'split' column (train/dev).

Usage: python scripts/11_merge_or_split_datasets.py [--merge]
"""

import os
import argparse
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings')


def load_datasets():
    """Load Train and Dev face & voice embeddings."""
    f_train_path = os.path.join(EMB_DIR, 'face_embeddings.csv')
    f_dev_path = os.path.join(EMB_DIR, 'dev_English_face_embeddings.csv')
    v_train_path = os.path.join(EMB_DIR, 'voice_embeddings.csv')
    v_dev_path = os.path.join(EMB_DIR, 'dev_English_voice_embeddings.csv')

    print("📖 Loading embedding datasets...")
    f_train = pd.read_csv(f_train_path)
    f_dev = pd.read_csv(f_dev_path)
    v_train = pd.read_csv(v_train_path)
    v_dev = pd.read_csv(v_dev_path)

    print(f"  • Face Train: {f_train.shape[0]:,} samples × {f_train.shape[1]} cols")
    print(f"  • Face Dev:   {f_dev.shape[0]:,} samples × {f_dev.shape[1]} cols")
    print(f"  • Voice Train: {v_train.shape[0]:,} samples × {v_train.shape[1]} cols")
    print(f"  • Voice Dev:   {v_dev.shape[0]:,} samples × {v_dev.shape[1]} cols")

    return f_train, f_dev, v_train, v_dev


def merge_datasets(f_train, f_dev, v_train, v_dev):
    """Merge Train and Dev sets while preserving a 'split' indicator column."""
    print("\n🔗 Merging Datasets...")

    # 1. Merge Face Embeddings
    f_tr = f_train.copy()
    f_tr['split'] = 'train'
    f_d = f_dev.copy()
    f_d['split'] = 'dev'

    # Standardize column naming if needed (keep feat_0..feat_511)
    feat_cols_face = [f"feat_{i}" for i in range(512)]
    f_combined = pd.concat([
        f_tr[feat_cols_face + ['split']],
        f_d[feat_cols_face + ['split']]
    ], ignore_index=True)

    out_face_csv = os.path.join(EMB_DIR, 'combined_face_embeddings.csv')
    f_combined.to_csv(out_face_csv, index=False)
    print(f"  ✅ Combined Face CSV Saved: {out_face_csv} ({f_combined.shape[0]:,} rows × {f_combined.shape[1]} cols)")

    # 2. Merge Voice Embeddings
    v_tr = v_train.copy()
    v_tr['split'] = 'train'
    v_d = v_dev.copy()
    v_d['split'] = 'dev'

    feat_cols_voice = [f"feat_{i}" for i in range(192)]
    v_combined = pd.concat([
        v_tr[feat_cols_voice + ['split']],
        v_d[feat_cols_voice + ['split']]
    ], ignore_index=True)

    out_voice_csv = os.path.join(EMB_DIR, 'combined_voice_embeddings.csv')
    v_combined.to_csv(out_voice_csv, index=False)
    print(f"  ✅ Combined Voice CSV Saved: {out_voice_csv} ({v_combined.shape[0]:,} rows × {v_combined.shape[1]} cols)")


def main():
    parser = argparse.ArgumentParser(description="Dataset Split vs Merge Utility")
    parser.add_argument('--merge', action='store_true', help="Merge train and dev datasets into single combined files")
    args = parser.parse_args()

    print("=" * 60)
    print("⚖️  Biometric Dataset Split / Merge Utility")
    print("=" * 60)

    f_train, f_dev, v_train, v_dev = load_datasets()

    if args.merge:
        merge_datasets(f_train, f_dev, v_train, v_dev)
    else:
        print("\n💡 Recommendation: Keeping Train and Dev sets SEPARATE (DO NOT MERGE)")
        print("   • Train Set: Used for local client training & federated aggregation")
        print("   • Dev Set:   Reserved for evaluating unbiased verification performance")
        print("\n   (To merge anyway, run: python scripts/11_merge_or_split_datasets.py --merge)")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
