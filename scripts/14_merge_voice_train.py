#!/usr/bin/env python3
"""
Step 14: Merge voice_train split (5,187 samples) with voice_embeddings dataset (30,954 samples).
Outputs:
  - embeddings/merged_voice_train.csv (36,141 samples × 193 columns)

Usage: python scripts/14_merge_voice_train.py
"""

import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
VOICE_TRAIN_SPLIT = os.path.join(BASE_DIR, 'embeddings', 'splits', 'voice_train.csv')
VOICE_MAIN_EMB = os.path.join(BASE_DIR, 'embeddings', 'voice_embeddings.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'embeddings', 'merged_voice_train.csv')


def main():
    print("=" * 60)
    print("🔗 Merging Voice Train Split + Voice Embeddings Dataset")
    print("=" * 60)

    print("📖 Loading CSV datasets...")
    df_split = pd.read_csv(VOICE_TRAIN_SPLIT)
    df_main = pd.read_csv(VOICE_MAIN_EMB)

    print(f"  • Voice Train Split (`voice_train.csv`): {df_split.shape[0]:,} rows × {df_split.shape[1]} cols")
    print(f"  • Voice Main Dataset (`voice_embeddings.csv`): {df_main.shape[0]:,} rows × {df_main.shape[1]} cols")

    # Standardize column names to feat_0..feat_191, label
    feat_cols = [f"feat_{i}" for i in range(192)] + ["label"]
    df_split.columns = feat_cols
    df_main.columns = feat_cols

    # Concatenate along rows
    print("\n🔄 Concatenating voice samples...")
    merged_df = pd.concat([df_split, df_main], ignore_index=True)

    print(f"  ✅ Merged Voice Dataset Shape: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} cols")
    print(f"  • Missing Values (NaNs): {merged_df.isna().sum().sum()}")
    print(f"  • Unique Labels/Identities: {merged_df['label'].nunique():,}")

    # Save to CSV
    print(f"\n💾 Saving to {OUTPUT_CSV}...")
    merged_df.to_csv(OUTPUT_CSV, index=False)
    file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)

    print(f"  ✅ File saved successfully! Disk size: {file_size_mb:.2f} MB")
    print(f"\n{'='*60}")
    print("🎉 VOICE TRAIN MERGE COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
