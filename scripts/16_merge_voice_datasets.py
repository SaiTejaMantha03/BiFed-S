#!/usr/bin/env python3
"""
Step 16: Merge VOICE datasets ONLY (row-wise concatenation).
Combines Voice Train, Voice Test, Dev Voice, and Main Voice embeddings into a single Voice-only dataset.
Voice and Face remain 100% SEPARATE.

Outputs:
  - embeddings/voice_merged_dataset.csv (Voice ONLY)

Usage: python scripts/16_merge_voice_datasets.py
"""

import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings')
SPLIT_DIR = os.path.join(EMB_DIR, 'splits')


def main():
    print("=" * 60)
    print("🎙️ Merging VOICE Datasets ONLY (100% Separate from Face)")
    print("=" * 60)

    # 1. Voice splits & files
    v_main = os.path.join(EMB_DIR, 'voice_embeddings.csv')
    v_dev = os.path.join(EMB_DIR, 'dev_English_voice_embeddings.csv')
    v_train_split = os.path.join(SPLIT_DIR, 'voice_train.csv')
    v_test_split = os.path.join(SPLIT_DIR, 'voice_test.csv')

    feat_cols = [f"feat_{i}" for i in range(192)] + ["label"]
    dfs = []

    if os.path.exists(v_train_split):
        df = pd.read_csv(v_train_split)
        df.columns = feat_cols
        df['source_split'] = 'voice_train_split'
        dfs.append(df)
        print(f"  • Voice Train Split (`voice_train.csv`): {df.shape[0]:,} rows × 193 cols")

    if os.path.exists(v_test_split):
        df = pd.read_csv(v_test_split)
        df.columns = feat_cols
        df['source_split'] = 'voice_test_split'
        dfs.append(df)
        print(f"  • Voice Test Split (`voice_test.csv`): {df.shape[0]:,} rows × 193 cols")

    if os.path.exists(v_dev):
        df = pd.read_csv(v_dev)
        df_feats = df.iloc[:, :192].copy()
        df_feats.columns = [f"feat_{i}" for i in range(192)]
        df_feats['label'] = df['file_path'].apply(lambda p: os.path.basename(os.path.dirname(p)))
        df_feats['source_split'] = 'voice_dev_english'
        dfs.append(df_feats)
        print(f"  • Voice Dev English (`dev_English_voice_embeddings.csv`): {df.shape[0]:,} rows × 193 cols")

    if os.path.exists(v_main):
        df = pd.read_csv(v_main)
        df.columns = feat_cols
        df['source_split'] = 'voice_main_dataset'
        dfs.append(df)
        print(f"  • Voice Main Dataset (`voice_embeddings.csv`): {df.shape[0]:,} rows × 193 cols")

    if not dfs:
        print("❌ No voice embedding files found!")
        return

    merged_voice_df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(EMB_DIR, 'voice_merged_dataset.csv')
    merged_voice_df.to_csv(out_path, index=False)

    print(f"\n  ✅ Merged Voice Dataset Saved: {out_path}")
    print(f"  • Total Voice Samples: {merged_voice_df.shape[0]:,}")
    print(f"  • Total Columns: {merged_voice_df.shape[1]} (192 Voice Features + Label + Source Split)")
    print(f"  • Missing Values (NaNs): {merged_voice_df.isna().sum().sum()}")
    print(f"\n{'='*60}")
    print("✅ VOICE MERGE COMPLETE (VOICE ONLY)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
