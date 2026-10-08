#!/usr/bin/env python3
"""
Step 15: Merge Face Train and Voice Train embeddings into a unified Multimodal (Face + Voice) Dataset.
Outputs:
  - embeddings/splits/multimodal_train.csv (5,187 samples × 4,289 columns)
  - embeddings/splits/multimodal_test.csv  (1,297 samples × 4,289 columns)

Usage: python scripts/15_merge_multimodal_train.py
"""

import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SPLIT_DIR = os.path.join(BASE_DIR, 'embeddings', 'splits')


def merge_modality_pair(face_csv_path, voice_csv_path, output_csv_path, split_name):
    print(f"\n🔄 Merging Face and Voice features for {split_name} split...")
    
    if not os.path.exists(face_csv_path) or not os.path.exists(voice_csv_path):
        print(f"❌ Error: Required split CSV files not found!")
        return

    df_face = pd.read_csv(face_csv_path)
    df_voice = pd.read_csv(voice_csv_path)

    print(f"  • Face {split_name} Shape:  {df_face.shape[0]:,} rows × {df_face.shape[1]} cols")
    print(f"  • Voice {split_name} Shape: {df_voice.shape[0]:,} rows × {df_voice.shape[1]} cols")

    # Verify matching labels
    face_labels = df_face.iloc[:, -1].values
    voice_labels = df_voice.iloc[:, -1].values
    assert (face_labels == voice_labels).all(), f"Error: {split_name} labels do not align row-by-row!"

    # Extract & rename face features
    face_feats = df_face.iloc[:, :-1]
    face_feats.columns = [f"face_feat_{i}" for i in range(face_feats.shape[1])]

    # Extract & rename voice features
    voice_feats = df_voice.iloc[:, :-1]
    voice_feats.columns = [f"voice_feat_{i}" for i in range(voice_feats.shape[1])]

    # Extract label
    labels = df_face.iloc[:, -1].rename("label")

    # Column-wise concatenation: Face (4096) + Voice (192) + Label (1) = 4289 cols
    multimodal_df = pd.concat([face_feats, voice_feats, labels], axis=1)

    print(f"  ✅ Multimodal {split_name} Shape: {multimodal_df.shape[0]:,} rows × {multimodal_df.shape[1]} cols")
    print(f"  • Feature Composition: {face_feats.shape[1]} Face features + {voice_feats.shape[1]} Voice features = {face_feats.shape[1] + voice_feats.shape[1]} total features")
    print(f"  • Missing Values (NaNs): {multimodal_df.isna().sum().sum()}")

    # Save to CSV
    multimodal_df.to_csv(output_csv_path, index=False)
    file_size_mb = os.path.getsize(output_csv_path) / (1024 * 1024)
    print(f"  💾 Saved: {output_csv_path} ({file_size_mb:.2f} MB)")


def main():
    print("=" * 60)
    print("🧬 Multimodal Feature Fusion: Merging Face + Voice Datasets")
    print("=" * 60)

    # 1. Merge Training Split
    merge_modality_pair(
        os.path.join(SPLIT_DIR, 'face_train.csv'),
        os.path.join(SPLIT_DIR, 'voice_train.csv'),
        os.path.join(SPLIT_DIR, 'multimodal_train.csv'),
        "Training"
    )

    # 2. Merge Testing Split
    merge_modality_pair(
        os.path.join(SPLIT_DIR, 'face_test.csv'),
        os.path.join(SPLIT_DIR, 'voice_test.csv'),
        os.path.join(SPLIT_DIR, 'multimodal_test.csv'),
        "Testing"
    )

    print(f"\n{'='*60}")
    print("🎉 MULTIMODAL (FACE + VOICE) MERGE COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
