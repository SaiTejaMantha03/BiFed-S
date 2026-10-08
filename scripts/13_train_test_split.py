#!/usr/bin/env python3
"""
Step 13: Stratified and Synchronized 80/20 Train-Test Split for Face and Voice Embeddings.
Outputs to embeddings/splits/:
  - face_train.csv  (5,187 samples × 4,097 cols)
  - face_test.csv   (1,297 samples × 4,097 cols)
  - voice_train.csv (5,187 samples × 193 cols)
  - voice_test.csv  (1,297 samples × 193 cols)

Usage: python scripts/13_train_test_split.py [--test-ratio 0.2]
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
FACE_CSV = os.path.join(BASE_DIR, 'data', 'train_set', 'features', 'faces', 'train_English_faces.csv')
VOICE_CSV = os.path.join(BASE_DIR, 'data', 'train_set', 'features', 'voices', 'train_English_voices.csv')
SPLIT_DIR = os.path.join(BASE_DIR, 'embeddings', 'splits')


def main():
    parser = argparse.ArgumentParser(description="Synchronized Train-Test Split for Face and Voice Embeddings")
    parser.add_argument('--test-size', type=float, default=0.2, help="Test set ratio (default: 0.2 for 80/20 split)")
    parser.add_argument('--random-state', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(SPLIT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"✂️  Synchronized Train-Test Split ({int((1-args.test_size)*100)}% Train / {int(args.test_size*100)}% Test)")
    print("=" * 60)

    # 1. Load Face and Voice CSVs
    print("📖 Loading datasets...")
    df_face = pd.read_csv(FACE_CSV)
    df_voice = pd.read_csv(VOICE_CSV)

    print(f"  • Face Features: {df_face.shape[0]:,} samples × {df_face.shape[1]} columns")
    print(f"  • Voice Features: {df_voice.shape[0]:,} samples × {df_voice.shape[1]} columns")

    # Verify matching length
    assert len(df_face) == len(df_voice), "Error: Face and Voice dataset lengths do not match!"

    # 2. Extract labels and generate synchronized stratified indices
    labels = df_face.iloc[:, -1].values
    indices = list(range(len(df_face)))

    print("\n🔄 Performing synchronized stratified split...")
    idx_train, idx_test = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels
    )

    # 3. Split Face Data
    face_train = df_face.iloc[idx_train]
    face_test = df_face.iloc[idx_test]

    # 4. Split Voice Data
    voice_train = df_voice.iloc[idx_train]
    voice_test = df_voice.iloc[idx_test]

    # 5. Save Split CSV Files
    print("\n💾 Saving split CSV files to embeddings/splits/...")
    
    f_train_path = os.path.join(SPLIT_DIR, 'face_train.csv')
    f_test_path = os.path.join(SPLIT_DIR, 'face_test.csv')
    v_train_path = os.path.join(SPLIT_DIR, 'voice_train.csv')
    v_test_path = os.path.join(SPLIT_DIR, 'voice_test.csv')

    face_train.to_csv(f_train_path, index=False)
    face_test.to_csv(f_test_path, index=False)
    voice_train.to_csv(v_train_path, index=False)
    voice_test.to_csv(v_test_path, index=False)

    print(f"  ✅ Saved: {f_train_path} ({face_train.shape[0]:,} rows × {face_train.shape[1]} cols)")
    print(f"  ✅ Saved: {f_test_path} ({face_test.shape[0]:,} rows × {face_test.shape[1]} cols)")
    print(f"  ✅ Saved: {v_train_path} ({voice_train.shape[0]:,} rows × {voice_train.shape[1]} cols)")
    print(f"  ✅ Saved: {v_test_path} ({voice_test.shape[0]:,} rows × {voice_test.shape[1]} cols)")

    print(f"\n{'='*60}")
    print("🎉 TRAIN-TEST SPLIT COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
