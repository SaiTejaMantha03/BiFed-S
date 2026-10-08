#!/usr/bin/env python3
"""
Step 9: Convert .npz vector embeddings to CSV format.
Generates:
  - embeddings/face_embeddings.csv  (512 feature columns + label column)
  - embeddings/voice_embeddings.csv (192 feature columns + label column)
  - embeddings/label_mapping.json   (Mapping between string person IDs and integer labels)

Usage: python scripts/09_convert_npz_to_csv.py
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings')


def convert_npz_to_csv(npz_file, csv_output, modality_name):
    npz_path = os.path.join(EMB_DIR, npz_file)
    csv_path = os.path.join(EMB_DIR, csv_output)

    print(f"\n🔄 Converting {modality_name} embeddings ({npz_file}) to CSV...")
    if not os.path.exists(npz_path):
        print(f"❌ File not found: {npz_path}")
        return None

    data = np.load(npz_path)
    person_ids = sorted(list(data.keys()))

    # Build consistent integer label mapping
    label_mapping = {pid: i + 1 for i, pid in enumerate(person_ids)}

    rows = []
    total_samples = 0
    feature_dim = 0

    for pid in person_ids:
        embs = data[pid]
        label = label_mapping[pid]
        feature_dim = embs.shape[1] if embs.ndim == 2 else embs.shape[0]

        for emb in embs:
            # Append feature values + integer label at the end
            row = list(emb) + [label]
            rows.append(row)
            total_samples += 1

    # Define column names: feat_0, feat_1, ..., feat_{D-1}, label
    col_names = [f"feat_{i}" for i in range(feature_dim)] + ["label"]

    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(csv_path, index=False)

    print(f"  ✅ Saved: {csv_path}")
    print(f"  • Rows (samples): {total_samples:,}")
    print(f"  • Columns: {feature_dim} features + 1 label column = {df.shape[1]} columns")
    print(f"  • Unique identities (labels): {len(label_mapping)}")
    print(f"  • File size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")

    return {
        'csv_file': csv_output,
        'rows': total_samples,
        'columns': df.shape[1],
        'feature_dim': feature_dim,
        'unique_labels': len(label_mapping),
        'file_size_mb': round(os.path.getsize(csv_path) / (1024*1024), 2),
        'label_mapping': label_mapping
    }


def main():
    print("=" * 60)
    print("📦 Converting Biometric Embeddings (.npz → .csv)")
    print("=" * 60)

    # 1. Face Embeddings CSV
    face_info = convert_npz_to_csv('face_embeddings.npz', 'face_embeddings.csv', 'Face')

    # 2. Voice Embeddings CSV
    voice_info = convert_npz_to_csv('voice_embeddings.npz', 'voice_embeddings.csv', 'Voice')

    # Save Label Mappings
    mappings = {}
    if face_info:
        mappings['face_label_mapping'] = face_info['label_mapping']
    if voice_info:
        mappings['voice_label_mapping'] = voice_info['label_mapping']

    mapping_path = os.path.join(EMB_DIR, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"\n💾 Saved Label Mappings to: {mapping_path}")

    # Update manifest.json to reference CSVs
    manifest_path = os.path.join(EMB_DIR, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if face_info and 'face' in manifest.get('modalities', {}):
            manifest['modalities']['face']['csv_file'] = 'face_embeddings.csv'
        if voice_info and 'voice' in manifest.get('modalities', {}):
            manifest['modalities']['voice']['csv_file'] = 'voice_embeddings.csv'

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"💾 Updated manifest.json with CSV file references.")

    print(f"\n{'='*60}")
    print("✅ CSV CONVERSION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
