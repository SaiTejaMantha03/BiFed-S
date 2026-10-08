#!/usr/bin/env python3
"""
Step 7: Validate embeddings and create unified manifest.
Usage: python scripts/07_validate_embeddings.py
"""

import os
import json
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings')


def validate_embeddings(npz_path, name):
    """Validate an embeddings .npz file."""
    print(f"\n{'─'*50}")
    print(f"Validating: {name}")
    print(f"{'─'*50}")

    if not os.path.exists(npz_path):
        print(f"  ❌ File not found: {npz_path}")
        return None

    data = np.load(npz_path)
    persons = list(data.keys())

    if len(persons) == 0:
        print(f"  ❌ No embeddings found!")
        return None

    # Collect stats
    total = 0
    dims = set()
    has_nan = False
    has_inf = False

    for pid in persons:
        embs = data[pid]
        total += len(embs)
        dims.add(embs.shape[1] if embs.ndim == 2 else embs.shape[0])
        if np.any(np.isnan(embs)):
            has_nan = True
        if np.any(np.isinf(embs)):
            has_inf = True

    # Check consistency
    dim_consistent = len(dims) == 1
    emb_dim = dims.pop() if dim_consistent else dims

    print(f"  Persons:          {len(persons)}")
    print(f"  Total embeddings: {total}")
    print(f"  Embedding dim:    {emb_dim} {'✅' if dim_consistent else '❌ INCONSISTENT!'}")
    print(f"  NaN values:       {'❌ YES' if has_nan else '✅ None'}")
    print(f"  Inf values:       {'❌ YES' if has_inf else '✅ None'}")
    print(f"  File size:        {os.path.getsize(npz_path) / (1024*1024):.2f} MB")

    # Samples per person distribution
    samples = [len(data[pid]) for pid in persons]
    print(f"  Samples/person:   min={min(samples)}, max={max(samples)}, "
          f"avg={np.mean(samples):.1f}, median={np.median(samples):.1f}")

    return {
        'persons': len(persons),
        'total_embeddings': total,
        'embedding_dim': int(emb_dim) if dim_consistent else list(dims),
        'dim_consistent': dim_consistent,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'file_size_mb': round(os.path.getsize(npz_path) / (1024*1024), 2),
        'person_ids': persons,
    }


def main():
    print("=" * 50)
    print("🔍 Embedding Validation & Manifest Generation")
    print("=" * 50)

    manifest = {
        'project': 'Biometric Federated Learning',
        'description': 'Face and voice embeddings for biometric verification',
        'modalities': {},
    }

    # Validate face embeddings
    face_info = validate_embeddings(
        os.path.join(EMB_DIR, 'face_embeddings.npz'),
        'Face Embeddings'
    )
    if face_info:
        manifest['modalities']['face'] = {
            'file': 'face_embeddings.npz',
            'metadata_file': 'face_metadata.json',
            'model': 'FaceNet InceptionResnetV1 (VGGFace2)',
            'embedding_dim': face_info['embedding_dim'],
            'persons': face_info['persons'],
            'total_embeddings': face_info['total_embeddings'],
            'valid': face_info['dim_consistent'] and not face_info['has_nan'] and not face_info['has_inf'],
        }

    # Validate voice embeddings
    voice_info = validate_embeddings(
        os.path.join(EMB_DIR, 'voice_embeddings.npz'),
        'Voice Embeddings'
    )
    if voice_info:
        manifest['modalities']['voice'] = {
            'file': 'voice_embeddings.npz',
            'metadata_file': 'voice_metadata.json',
            'model': 'ECAPA-TDNN (SpeechBrain, VoxCeleb)',
            'embedding_dim': voice_info['embedding_dim'],
            'persons': voice_info['persons'],
            'total_embeddings': voice_info['total_embeddings'],
            'valid': voice_info['dim_consistent'] and not voice_info['has_nan'] and not voice_info['has_inf'],
        }

    # Cross-modality analysis
    if face_info and voice_info:
        face_persons = set(face_info['person_ids'])
        voice_persons = set(voice_info['person_ids'])
        common = face_persons & voice_persons
        face_only = face_persons - voice_persons
        voice_only = voice_persons - face_persons

        print(f"\n{'─'*50}")
        print(f"Cross-Modality Analysis")
        print(f"{'─'*50}")
        print(f"  Face-only persons:  {len(face_only)}")
        print(f"  Voice-only persons: {len(voice_only)}")
        print(f"  Common persons:     {len(common)}")

        manifest['cross_modality'] = {
            'common_persons': len(common),
            'face_only_persons': len(face_only),
            'voice_only_persons': len(voice_only),
            'common_person_ids': sorted(list(common)),
        }

    # Save manifest
    manifest_path = os.path.join(EMB_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n💾 Manifest saved: {manifest_path}")

    # Final verdict
    all_valid = all(
        m.get('valid', False)
        for m in manifest.get('modalities', {}).values()
    )
    print(f"\n{'='*50}")
    if all_valid:
        print("✅ ALL EMBEDDINGS VALID — Ready for Federated Learning!")
    else:
        print("⚠️  Some embeddings have issues — review above.")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
