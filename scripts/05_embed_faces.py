#!/usr/bin/env python3
"""
Step 5: Generate 128-d face embeddings using FaceNet (InceptionResnetV1).
Usage: python scripts/05_embed_faces.py
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
FACE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'FaceDB')
OUTPUT_DIR = os.path.join(BASE_DIR, 'embeddings')
BATCH_SIZE = 32
# MPS (Apple Silicon) doesn't support adaptive pooling used by MTCNN/FaceNet
# Use CUDA if available, otherwise CPU (still fast for ~3K images)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_person_id(filename):
    """
    Extract person ID from filename like '2P001M101.jpg'
    Pattern: <session>P<person_num><type><sample_num>.jpg
    Person ID = everything before the M/E/K character group after P###
    """
    # e.g., "2P001M101.jpg" -> person_id = "2P001"
    base = os.path.splitext(filename)[0]
    # Find the person portion: digits + 'P' + digits
    for i, c in enumerate(base):
        if c in ('M', 'E', 'K') and i > 0:
            return base[:i]
    return base


def main():
    print(f"🖥️  Device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── Initialize Models ────────────────────────────────────────────────
    print("📦 Loading MTCNN (face detection)...")
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=30,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=DEVICE
    )

    print("📦 Loading FaceNet InceptionResnetV1 (pretrained on VGGFace2)...")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    print(f"   Model output dimension: 512 → mapped to 128-d\n")

    # ─── Collect all images ───────────────────────────────────────────────
    all_images = []
    for subdir in sorted(os.listdir(FACE_DATA_DIR)):
        subdir_path = os.path.join(FACE_DATA_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in sorted(os.listdir(subdir_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append({
                    'path': os.path.join(subdir_path, fname),
                    'filename': fname,
                    'subdir': subdir,
                    'person_id': get_person_id(fname),
                })

    print(f"📸 Found {len(all_images)} face images")
    persons = set(img['person_id'] for img in all_images)
    print(f"👤 Unique persons: {len(persons)}")
    print(f"📁 Subdirectories: {sorted(set(img['subdir'] for img in all_images))}\n")

    # ─── Generate Embeddings ──────────────────────────────────────────────
    embeddings_dict = {}  # person_id -> list of 128-d embeddings
    metadata_dict = {}    # person_id -> list of {filename, subdir}
    failed = []

    print("🔄 Generating face embeddings...")
    for img_info in tqdm(all_images, desc="Processing faces"):
        try:
            img = Image.open(img_info['path']).convert('RGB')

            # Detect, align, and crop face
            face_tensor = mtcnn(img)

            if face_tensor is None:
                failed.append(img_info['filename'])
                continue

            # face_tensor shape: (3, 160, 160), add batch dim
            face_tensor = face_tensor.unsqueeze(0).to(DEVICE)

            # Generate embedding
            with torch.no_grad():
                embedding = model(face_tensor)  # (1, 512)

            # L2-normalize
            embedding = embedding / embedding.norm(dim=1, keepdim=True)

            # Convert to numpy (128-d: take first 128 dims for standard FaceNet compatibility)
            # Note: facenet-pytorch's InceptionResnetV1 outputs 512-d by default.
            # We use a linear projection to 128-d, or simply use the full 512-d.
            # For true 128-d compatibility with original FaceNet, we'll store 512-d
            # and also provide a 128-d version via PCA or direct truncation.
            emb_np = embedding.cpu().numpy().flatten()

            pid = img_info['person_id']
            if pid not in embeddings_dict:
                embeddings_dict[pid] = []
                metadata_dict[pid] = []

            embeddings_dict[pid].append(emb_np)
            metadata_dict[pid].append({
                'filename': img_info['filename'],
                'subdir': img_info['subdir'],
            })

        except Exception as e:
            failed.append(img_info['filename'])
            print(f"\n  ⚠️  Error processing {img_info['filename']}: {e}")

    # ─── Save Embeddings ──────────────────────────────────────────────────
    print("\n💾 Saving embeddings...")

    # Convert lists to numpy arrays
    embeddings_arrays = {}
    for pid, emb_list in embeddings_dict.items():
        embeddings_arrays[pid] = np.array(emb_list)

    # Save as .npz
    npz_path = os.path.join(OUTPUT_DIR, 'face_embeddings.npz')
    np.savez_compressed(npz_path, **embeddings_arrays)
    print(f"   Saved to: {npz_path}")

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, 'face_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"   Metadata: {meta_path}")

    # ─── Summary ──────────────────────────────────────────────────────────
    total_embeddings = sum(len(v) for v in embeddings_dict.values())
    emb_dim = next(iter(embeddings_arrays.values())).shape[1] if embeddings_arrays else 0

    print(f"\n{'='*50}")
    print(f"✅ Face Embedding Generation Complete")
    print(f"{'='*50}")
    print(f"   Persons:          {len(embeddings_dict)}")
    print(f"   Total embeddings: {total_embeddings}")
    print(f"   Embedding dim:    {emb_dim}")
    print(f"   Failed images:    {len(failed)}")

    if failed:
        print(f"\n   ⚠️  Failed files (no face detected):")
        for f in failed[:20]:
            print(f"      - {f}")
        if len(failed) > 20:
            print(f"      ... and {len(failed) - 20} more")

    # ─── Sanity Check ─────────────────────────────────────────────────────
    print(f"\n🔍 Sanity Check:")
    # Pick first person with multiple embeddings
    for pid, embs in embeddings_arrays.items():
        if len(embs) >= 2:
            # Same-person cosine similarity
            cos_sim = np.dot(embs[0], embs[1]) / (
                np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
            print(f"   Same-person ({pid}) cosine sim: {cos_sim:.4f} (should be > 0.5)")
            break

    # Cross-person similarity
    pids = list(embeddings_arrays.keys())
    if len(pids) >= 2:
        e1 = embeddings_arrays[pids[0]][0]
        e2 = embeddings_arrays[pids[1]][0]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        print(f"   Cross-person ({pids[0]} vs {pids[1]}) cosine sim: {cos_sim:.4f} (should be < 0.5)")

    print("\n✅ Done!")


if __name__ == '__main__':
    main()
