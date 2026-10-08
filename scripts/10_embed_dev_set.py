#!/usr/bin/env python3
"""
Step 10: Extract face & voice embeddings for dev_set (English test data ONLY).
Generates:
  - embeddings/dev_English_face_embeddings.npz & dev_English_face_embeddings.csv
  - embeddings/dev_English_voice_embeddings.npz & dev_English_voice_embeddings.csv

Usage: python scripts/10_embed_dev_set.py
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from speechbrain.inference.speaker import EncoderClassifier

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DEV_SET_DIR = os.path.join(BASE_DIR, 'data', 'dev_set')
OUTPUT_DIR = os.path.join(BASE_DIR, 'embeddings')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16000


def run_dev_english_face_embeddings():
    print(f"\n{'='*60}")
    print("📸 1. Generating Face Embeddings for dev_set (English Only)")
    print(f"{'='*60}")

    mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=DEVICE)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    # Search for English_test face image paths only
    image_paths = sorted(
        glob.glob(os.path.join(DEV_SET_DIR, '**', 'English_test', 'faces', '*.jpg'), recursive=True) +
        glob.glob(os.path.join(DEV_SET_DIR, '**', 'English_test', 'faces', '*.png'), recursive=True)
    )

    print(f"  Found {len(image_paths)} English face images in dev_set")

    embeddings = []
    metadata = []

    for img_path in tqdm(image_paths, desc="English Face Embeddings (dev_set)"):
        fname = os.path.basename(img_path)
        rel_path = os.path.relpath(img_path, DEV_SET_DIR)

        try:
            img = Image.open(img_path).convert('RGB')
            face_tensor = mtcnn(img)

            if face_tensor is None:
                img_resized = img.resize((160, 160))
                arr = np.array(img_resized).transpose((2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(arr)

            face_tensor = face_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = facenet(face_tensor)
                emb = emb / emb.norm(dim=1, keepdim=True)

            emb_np = emb.cpu().numpy().flatten()
            embeddings.append(emb_np)
            metadata.append(rel_path)

        except Exception as e:
            print(f"\n Error {fname}: {e}")

    embeddings = np.array(embeddings)
    print(f"\n  ✅ Processed {len(embeddings)} English face embeddings (Dim: {embeddings.shape[1]})")

    # Save NPZ
    npz_path = os.path.join(OUTPUT_DIR, 'dev_English_face_embeddings.npz')
    np.savez_compressed(npz_path, embeddings=embeddings, paths=metadata)
    print(f"  💾 Saved NPZ: {npz_path}")

    # Save CSV: feat_0, feat_1, ..., feat_511, file_path
    cols = [f"feat_{i}" for i in range(embeddings.shape[1])] + ["file_path"]
    df = pd.DataFrame(embeddings, columns=cols[:-1])
    df["file_path"] = metadata
    csv_path = os.path.join(OUTPUT_DIR, 'dev_English_face_embeddings.csv')
    df.to_csv(csv_path, index=False)
    print(f"  💾 Saved CSV: {csv_path} ({df.shape[0]} rows × {df.shape[1]} cols)")


def run_dev_english_voice_embeddings():
    print(f"\n{'='*60}")
    print("🎙️ 2. Generating Voice Embeddings for dev_set (English Only)")
    print(f"{'='*60}")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(BASE_DIR, "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": 'cpu'},
    )

    # Search for English_test voice WAV paths only
    wav_paths = sorted(
        glob.glob(os.path.join(DEV_SET_DIR, '**', 'English_test', 'voices', '*.wav'), recursive=True)
    )
    print(f"  Found {len(wav_paths)} English voice WAV files in dev_set")

    embeddings = []
    metadata = []

    for wav_path in tqdm(wav_paths, desc="English Voice Embeddings (dev_set)"):
        fname = os.path.basename(wav_path)
        rel_path = os.path.relpath(wav_path, DEV_SET_DIR)

        try:
            data, sr = sf.read(wav_path)
            waveform = torch.from_numpy(data.astype(np.float32))

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t().mean(dim=0, keepdim=True)

            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)

            with torch.no_grad():
                emb = classifier.encode_batch(waveform)
                emb_np = emb.squeeze().cpu().numpy()

            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-12)
            embeddings.append(emb_np)
            metadata.append(rel_path)

        except Exception as e:
            print(f"\n Error {fname}: {e}")

    embeddings = np.array(embeddings)
    print(f"\n  ✅ Processed {len(embeddings)} English voice embeddings (Dim: {embeddings.shape[1]})")

    # Save NPZ
    npz_path = os.path.join(OUTPUT_DIR, 'dev_English_voice_embeddings.npz')
    np.savez_compressed(npz_path, embeddings=embeddings, paths=metadata)
    print(f"  💾 Saved NPZ: {npz_path}")

    # Save CSV: feat_0, feat_1, ..., feat_191, file_path
    cols = [f"feat_{i}" for i in range(embeddings.shape[1])] + ["file_path"]
    df = pd.DataFrame(embeddings, columns=cols[:-1])
    df["file_path"] = metadata
    csv_path = os.path.join(OUTPUT_DIR, 'dev_English_voice_embeddings.csv')
    df.to_csv(csv_path, index=False)
    print(f"  💾 Saved CSV: {csv_path} ({df.shape[0]} rows × {df.shape[1]} cols)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dev_english_face_embeddings()
    run_dev_english_voice_embeddings()

    print(f"\n{'='*60}")
    print("✅ DEV SET ENGLISH-ONLY EMBEDDING EXTRACTION COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
