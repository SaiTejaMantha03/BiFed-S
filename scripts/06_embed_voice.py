#!/usr/bin/env python3
"""
Step 6: Generate voice embeddings using SpeechBrain ECAPA-TDNN.
Produces 192-d speaker embeddings from English WAV files.
Usage: python scripts/06_embed_voice.py
"""

import os
import sys
import json
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
VOICE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'VoiceDB', 'English')
OUTPUT_DIR = os.path.join(BASE_DIR, 'embeddings')
TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SEC = 1.0
DEVICE = 'cpu'  # SpeechBrain works best on CPU for inference on Mac


def get_person_id(filename):
    """
    Extract person ID from filename like '1P001E100.WAV'
    Pattern: <session>P<person_num>E<sentence_num>.WAV
    Person ID = everything before 'E'
    """
    base = os.path.splitext(filename)[0]
    # e.g., "1P001E100" -> "1P001"
    parts = base.split('E')
    if len(parts) >= 2:
        return parts[0]
    return base


import soundfile as sf

def load_and_preprocess(filepath, target_sr=TARGET_SAMPLE_RATE):
    """Load audio file using soundfile, convert to mono, resample to target_sr."""
    try:
        data, sr = sf.read(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")

    # Convert to float32 tensor
    waveform = torch.from_numpy(data.astype(np.float32))

    # Soundfile loads mono as 1D (samples,) and stereo as 2D (samples, channels)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    else:
        waveform = waveform.t().mean(dim=0, keepdim=True)  # average channels to mono: (1, samples)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def main():
    print(f"🖥️  Device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── Verify Data ──────────────────────────────────────────────────────
    if not os.path.isdir(VOICE_DATA_DIR):
        print(f"❌ ERROR: Voice data directory not found: {VOICE_DATA_DIR}")
        print("   Run the extraction and filtering steps first.")
        sys.exit(1)

    # ─── Initialize Model ─────────────────────────────────────────────────
    print("📦 Loading ECAPA-TDNN model from SpeechBrain...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(BASE_DIR, "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": DEVICE},
    )
    print("   ✅ Model loaded. Output dimension: 192-d\n")

    # ─── Collect all WAV files ────────────────────────────────────────────
    all_audio = []
    for session in sorted(os.listdir(VOICE_DATA_DIR)):
        session_dir = os.path.join(VOICE_DATA_DIR, session)
        if not os.path.isdir(session_dir):
            continue
        for fname in sorted(os.listdir(session_dir)):
            if fname.upper().endswith('.WAV'):
                all_audio.append({
                    'path': os.path.join(session_dir, fname),
                    'filename': fname,
                    'session': session,
                    'person_id': get_person_id(fname),
                })

    print(f"🎙️  Found {len(all_audio)} WAV files")
    persons = set(a['person_id'] for a in all_audio)
    print(f"👤 Unique speakers: {len(persons)}")
    print(f"📁 Sessions: {sorted(set(a['session'] for a in all_audio))}\n")

    # ─── Generate Embeddings ──────────────────────────────────────────────
    embeddings_dict = {}  # person_id -> list of 192-d embeddings
    metadata_dict = {}    # person_id -> list of {filename, session, duration}
    failed = []
    skipped_short = []

    print("🔄 Generating voice embeddings...")
    for audio_info in tqdm(all_audio, desc="Processing voice"):
        try:
            waveform, sr = load_and_preprocess(audio_info['path'])

            # Check minimum duration
            duration = waveform.shape[1] / sr
            if duration < MIN_DURATION_SEC:
                skipped_short.append(audio_info['filename'])
                continue

            # Generate embedding
            with torch.no_grad():
                embedding = classifier.encode_batch(waveform)
                # Shape: (1, 1, 192) -> (192,)
                emb_np = embedding.squeeze().cpu().numpy()

            # L2-normalize
            emb_np = emb_np / np.linalg.norm(emb_np)

            pid = audio_info['person_id']
            if pid not in embeddings_dict:
                embeddings_dict[pid] = []
                metadata_dict[pid] = []

            embeddings_dict[pid].append(emb_np)
            metadata_dict[pid].append({
                'filename': audio_info['filename'],
                'session': audio_info['session'],
                'duration_sec': round(duration, 2),
            })

        except Exception as e:
            failed.append(audio_info['filename'])
            if len(failed) <= 5:
                print(f"\n  ⚠️  Error processing {audio_info['filename']}: {e}")

    # ─── Save Embeddings ──────────────────────────────────────────────────
    print("\n💾 Saving embeddings...")

    # Convert lists to numpy arrays
    embeddings_arrays = {}
    for pid, emb_list in embeddings_dict.items():
        embeddings_arrays[pid] = np.array(emb_list)

    # Save as .npz
    npz_path = os.path.join(OUTPUT_DIR, 'voice_embeddings.npz')
    np.savez_compressed(npz_path, **embeddings_arrays)
    print(f"   Saved to: {npz_path}")

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, 'voice_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"   Metadata: {meta_path}")

    # ─── Summary ──────────────────────────────────────────────────────────
    total_embeddings = sum(len(v) for v in embeddings_dict.values())
    emb_dim = next(iter(embeddings_arrays.values())).shape[1] if embeddings_arrays else 0

    print(f"\n{'='*50}")
    print(f"✅ Voice Embedding Generation Complete")
    print(f"{'='*50}")
    print(f"   Speakers:         {len(embeddings_dict)}")
    print(f"   Total embeddings: {total_embeddings}")
    print(f"   Embedding dim:    {emb_dim}")
    print(f"   Skipped (short):  {len(skipped_short)}")
    print(f"   Failed:           {len(failed)}")

    if failed:
        print(f"\n   ⚠️  Failed files:")
        for f in failed[:10]:
            print(f"      - {f}")
        if len(failed) > 10:
            print(f"      ... and {len(failed) - 10} more")

    # ─── Sanity Check ─────────────────────────────────────────────────────
    print(f"\n🔍 Sanity Check:")
    # Same-speaker similarity
    for pid, embs in embeddings_arrays.items():
        if len(embs) >= 2:
            cos_sim = np.dot(embs[0], embs[1]) / (
                np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
            print(f"   Same-speaker ({pid}) cosine sim: {cos_sim:.4f} (should be > 0.5)")
            break

    # Cross-speaker similarity
    pids = list(embeddings_arrays.keys())
    if len(pids) >= 2:
        e1 = embeddings_arrays[pids[0]][0]
        e2 = embeddings_arrays[pids[1]][0]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        print(f"   Cross-speaker ({pids[0]} vs {pids[1]}) cosine sim: {cos_sim:.4f} (should be < 0.5)")

    print("\n✅ Done!")


if __name__ == '__main__':
    main()
