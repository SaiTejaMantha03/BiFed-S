# 🧬 Biometric Federated Learning — Data Preprocessing & Embedding Pipeline

This repository implements Phase 1 of the Biometric Federated Learning project: extracting, filtering, preprocessing, and generating vector embeddings for multimodal biometric verification (Face + Voice).

## 📁 Repository Structure

```
trifed-s/
├── data/                        # Extracted raw data (git ignored)
│   ├── FaceDB/                  # Face dataset (300 identities, 2,999 images)
│   └── VoiceDB/                 # Voice dataset (200 identities, English only)
├── embeddings/                  # Vector embeddings & metadata
│   ├── face_embeddings.npz      # 512-d / 128-d FaceNet embeddings
│   ├── voice_embeddings.npz     # 192-d ECAPA-TDNN voice embeddings
│   ├── manifest.json            # Dataset manifest & cross-modality index
│   └── plots/                   # ROC-AUC & Cosine similarity distribution plots
├── scripts/                     # Preprocessing & embedding pipeline scripts
│   ├── 02_filter_voice_en.py    # Language filtering (Remove Kannada, keep English)
│   ├── 05_embed_faces.py        # Face detection (MTCNN) + FaceNet embeddings
│   ├── 06_embed_voice.py        # Audio resample (16kHz mono) + ECAPA-TDNN embeddings
│   ├── 07_validate_embeddings.py# Data validation & manifest generator
│   └── 08_visualize_embeddings.py# EER, ROC-AUC, Cosine Sim & PCA cluster plots
├── requirements.txt             # Python dependencies
└── README.md
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Pipeline Steps**:
   ```bash
   # Filter Voice Data (Remove Kannada)
   python scripts/02_filter_voice_en.py

   # Generate Face Embeddings (FaceNet)
   python scripts/05_embed_faces.py

   # Generate Voice Embeddings (ECAPA-TDNN)
   python scripts/06_embed_voice.py

   # Validate & Manifest
   python scripts/07_validate_embeddings.py

   # Visualize Accuracy & Metrics
   python scripts/08_visualize_embeddings.py
   ```

## 📊 Verification Metrics
- **Face (FaceNet)**: **99.73% ROC-AUC**, **0.78% EER**
- **Voice (ECAPA-TDNN)**: 192-d Speaker Vectors
