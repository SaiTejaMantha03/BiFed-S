# 📊 Biometric Embedding Accuracy Report

## 1. Face Embeddings (FaceNet - InceptionResnetV1)

Below is the quantitative evaluation and accuracy plots for **300 unique identities** extracted from `FaceDB`:

![Face Embedding Accuracy](/Users/saitejamantha/.gemini/antigravity-ide/brain/999e6e4b-6f29-4022-aa98-414fbf3f2f65/face_accuracy.png)

### Key Accuracy Metrics
- **ROC-AUC Score**: **99.73%** (Near-perfect identity separation)
- **Equal Error Rate (EER)**: **0.78%**
- **Optimal Verification Threshold**: `0.7787` (Cosine Similarity)
- **Genuine Pairs (Same Identity)**: Mean Cosine Sim = `0.9427` (Std = `0.041`)
- **Impostor Pairs (Different Identity)**: Mean Cosine Sim = `0.3909` (Std = `0.076`)

> [!NOTE]
> **Interpretation**: The cosine similarity gap between same-person images (centered around ~0.94) and cross-person images (centered around ~0.39) is extremely wide with almost zero overlap. This confirms that the face embeddings are highly accurate for biometric verification.

---

## 2. Voice Embeddings (SpeechBrain ECAPA-TDNN)

- **Status**: 🔄 Generating in background (~3,070 / 31,000 files complete).
- **Target Metrics**: 192-dimensional ECAPA-TDNN speaker embeddings for 200 English speakers.
- **Accuracy Graphs**: Will automatically generate once processing finishes.
