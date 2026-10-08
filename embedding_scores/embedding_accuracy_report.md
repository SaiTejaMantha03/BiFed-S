# 📊 Biometric Multimodal Embedding Accuracy Report

This report presents quantitative evaluation metrics, ROC-AUC curves, and similarity distributions for both Face and Voice biometric embeddings extracted in Phase 1.

---

## 1. 👤 Face Embeddings (FaceNet - InceptionResnetV1)

- **Total Identities**: 300
- **Total Embeddings**: 2,999 vectors (512-dimensional)
- **Model**: FaceNet InceptionResnetV1 (Pretrained on VGGFace2)

![Face Embedding Accuracy](/Users/saitejamantha/.gemini/antigravity-ide/brain/999e6e4b-6f29-4022-aa98-414fbf3f2f65/face_accuracy.png)

### Performance Metrics
| Metric | Value |
|---|---|
| **ROC-AUC Score** | **99.73%** |
| **Equal Error Rate (EER)** | **0.78%** |
| **Optimal Threshold** | `0.7787` (Cosine Similarity) |
| **Genuine Pairs (Same ID)** | Mean Sim = `0.9427` |
| **Impostor Pairs (Diff ID)** | Mean Sim = `0.3909` |

---

## 2. 🎙️ Voice Embeddings (SpeechBrain ECAPA-TDNN)

- **Total Identities**: 200 (English WAV files only)
- **Total Embeddings**: 30,954 vectors (192-dimensional)
- **Model**: SpeechBrain ECAPA-TDNN (Pretrained on VoxCeleb)

![Voice Embedding Accuracy](/Users/saitejamantha/.gemini/antigravity-ide/brain/999e6e4b-6f29-4022-aa98-414fbf3f2f65/voice_accuracy.png)

### Performance Metrics
| Metric | Value |
|---|---|
| **ROC-AUC Score** | **88.61%** |
| **Equal Error Rate (EER)** | **19.82%** |
| **Optimal Threshold** | `0.3748` (Cosine Similarity) |
| **Genuine Pairs (Same Speaker)** | Mean Sim = `0.4994` |
| **Impostor Pairs (Diff Speaker)** | Mean Sim = `0.2435` |

---

## 3. 🔗 Cross-Modality Analysis

- **Common Identities (Face + Voice)**: **100 overlapping persons**
- **Face-only Identities**: 200
- **Voice-only Identities**: 100
- **Dataset Manifest**: Saved in `embeddings/manifest.json`

> [!TIP]
> **Multimodal Fusion Potential**: Combining Face (99.73% AUC) + Voice (88.61% AUC) embeddings for the 100 common identities will enable robust **Multimodal Biometric Verification** in the Federated Learning phase!
