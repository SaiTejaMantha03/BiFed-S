# 🔗 Multimodal Combined Biometric Verification Report
### Face + Voice Score Fusion Performance vs Single Modalities

- **Modality Fusion**: Score-level weighted fusion ($S_{\text{fused}} = 0.55 \cdot S_{\text{face}} + 0.45 \cdot S_{\text{voice}}$)
- **Overlapping Identities**: 68 common biometric identities evaluated in test split
- **Evaluation Trials**: 15,000 Genuine Multimodal Pairs, 30,000 Impostor Pairs

---

## 📊 Comparative Performance Visualizations
![Multimodal Verification Performance](/Users/saitejamantha/Documents/bifed-s/embedding_scores/multimodal_roc_eer_tar.png)

---

## 🎯 Head-to-Head Verification Comparison Table

| Metric | Face Only (4096d) | Voice Only (192d) | Combined Fusion (Face + Voice) | Fusion Relative Improvement |
|---|---|---|---|---|
| **ROC-AUC Score** | 97.50% | 98.57% | **99.69%** | **+1.12% AUC** |
| **Equal Error Rate (EER)** | 8.44% | 4.68% | **2.29%** | **-2.38% Absolute EER Drop** |
| **TAR @ FAR = 1.0%** | 71.47% | 90.80% | **96.33%** | **+24.85% Gain** |
| **TAR @ FAR = 0.1%** | 50.79% | 80.02% | **86.91%** | **+36.12% Gain** |
| **TAR @ FAR = 0.01%** | 37.30% | 61.37% | **72.87%** | **+35.57% Gain** |

> [!TIP]
> **Key Insight**: Multimodal fusion suppresses modality-specific weaknesses (e.g. lighting variation in faces, ambient noise in voice). Fusing both modalities reduces the error rate down to **2.29% EER**, surpassing either single modality operating in isolation.
