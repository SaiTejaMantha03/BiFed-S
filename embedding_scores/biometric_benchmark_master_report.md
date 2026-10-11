# 🏆 Biometric Verification & Presentation Attack Detection Master Report
### Consolidated Evaluation for Face, Voice, Multimodal Score Fusion, SMOTE Balancing, and Anti-Spoof Detection

---

## 1. 🧬 Biometric Encoding Models & Feature Spaces

| Modality | Primary Deep Architecture | Pretraining Dataset | Feature Dimension | Input Preprocessing & Normalization |
|---|---|---|---|---|
| **Face (Primary)** | DeepFace / VGG-Face Deep CNN | VGGFace (Millions of facial images) | **4,096-d** | $160 \times 160$ aligned face crops, $L_2$ unit hyper-sphere normalized |
| **Face (Compact)** | Inception-ResNet-v1 (`facenet-pytorch`) | VGGFace2 (3.31M images, 9,131 IDs) | **512-d** | MTCNN landmark detection, affine warp, $L_2$ normalized |
| **Voice (Speech)** | SpeechBrain ECAPA-TDNN | VoxCeleb 1 & 2 (>1M utterances, >7k IDs)| **192-d** | 16 kHz mono, $\ge 1.0\text{ s}$ duration, 80-dim log-mel filterbanks, $L_2$ normalized |
| **Multimodal** | Weighted Score-Level Fusion | Joint aligned biometric identities | **Score Fusion** | Min-Max score normalization ($[0, 1]$), $w_{\text{face}} = 0.55$, $w_{\text{voice}} = 0.45$ |

---

## 2. ⚖️ SMOTE Class Balancing Analysis (Before vs. After)

Applied strictly on the training partitions (`smote_face_train.csv`, `smote_voice_train.csv`) to eliminate biometric subject imbalance without evaluation split data leakage.

| Modality Split | Pre-SMOTE Samples (`train_*.csv`) | Post-SMOTE Samples (`smote_*_train.csv`) | Min / Max Samples per Class | Post-SMOTE Class Distribution | $L_2$ Norm Preservation |
|---|---|---|---|---|---|
| **Face Train** | 11,671 samples (70 IDs) | **28,490 samples** (+16,819 synthetic) | Min: 4 / Max: 407 | **Uniform: 407 samples / ID** | $\mu = 0.9998 \pm 0.0014$ |
| **Voice Train** | 11,671 samples (70 IDs) | **28,490 samples** (+16,819 synthetic) | Min: 4 / Max: 407 | **Uniform: 407 samples / ID** | $\mu = 0.9996 \pm 0.0019$ |

> [!NOTE]
> **Manifold Stability**: SMOTE interpolates along local intra-class geodesics on the unit hypersphere, populating the convex hulls of minority identities while fully preserving the genuine vs. impostor verification margins ($d'_{\text{face}} = 2.61$, $d'_{\text{voice}} = 3.55$).

---

## 3. 🎯 Biometric Verification Summary (ROC-AUC, EER, TAR@FAR)

Evaluated across **15,347 Genuine Pairs** and **46,041 Impostor Pairs** on held-out test splits (68 common subjects):

| Modality Stream | ROC-AUC | Equal Error Rate (EER) | Optimal Cutoff | TAR @ FAR=1.0% | TAR @ FAR=0.1% | TAR @ FAR=0.01% | Separation ($d'$) |
|---|---|---|---|---|---|---|---|
| **Face Only (4,096-d)** | **97.50%** | **8.44%** | `0.2574` | **71.47%** | **50.79%** | **37.30%** | `2.61` |
| **Voice Only (192-d)** | **98.57%** | **4.68%** | `0.2859` | **90.80%** | **80.02%** | **61.37%** | `3.55` |
| **Combined Fusion (Face + Voice)** | **99.69%** | **2.29%** | `0.3727` | **96.33%** | **86.91%** | **72.87%** | **`4.14`** |

### How Multimodal Fusion Was Performed
- **Level**: Score-Level Fusion preserving full analog confidence from deep encoders.
- **Normalization**: Min-Max normalization mapping Face and Voice cosine similarities onto $[0, 1]$.
- **Formulation**:
  $$S_{\text{fused}} = 0.55 \cdot S'_{\text{face}} + 0.45 \cdot S'_{\text{voice}}$$
- **Result**: Fused verification slashes Equal Error Rate down to **2.29%** (a 51.1% error reduction compared to voice alone and 72.9% reduction compared to face alone), while True Acceptance at high-security $\text{FAR} = 10^{-4}$ expands to **72.87%**.

---

## 4. 🛡️ Presentation Attack Detection Summary (ISO/IEC 30107-3)

Evaluated across Bona Fide presentations and synthetic / replay attack presentations:

| Modality PAD | PAD ROC-AUC | EER (APCER = BPCER) | Minimum ACER | APCER @ BPCER = 1% | APCER @ BPCER = 5% |
|---|---|---|---|---|---|
| **Face Anti-Spoof** | **56.01%** | **45.97%** | **44.57%** | **99.40%** | **95.97%** |
| **Voice Anti-Spoof** | **55.78%** | **46.52%** | **45.77%** | **98.53%** | **92.13%** |
| **Combined Multimodal PAD** | **57.17%** | **45.43%** | **44.88%** | **95.80%** | **89.97%** |

---

## 5. 📁 Generated Benchmark Artifacts & Reports
- **Face Verification Report**: [`embedding_scores/face_verification_report.md`](face_verification_report.md)
- **Voice Verification Report**: [`embedding_scores/voice_verification_report.md`](voice_verification_report.md)
- **Multimodal Verification Report**: [`embedding_scores/multimodal_verification_report.md`](multimodal_verification_report.md)
- **Spoof Detection (PAD) Report**: [`embedding_scores/spoof_detection_report.md`](spoof_detection_report.md)
- **Visual Plots**:
  - `embedding_scores/face_roc_eer_tar.png`
  - `embedding_scores/voice_roc_eer_tar.png`
  - `embedding_scores/multimodal_roc_eer_tar.png`
  - `embedding_scores/spoof_detection_pad_analysis.png`
