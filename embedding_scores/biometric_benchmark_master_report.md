# 🏆 Biometric Verification & Presentation Attack Detection Master Report
### Consolidated Evaluation for Face, Voice, Multimodal Fusion, and Spoof Detection

---

## 1. 🎯 Biometric Verification Summary (ROC-AUC, EER, TAR@FAR)

| Modality | ROC-AUC | Equal Error Rate (EER) | Optimal Threshold | TAR @ FAR=1% | TAR @ FAR=0.1% | TAR @ FAR=0.01% |
|---|---|---|---|---|---|---|
| **Face (4096d)** | **97.50%** | **8.44%** | `0.2574` | **71.47%** | **50.79%** | **37.30%** |
| **Voice (192d)** | **98.57%** | **4.68%** | `0.2859` | **90.80%** | **80.02%** | **61.37%** |
| **Combined Fusion** | **99.69%** | **2.29%** | `0.3727` | **96.33%** | **86.91%** | **72.87%** |

---

## 2. 🛡️ Presentation Attack Detection Summary (ISO/IEC 30107-3: APCER, BPCER, ACER)

| Modality PAD | PAD ROC-AUC | EER (APCER=BPCER) | Min ACER | APCER @ BPCER=1% | APCER @ BPCER=5% |
|---|---|---|---|---|---|
| **Face Anti-Spoof** | **56.01%** | **45.97%** | **44.57%** | **99.40%** | **95.97%** |
| **Voice Anti-Spoof** | **55.78%** | **46.52%** | **45.77%** | **98.53%** | **92.13%** |
| **Combined Multimodal PAD** | **57.17%** | **45.43%** | **44.88%** | **95.80%** | **89.97%** |

---

## 3. 📁 Generated Artifacts
- **Face Report**: `embedding_scores/face_verification_report.md`
- **Voice Report**: `embedding_scores/voice_verification_report.md`
- **Combined Multimodal Report**: `embedding_scores/multimodal_verification_report.md`
- **Spoof Detection (PAD) Report**: `embedding_scores/spoof_detection_report.md`
- **Visual Plots**:
  - `embedding_scores/face_roc_eer_tar.png`
  - `embedding_scores/voice_roc_eer_tar.png`
  - `embedding_scores/multimodal_roc_eer_tar.png`
  - `embedding_scores/spoof_detection_pad_analysis.png`
