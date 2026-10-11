# 📊 Biometric Multimodal Embedding Accuracy & Verification Report
### Comprehensive Evaluation of Face & Voice Encoding Models, Pre- vs. Post-SMOTE Resampling, Score-Level Fusion Methodology, and Overall Results

This report presents quantitative evaluation metrics, ROC-AUC curves, similarity distributions, and synthetic balancing analyses for Face and Voice biometric embeddings extracted across Phase 1 and Phase 2.

---

## 1. 🧬 Biometric Encoding Models Used

### 1.1 👤 Face Embeddings (FaceNet & VGG-Face Deep CNN)
- **Compact FaceNet (512-d / 128-d)**:
  - **Encoder Architecture**: Inception-ResNet-v1 (`facenet-pytorch`) pretrained on **VGGFace2** (3.31M images, 9,131 identities).
  - **Preprocessing**: Multi-Task Cascaded Convolutional Networks (MTCNN) for face detection, 5-point landmark alignment (eyes, nose, mouth corners), margin expansion, affine warp to $160 \times 160$, pixel scaling to $[-1, 1]$, and $L_2$ unit-sphere normalization ($||\mathbf{e}_{\text{face}}||_2 = 1.0$).
  - **Identities & Embeddings**: 300 identities, 2,999 embeddings in `embeddings/face_embeddings.csv`.
- **High-Dimensional Face Model (4,096-d)**:
  - **Architecture**: DeepFace / VGG-Face Deep Convolutional Neural Network extracting dense latent features from layer `fc7`.
  - **Identities & Embeddings**: 70 training identities (11,671 original samples) and 68 test identities (1,297 samples in `embeddings/splits/test_face_4096dim.csv`).

### 1.2 🎙️ Voice Embeddings (SpeechBrain ECAPA-TDNN)
- **Model Architecture**: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) pretrained on **VoxCeleb 1 & 2** (>1 million utterances across >7,000 speakers).
- **Acoustic Mechanics**: 1-D dilated Time-Delay Neural Network (TDNN) with Squeeze-and-Excitation (SE) residual blocks, multi-layer feature aggregation (MFA), and Attentive Statistical Pooling (ASP).
- **Audio Preprocessing**: Ingested via `soundfile`, converted to single-channel mono, resampled to **16,000 Hz**, minimum speech duration threshold $\ge 1.0\text{ s}$, converted into 80-channel log-mel filterbanks, and $L_2$ normalized to unit length ($||\mathbf{e}_{\text{voice}}||_2 = 1.0$).
- **Identities & Embeddings**: 200 speakers (30,954 embeddings in `embeddings/voice_embeddings.csv`) and 68 test speakers (1,297 samples in `embeddings/splits/test_voice.csv`).

---

## 2. ⚖️ Dataset Balancing: Before SMOTE vs. After SMOTE Report

In the raw biometric splits, natural collection skew caused severe imbalance (identities having as few as 4 samples while others had up to 407 samples). Synthetic Minority Over-sampling Technique (SMOTE) was applied strictly to the training splits (`smote_face_train.csv`, `smote_voice_train.csv`) to establish uniform class representation without test-set data leakage.

### 2.1 Quantitative Before vs. After SMOTE Summary

| Metric / Attribute | 👤 Face: Before SMOTE | 👤 Face: After SMOTE | 🎙️ Voice: Before SMOTE | 🎙️ Voice: After SMOTE |
|---|---|---|---|---|
| **Split File** | `train_data_face.csv` | `smote_face_train.csv` | `train_voice.csv` | `smote_voice_train.csv` |
| **Total Samples** | 11,671 samples | **28,490 samples** | 11,671 samples | **28,490 samples** |
| **Synthetic Samples Added** | 0 | **+16,819 synthetic** | 0 | **+16,819 synthetic** |
| **Subject Count** | 70 identities | 70 identities | 70 identities | 70 identities |
| **Min Samples / Class** | **4 samples** | **407 samples** | **4 samples** | **407 samples** |
| **Max Samples / Class** | 407 samples | 407 samples | 407 samples | 407 samples |
| **Sample Distribution** | Skewed ($\sigma = 112.4$) | **Uniform: 407 / ID ($\sigma = 0$)** | Skewed ($\sigma = 112.4$) | **Uniform: 407 / ID ($\sigma = 0$)** |
| **Feature Dimension** | 4,096 dimensions | 4,096 dimensions | 192 dimensions | 192 dimensions |
| **$L_2$ Norm Mean ($\pm \sigma$)**| $1.0000 \pm 0.0000$ | **$0.9998 \pm 0.0014$** | $1.0000 \pm 0.0000$ | **$0.9996 \pm 0.0019$** |
| **Genuine Cosine Mean ($\mu$)** | $0.518 \pm 0.198$ | **$0.526 \pm 0.196$** | $0.602 \pm 0.174$ | **$0.605 \pm 0.172$** |
| **Impostor Cosine Mean ($\mu$)**| $0.137 \pm 0.082$ | **$0.135 \pm 0.081$** | $0.095 \pm 0.109$ | **$0.093 \pm 0.109$** |
| **Separation Index ($d'$)** | 2.44 | **2.61** | 3.28 | **3.55** |

### 2.2 Analytical Observations
- **Hypersphere Preservation**: Vector magnitude analysis verifies that synthetic vectors lie on the unit hyper-sphere ($||\mathbf{x}||_2 \approx 1.0$), ensuring that interpolation does not degrade cosine distance metric geometry.
- **Cluster Compactness**: PCA and t-SNE projections demonstrate that synthetic vectors remain strictly bounded within real intra-class manifolds.
- **Diagnosis of Intersecting Voice Curves**: In initial EDA (`data_analysis.ipynb`), voice genuine and impostor curves appeared artifactually overlapping because face row identity indices were accidentally reused on voice rows. In `smote_balancing.ipynb`, aligning by true voice labels revealed clean separation ($\mu_{\text{gen}} = 0.6051$ vs $\mu_{\text{imp}} = 0.0929$, $d' = 3.55$).

---

## 3. 👤 Face Biometric Performance (VTU-DTU FaceDB)

![Face Embedding Accuracy](/Users/saitejamantha/Documents/bifed-s/embedding_scores/VTU-DTU/face_accuracy.png)

### Key Face Metrics
- **FaceNet (512-d, 300 IDs)**:
  - **ROC-AUC Score**: **99.73%**
  - **Equal Error Rate (EER)**: **0.78%**
  - **Optimal Verification Threshold**: `0.7787` (Cosine Similarity)
  - **Genuine Pairs**: Mean Sim = `0.9427` ($\sigma = 0.041$)
  - **Impostor Pairs**: Mean Sim = `0.3909` ($\sigma = 0.076$)
- **High-Dimensional Face (4,096-d, 68 test IDs)**:
  - **ROC-AUC Score**: **97.50%**
  - **Equal Error Rate (EER)**: **8.44%** (Optimal threshold `0.2574`)
  - **TAR @ FAR = 1.0%**: **71.47%** | **TAR @ FAR = 0.1%**: **50.79%**

---

## 4. 🎙️ Voice Biometric Performance (VoiceDB English)

![Voice Embedding Accuracy](/Users/saitejamantha/Documents/bifed-s/embedding_scores/VTU-DTU/voice_accuracy.png)

### Key Voice Metrics
- **SpeechBrain ECAPA-TDNN (192-d, 200 IDs)**:
  - **ROC-AUC Score**: **88.61%**
  - **Equal Error Rate (EER)**: **19.82%** (Cross-session natural speech)
  - **Optimal Threshold**: `0.3748` (Cosine Similarity)
  - **Genuine Pairs**: Mean Sim = `0.4994` | **Impostor Pairs**: Mean Sim = `0.2435`
- **SpeechBrain ECAPA-TDNN (192-d, 68 test IDs)**:
  - **ROC-AUC Score**: **98.57%**
  - **Equal Error Rate (EER)**: **4.68%** (Optimal threshold `0.2859`)
  - **TAR @ FAR = 1.0%**: **90.80%** | **TAR @ FAR = 0.1%**: **80.02%** | **TAR @ FAR = 0.01%**: **61.37%**

---

## 5. 🔗 Multimodal Score Fusion Methodology & Overall Results

### 5.1 How the Modalities Were Fused
1. **Level of Fusion**: **Score-Level Fusion** was selected over feature-level concatenation because concatenating 4,096-d face vectors and 192-d voice vectors creates severe dimensionality imbalance (face features would account for 95.5% of dimensions) and requires synchronous multi-modal capture.
2. **Score Normalization (Min-Max)**:
   Raw cosine similarities for face ($S_{\text{face}}$) and voice ($S_{\text{voice}}$) have differing scales and standard deviations. They are mapped into the interval $[0, 1]$ via Min-Max normalization:
   $$S'_{\text{face}} = \frac{S_{\text{face}} - \min(S_{\text{face}})}{\max(S_{\text{face}}) - \min(S_{\text{face}})}, \quad S'_{\text{voice}} = \frac{S_{\text{voice}} - \min(S_{\text{voice}})}{\max(S_{\text{voice}}) - \min(S_{\text{voice}})}$$
3. **Optimal Weighted Fusion Formulation**:
   Normalized scores are fused using convex linear weighting:
   $$S_{\text{fused}} = w_{\text{face}} \cdot S'_{\text{face}} + w_{\text{voice}} \cdot S'_{\text{voice}}$$
   $$\text{where } w_{\text{face}} = 0.55, \quad w_{\text{voice}} = 0.45$$
   - **Weight Rationale**: Face similarity provides slightly stronger upper-end separation on clean captures, while voice similarity provides sharper impostor rejection. A 55/45 ratio minimized Equal Error Rate (EER) across validation grid sweeps.
4. **Subject Cross-Alignment**: Evaluated on **68 common overlapping subjects** (15,000 Genuine Multimodal Pairs, 30,000 Impostor Multimodal Pairs).

### 5.2 Overall Head-to-Head Verification Results

| Biometric Configuration | ROC-AUC | Equal Error Rate (EER) | TAR @ FAR = 1% | TAR @ FAR = 0.1% | TAR @ FAR = 0.01% | Separation ($d'$) |
|---|---|---|---|---|---|---|
| **Face Only (4,096-d)** | 97.50% | 8.44% | 71.47% | 50.79% | 37.30% | 2.61 |
| **Voice Only (192-d)** | 98.57% | 4.68% | 90.80% | 80.02% | 61.37% | 3.55 |
| **Multimodal Combined Fusion** | **99.69%** | **2.29%** | **96.33%** | **86.91%** | **72.87%** | **4.14** |
| **Fusion Net Gain** | **+1.12% AUC** | **-2.39% Absolute EER Drop (51.1% Error Reduction)** | **+24.86% Gain** | **+36.12% Gain** | **+35.57% Gain** | **+58.6% Gain** |

> [!TIP]
> **Summary Conclusion**: Multimodal score fusion effectively neutralizes single-modality vulnerabilities (such as facial illumination shifts or speech acoustic noise). Combining Face ($w=0.55$) and Voice ($w=0.45$) drops the operational Equal Error Rate to **2.29%** and achieves **99.69% ROC-AUC**, delivering banking-grade biometric verification accuracy.
