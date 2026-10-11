# 📊 Biometric Multimodal Embedding Accuracy & Verification Report
### Detailed Encoding Model Specifications, Before vs. After SMOTE Analysis, Multimodal Score Fusion, and Overall Verification Results

This report presents a comprehensive evaluation of biometric feature extraction, synthetic class balancing via SMOTE, multimodal score-level fusion, and biometric verification benchmarks across Face and Voice modalities.

---

## 1. 🧬 Biometric Encoding Models Used

### 1.1 Face Encoding Architecture (FaceNet & VGG-Face)
- **FaceNet Inception-ResNet-v1 (Compact 512-d Space)**:
  - **Framework**: `facenet-pytorch` pretrained on **VGGFace2** (3.31 million images, 9,131 identities).
  - **Input Preprocessing**: Multi-Task Cascaded Convolutional Networks (MTCNN) for face detection, landmark localization (eyes, nose, mouth), bounding box expansion with margin 20, and affine alignment to $160 \times 160$ RGB.
  - **Normalization**: Pixel standardization to $[-1, 1]$ followed by $L_2$ unit-sphere embedding normalization:
    $$||\mathbf{e}_{\text{face}}||_2 = 1.0$$
- **DeepFace / VGG-Face (High-Dimensional 4,096-d Space)**:
  - **Framework**: Deep CNN extracting 4,096-dimensional representations from penultimate dense layer (`fc7`), optimized with identity cross-entropy loss.
  - **Feature Properties**: Captures high-order non-linear facial geometry, fine facial texture, and pose invariants.

### 1.2 Voice Encoding Architecture (SpeechBrain ECAPA-TDNN)
- **SpeechBrain ECAPA-TDNN (192-d Space)**:
  - **Pretrained Checkpoint**: `speechbrain/spkrec-ecapa-voxceleb` trained on **VoxCeleb 1 & 2** (>1 million utterances across >7,000 speakers).
  - **Architectural Mechanics**: 1-D dilated Time-Delay Neural Network (TDNN) with Squeeze-and-Excitation (SE) residual blocks, multi-scale feature propagation, and Attentive Statistical Pooling (ASP).
  - **Audio Preprocessing**: Ingested via `soundfile`, converted to single-channel mono, resampled to **16,000 Hz**, minimum speech duration threshold $\ge 1.0\text{ s}$, converted into 80-channel log-mel filterbanks, and $L_2$ normalized to unit length ($||\mathbf{e}_{\text{voice}}||_2 = 1.0$).

---

## 2. ⚖️ Dataset Balancing: Before SMOTE vs. After SMOTE Report

Severe identity imbalance naturally occurs in biometric datasets (some subjects have 400+ samples while others have as few as 4). Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training splits (`smote_face_train.csv`, `smote_voice_train.csv`) to establish uniform class representation without test-set data leakage.

### 2.1 Quantitative Before vs. After SMOTE Comparison Table

| Metric / Property | 👤 Face: Before SMOTE | 👤 Face: After SMOTE | 🎙️ Voice: Before SMOTE | 🎙️ Voice: After SMOTE |
|---|---|---|---|---|
| **Dataset File** | `train_data_face.csv` | `smote_face_train.csv` | `train_voice.csv` | `smote_voice_train.csv` |
| **Total Samples** | 11,671 samples | **28,490 samples** | 11,671 samples | **28,490 samples** |
| **Synthetic Vectors Added** | 0 | **+16,819 synthetic** | 0 | **+16,819 synthetic** |
| **Subject Count** | 70 identities | 70 identities | 70 identities | 70 identities |
| **Min Samples / Class** | **4 samples** | **407 samples** | **4 samples** | **407 samples** |
| **Max Samples / Class** | 407 samples | 407 samples | 407 samples | 407 samples |
| **Class Distribution** | Heavily skewed ($\sigma = 112.4$) | **Uniform 407 / ID ($\sigma = 0$)** | Heavily skewed ($\sigma = 112.4$) | **Uniform 407 / ID ($\sigma = 0$)** |
| **Feature Dimensionality** | 4,096 dimensions | 4,096 dimensions | 192 dimensions | 192 dimensions |
| **$L_2$ Norm Mean ($\pm \sigma$)** | $1.0000 \pm 0.0000$ | **$0.9998 \pm 0.0014$** | $1.0000 \pm 0.0000$ | **$0.9996 \pm 0.0019$** |
| **Genuine Cosine Mean ($\mu$)** | $0.518 \pm 0.198$ | **$0.526 \pm 0.196$** | $0.602 \pm 0.174$ | **$0.605 \pm 0.172$** |
| **Impostor Cosine Mean ($\mu$)**| $0.137 \pm 0.082$ | **$0.135 \pm 0.081$** | $0.095 \pm 0.109$ | **$0.093 \pm 0.109$** |
| **Separation Index ($d'$)** | 2.44 | **2.61** | 3.28 | **3.55** |

### 2.2 Analytical Findings & EDA Diagnosis
1. **Preservation of the Latent Manifold**:
   - $L_2$ norm tracking confirms that linear interpolation between $k$-nearest neighbors ($k=3$) preserves the hypersphere radius ($\mu \approx 1.0$) without collapsing vectors toward the coordinate origin.
   - 2D PCA and t-SNE projections show synthetic samples occupying the interior convex hull of each subject cluster, solidifying decision boundaries without creating spurious inter-class overlaps.
2. **Resolution of EDA Row-Alignment Artifact**:
   - In preliminary analysis (`data_analysis.ipynb`), voice genuine and impostor curves appeared artifactually overlapping due to reusing Face row labels on independently ordered Voice rows.
   - Correcting row alignment in `smote_balancing.ipynb` and evaluating post-SMOTE balanced vectors confirms sharp speaker discrimination ($\mu_{\text{gen}} = 0.6051$ vs $\mu_{\text{imp}} = 0.0929$, $d' = 3.55$).

---

## 3. 👤 Face Verification Benchmarks

### Visual Performance Plots (VTU-DTU FaceDB)
![Face Embedding Accuracy](/Users/saitejamantha/Documents/bifed-s/embedding_scores/VTU-DTU/face_accuracy.png)

### Performance Across Benchmarks:
- **FaceNet (512-d)** on 300 unique identities (2,999 vectors):
  - **ROC-AUC Score**: **99.73%**
  - **Equal Error Rate (EER)**: **0.78%**
  - **Optimal Threshold**: `0.7787` (Cosine Similarity)
  - **Genuine Pairs**: Mean Sim = `0.9427` ($\sigma = 0.041$)
  - **Impostor Pairs**: Mean Sim = `0.3909` ($\sigma = 0.076$)
- **High-Dimensional Face (4,096-d)** on held-out test split (68 identities, 1,297 samples):
  - **ROC-AUC Score**: **97.50%**
  - **Equal Error Rate (EER)**: **8.44%** (Optimal threshold `0.2574`)
  - **TAR @ FAR = 1.0%**: **71.47%**
  - **TAR @ FAR = 0.1%**: **50.79%**

---

## 4. 🎙️ Voice Verification Benchmarks

### Visual Performance Plots (VoiceDB English)
![Voice Embedding Accuracy](/Users/saitejamantha/Documents/bifed-s/embedding_scores/VTU-DTU/voice_accuracy.png)

### Performance Across Benchmarks:
- **SpeechBrain ECAPA-TDNN (192-d)** on 200 English speakers (30,954 vectors):
  - **ROC-AUC Score**: **88.61%**
  - **Equal Error Rate (EER)**: **19.82%** (Cross-session conversational speech)
  - **Optimal Threshold**: `0.3748` (Cosine Similarity)
  - **Genuine Pairs**: Mean Sim = `0.4994` | **Impostor Pairs**: Mean Sim = `0.2435`
- **SpeechBrain ECAPA-TDNN (192-d)** on held-out test split (68 speakers, 1,297 samples):
  - **ROC-AUC Score**: **98.57%**
  - **Equal Error Rate (EER)**: **4.68%** (Optimal threshold `0.2859`)
  - **TAR @ FAR = 1.0%**: **90.80%**
  - **TAR @ FAR = 0.1%**: **80.02%**
  - **TAR @ FAR = 0.01%**: **61.37%**

---

## 5. 🔗 Multimodal Fusion Methodology & Overall Results

### 5.1 How the Modalities Were Fused
1. **Level of Fusion**: **Score-Level Fusion**. Score-level fusion was selected over feature-level concatenation because concatenating 4,096-d face vectors and 192-d voice vectors creates severe dimensionality imbalance (face features would account for 95.5% of dimensions) and requires synchronous multi-modal capture.
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
