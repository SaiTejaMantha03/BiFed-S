# 🧬 BiFed-S: Multimodal Biometric Federated Learning & Verification Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Modality-Face%20%2B%20Voice%20Fusion-orange.svg)](#)
[![ISO Standard](https://img.shields.io/badge/PAD%20Standard-ISO%2FIEC%2030107--3-purple.svg)](#)

**BiFed-S** is a comprehensive multimodal biometric verification and presentation attack detection (PAD) framework. It provides an end-to-end reproducible pipeline encompassing raw biometric preprocessing, deep facial and vocal embedding extraction, synthetic minority class balancing via SMOTE, exploratory data analysis (EDA), score-level multimodal fusion, and standardized ISO/IEC 30107-3 biometric verification benchmarking.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Biometric Encoding Models & Feature Extraction](#-biometric-encoding-models--feature-extraction)
- [Class Balancing with SMOTE: Before vs. After Analysis](#-class-balancing-with-smote-before-vs-after-analysis)
- [Multimodal Score-Level Fusion Methodology](#-multimodal-score-level-fusion-methodology)
- [Biometric Benchmark Results](#-biometric-benchmark-results)
  - [1. Verification Performance (ROC-AUC, EER, TAR@FAR)](#1-verification-performance)
  - [2. VTU-DTU Dataset Benchmark Evaluation](#2-vtu-dtu-dataset-benchmark-evaluation)
  - [3. Presentation Attack Detection (ISO/IEC 30107-3)](#3-presentation-attack-detection-isoiec-30107-3)
- [Exploratory Data Analysis & Notebooks](#-exploratory-data-analysis--notebooks)
- [Getting Started](#-getting-started)
- [Pipeline Execution](#-pipeline-execution)
- [Generated Artifacts & Reports](#-generated-artifacts--reports)
- [License](#-license)

---

## ✨ Key Features

- **Dual-Modality Deep Encoders**:
  - **Face**: High-dimensional 4,096-d feature representation (DeepFace / VGG-Face) and compact 512-d FaceNet Inception-ResNet-v1 representations aligned via MTCNN.
  - **Voice**: 192-dimensional speaker verification embeddings extracted using **SpeechBrain ECAPA-TDNN** trained on VoxCeleb 1 & 2.
- **Synchronized Multimodal Train/Test Splitting**: Stratified 80/20 identity-aligned train and test partitions (`scripts/13_train_test_split.py`).
- **Class Balancing with SMOTE** (`smote_balancing.ipynb`):
  - Solves severe biometric identity sample skew (expanding 11,671 training samples to 28,490 samples, uniform 407 samples per identity).
  - Preserves $L_2$ unit-sphere embedding geometry without feature distortion or evaluation split data leakage.
- **Calibrated Score-Level Multimodal Fusion**:
  - Min-Max normalized score fusion combining Face ($w = 0.55$) and Voice ($w = 0.45$).
  - Cuts Equal Error Rate (EER) by **51.1%** down to **2.29%**, reaching **99.69% ROC-AUC**.
- **Standardized Biometric Benchmark Suite** ([`generate_biometric_reports.py`](scripts/generate_biometric_reports.py)):
  - Verification: **ROC-AUC**, **EER**, **FAR**, **FRR**, and **TAR @ FAR** (1%, 0.1%, 0.01%).
  - Presentation Attack Detection (PAD / Anti-Spoofing): Evaluated under **ISO/IEC 30107-3** (**APCER**, **BPCER**, and **ACER**).

---

## 📁 Repository Structure

```
BiFed-S/
├── data/                                      # Raw biometric datasets (git-ignored)
│   ├── FaceDB/                                # Facial image database
│   └── VoiceDB/                               # Speech audio database (English filtered)
├── embeddings/                                # Generated deep feature vectors & metadata
│   ├── face_embeddings.csv                    # Extracted 512-d FaceNet embeddings (300 IDs)
│   ├── voice_embeddings.csv                   # Extracted 192-d ECAPA-TDNN embeddings (200 IDs)
│   ├── face_metadata.json                     # Face dataset metadata
│   ├── voice_metadata.json                    # Voice dataset metadata
│   ├── label_mapping.json                     # Identity label index mapping
│   ├── manifest.json                          # Multimodal aligned dataset manifest
│   ├── plots/                                 # Visual embedding accuracy plots
│   └── splits/                                # Aligned train/test partitions
│       ├── train_data_face.csv                # Raw face training set (11,671 samples, 70 IDs)
│       ├── test_face_4096dim.csv              # Face test set (1,297 samples, 68 IDs, 4096-d)
│       ├── test_face_512dim.csv               # Face test set (6,854 samples, 512-d)
│       ├── train_voice.csv                    # Raw voice training set (11,671 samples, 70 IDs)
│       ├── test_voice.csv                     # Voice test set (1,297 samples, 68 IDs, 192-d)
│       ├── smote_face_train.csv               # Post-SMOTE balanced face train split (28,490 samples)
│       └── smote_voice_train.csv              # Post-SMOTE balanced voice train split (28,490 samples)
├── embedding_scores/                          # Quantitative benchmark reports & visual plots
│   ├── VTU-DTU/                               # VTU-DTU dataset benchmark partition
│   │   ├── embedding_accuracy_report.md       # Multimodal embedding accuracy summary
│   │   ├── embedding_accuracy_report_face.md  # Detailed face & voice verification report
│   │   ├── face_accuracy.png                  # Face cosine similarity & ROC distribution plot
│   │   └── voice_accuracy.png                 # Voice cosine similarity & ROC distribution plot
│   ├── biometric_benchmark_master_report.md   # Executive verification & PAD master report
│   ├── face_verification_report.md            # Deep face verification metrics & SMOTE analysis
│   ├── voice_verification_report.md           # ECAPA-TDNN voice verification & SMOTE analysis
│   ├── multimodal_verification_report.md      # Face + Voice score fusion performance report
│   ├── spoof_detection_report.md              # ISO/IEC 30107-3 presentation attack detection
│   ├── face_roc_eer_tar.png                   # Face ROC curve, DET tradeoff, and score densities
│   ├── voice_roc_eer_tar.png                  # Voice ROC curve, DET tradeoff, and score densities
│   ├── multimodal_roc_eer_tar.png             # Multimodal comparative ROC & score densities
│   └── spoof_detection_pad_analysis.png       # PAD APCER vs. BPCER operational tradeoff curves
├── scripts/                                   # Automated end-to-end pipeline scripts
│   ├── 02_filter_voice_en.py                  # English utterance audio filter
│   ├── 05_embed_faces.py                      # MTCNN face detection & FaceNet embedding extraction
│   ├── 06_embed_voice.py                      # 16 kHz audio preprocessing & ECAPA-TDNN extraction
│   ├── 07_validate_embeddings.py              # Embedding integrity checks & manifest compilation
│   ├── 08_visualize_embeddings.py             # Cosine similarity distribution & PCA visualizations
│   ├── 13_train_test_split.py                 # Synchronized stratified 80/20 train/test splitter
│   └── generate_biometric_reports.py          # Master benchmark suite (ROC-AUC, EER, TAR@FAR, PAD)
├── data_analysis.ipynb                        # Pre-SMOTE vs. Post-SMOTE statistical EDA & t-SNE
├── smote_balancing.ipynb                      # SMOTE dataset balancing pipeline & manifold analysis
├── requirements.txt                           # Python project dependencies
└── README.md                                  # Repository documentation
```

---

## 🧬 Biometric Encoding Models & Feature Extraction

| Modality Stream | Deep Architecture | Pretraining Corpus | Latent Dim | Input Processing & Normalization |
|---|---|---|:---:|---|
| **Face (High-Dim)** | DeepFace / VGG-Face Deep CNN | VGGFace (Millions of faces) | **4,096-d** | Layer `fc7` activations, $L_2$ unit-sphere normalized |
| **Face (Compact)** | Inception-ResNet-v1 (`facenet-pytorch`) | VGGFace2 (3.31M images, 9.1k IDs) | **512-d** | MTCNN landmark detection, $160 \times 160$ crop, $L_2$ normalized |
| **Voice (Speech)** | SpeechBrain ECAPA-TDNN | VoxCeleb 1 & 2 (>1M utterances) | **192-d** | 16 kHz mono, $\ge 1.0\text{ s}$, 80-dim log-mel filterbanks, $L_2$ normalized |
| **Multimodal** | Weighted Score-Level Fusion | Aligned biometric test identities | **Score** | Min-Max score normalization ($[0, 1]$), $w_{\text{face}} = 0.55, w_{\text{voice}} = 0.45$ |

---

## ⚖️ Class Balancing with SMOTE: Before vs. After Analysis

Natural biometric collections frequently suffer from severe sample imbalance (certain active subjects contribute hundreds of images/recordings while others have very few). Synthetic Minority Over-sampling Technique (SMOTE) was applied strictly to the training splits (`smote_face_train.csv`, `smote_voice_train.csv`) to establish uniform class representation without test-set data leakage.

### Quantitative Comparison: Before vs. After SMOTE

| Characteristic | 👤 Face: Before SMOTE | 👤 Face: After SMOTE | 🎙️ Voice: Before SMOTE | 🎙️ Voice: After SMOTE |
|---|---|---|---|---|
| **Training File** | `train_data_face.csv` | `smote_face_train.csv` | `train_voice.csv` | `smote_voice_train.csv` |
| **Total Samples** | 11,671 samples | **28,490 samples** | 11,671 samples | **28,490 samples** |
| **Synthetic Vectors Added** | 0 | **+16,819 synthetic** | 0 | **+16,819 synthetic** |
| **Subject Count** | 70 identities | 70 identities | 70 identities | 70 identities |
| **Min Samples / Class** | **4 samples** | **407 samples** | **4 samples** | **407 samples** |
| **Max Samples / Class** | 407 samples | 407 samples | 407 samples | 407 samples |
| **Sample Distribution** | Skewed ($\sigma = 112.4$) | **Uniform: 407 / ID ($\sigma = 0$)** | Skewed ($\sigma = 112.4$) | **Uniform: 407 / ID ($\sigma = 0$)** |
| **Feature Dimensionality** | 4,096 dimensions | 4,096 dimensions | 192 dimensions | 192 dimensions |
| **$L_2$ Norm Mean ($\pm \sigma$)**| $1.0000 \pm 0.0000$ | **$0.9998 \pm 0.0014$** | $1.0000 \pm 0.0000$ | **$0.9996 \pm 0.0019$** |
| **Separation Index ($d'$)** | 2.44 | **2.61** | 3.28 | **3.55** |

- **Manifold Preservation**: Vector magnitude checks verify that synthetic vectors remain tightly bound to the unit hyper-sphere ($||\mathbf{x}||_2 \approx 1.0$). PCA and t-SNE projections in [`smote_balancing.ipynb`](smote_balancing.ipynb) confirm synthetic points stay strictly within real intra-class manifolds.
- **EDA Alignment Bug Resolution**: Documented the resolution of artifactually overlapping voice curves in early EDA (`data_analysis.ipynb`) caused by face row label misalignment. Aligning by true speaker labels revealed clear genuine ($\mu = 0.6051$) vs. impostor ($\mu = 0.0929$) separation.

---

## 🔗 Multimodal Score-Level Fusion Methodology

Multimodal biometric systems overcome single-sensor limitations (such as facial lighting variation or acoustic room noise) by consolidating multi-sensor evidence.

### 1. Architectural Stage: Score-Level Fusion
- **Feature Concatenation Avoidance**: Concatenating Face (4,096-d) and Voice (192-d) creates severe dimensionality asymmetry (face features would dominate 95.5% of the vector). Score fusion allows each specialized encoder to operate independently and supports asynchronous capture.

### 2. Min-Max Score Normalization
Cosine similarities for face and voice have differing means and dynamic ranges. Both similarity streams are mapped into $[0.0, 1.0]$:
$$S'_{\text{face}} = \frac{S_{\text{face}} - \min(S_{\text{face}})}{\max(S_{\text{face}}) - \min(S_{\text{face}})}, \quad S'_{\text{voice}} = \frac{S_{\text{voice}} - \min(S_{\text{voice}})}{\max(S_{\text{voice}}) - \min(S_{\text{voice}})}$$

### 3. Convex Linear Combination
$$S_{\text{fused}} = w_{\text{face}} \cdot S'_{\text{face}} + w_{\text{voice}} \cdot S'_{\text{voice}}, \quad \text{where } w_{\text{face}} = 0.55, \, w_{\text{voice}} = 0.45$$
- **Weight Calibration Rationale**: Face similarity provides slightly stronger upper-end separation on clean genuine matches, while voice similarity provides sharper near-zero impostor rejection. A 55/45 balance achieved the global minimum Equal Error Rate across validation grid searches.

---

## 📊 Biometric Benchmark Results

### 1. Verification Performance
Evaluated on held-out test sets across **15,347 Genuine Pairs** and **46,041 Impostor Pairs** (68 common subjects):

| Modality Stream | Features | ROC-AUC | Equal Error Rate (EER) | Optimal Cutoff | TAR @ FAR=1.0% | TAR @ FAR=0.1% | TAR @ FAR=0.01% | Separation ($d'$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Face Only** | 4,096-d | **97.50%** | **8.44%** | `0.2574` | **71.47%** | **50.79%** | **37.30%** | `2.61` |
| **Voice Only** | 192-d ECAPA | **98.57%** | **4.68%** | `0.2859` | **90.80%** | **80.02%** | **61.37%** | `3.55` |
| **Multimodal Combined Fusion** | Score Fusion | **99.69%** | **2.29%** | `0.3727` | **96.33%** | **86.91%** | **72.87%** | **`4.14`** |

> **Key Finding**: Multimodal score fusion cuts the Equal Error Rate by **51.1%** down to **2.29% EER** and reaches **96.33% TAR @ 1% FAR**, significantly outperforming any single modality operating in isolation.

---

### 2. VTU-DTU Dataset Benchmark Evaluation

Evaluations on the VTU-DTU evaluation partition ([`embedding_scores/VTU-DTU/`](embedding_scores/VTU-DTU/)):

| Benchmark Set | Architecture | Scope | ROC-AUC | EER | Optimal Threshold | Genuine Cosine Sim | Impostor Cosine Sim |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **VTU-DTU Face** | FaceNet Inception-ResNet-v1 (512-d) | 300 identities, 2,999 vectors | **99.73%** | **0.78%** | `0.7787` | $\mu = 0.9427 \pm 0.041$ | $\mu = 0.3909 \pm 0.076$ |
| **VTU-DTU Voice** | SpeechBrain ECAPA-TDNN (192-d) | 200 speakers, 30,954 vectors | **88.61%** | **19.82%** | `0.3748` | $\mu = 0.4994 \pm 0.178$ | $\mu = 0.2435 \pm 0.112$ |

---

### 3. Presentation Attack Detection (ISO/IEC 30107-3)

Evaluated across bona fide presentations vs. synthetic, replay, and perturbed attack presentations:

| Modality PAD | PAD ROC-AUC | EER (APCER = BPCER) | Minimum ACER | APCER @ BPCER=1% | APCER @ BPCER=5% |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Face Anti-Spoof** | **56.01%** | **45.97%** | **44.57%** | **99.40%** | **95.97%** |
| **Voice Anti-Spoof** | **55.78%** | **46.52%** | **45.77%** | **98.53%** | **92.13%** |
| **Combined Multimodal PAD** | **57.17%** | **45.43%** | **44.88%** | **95.80%** | **89.97%** |

- **APCER**: Attack Presentation Classification Error Rate (proportion of attacks accepted as genuine).
- **BPCER**: Bona Fide Presentation Classification Error Rate (proportion of bona fide presentations rejected).
- **ACER**: Average Classification Error Rate: $\frac{\text{APCER} + \text{BPCER}}{2}$.

---

## 🔬 Exploratory Data Analysis & Notebooks

- **[`data_analysis.ipynb`](data_analysis.ipynb)**: Detailed statistical analysis, class balance verification, intra/inter-class cosine similarity distributions, and t-SNE / PCA manifold visualizations.
- **[`smote_balancing.ipynb`](smote_balancing.ipynb)**: Applies Synthetic Minority Over-sampling Technique (SMOTE) to rebalance sample counts per identity, outputting balanced training sets (`smote_face_train.csv` and `smote_voice_train.csv`) and validating latent hyper-sphere geometry.

---

## 🚀 Getting Started

### Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/SaiTejaMantha03/BiFed-S.git
cd BiFed-S

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Pipeline Execution

Execute the end-to-end multimodal workflow:

```bash
# 1. Filter English voice samples
python scripts/02_filter_voice_en.py

# 2. Extract Face Embeddings (MTCNN + FaceNet)
python scripts/05_embed_faces.py

# 3. Extract Voice Embeddings (ECAPA-TDNN)
python scripts/06_embed_voice.py

# 4. Validate embeddings & generate manifest
python scripts/07_validate_embeddings.py

# 5. Create synchronized 80/20 train/test splits
python scripts/13_train_test_split.py

# 6. Generate comprehensive verification and PAD benchmark reports
python scripts/generate_biometric_reports.py
```

---

## 📑 Generated Artifacts & Reports

All detailed quantitative reports and high-resolution plots are stored in [`embedding_scores/`](embedding_scores/):

### Markdown Reports
- **Executive Master Report**: [`embedding_scores/biometric_benchmark_master_report.md`](embedding_scores/biometric_benchmark_master_report.md)
- **Face Verification Report**: [`embedding_scores/face_verification_report.md`](embedding_scores/face_verification_report.md)
- **Voice Verification Report**: [`embedding_scores/voice_verification_report.md`](embedding_scores/voice_verification_report.md)
- **Multimodal Verification Report**: [`embedding_scores/multimodal_verification_report.md`](embedding_scores/multimodal_verification_report.md)
- **Presentation Attack Detection Report**: [`embedding_scores/spoof_detection_report.md`](embedding_scores/spoof_detection_report.md)
- **VTU-DTU Accuracy Report**: [`embedding_scores/VTU-DTU/embedding_accuracy_report_face.md`](embedding_scores/VTU-DTU/embedding_accuracy_report_face.md)
- **VTU-DTU Multimodal Report**: [`embedding_scores/VTU-DTU/embedding_accuracy_report.md`](embedding_scores/VTU-DTU/embedding_accuracy_report.md)

### Visual Plots
- **Face ROC, DET & Score Distributions**: `embedding_scores/face_roc_eer_tar.png`
- **Voice ROC, DET & Score Distributions**: `embedding_scores/voice_roc_eer_tar.png`
- **Multimodal Comparative Performance Curves**: `embedding_scores/multimodal_roc_eer_tar.png`
- **Presentation Attack Detection Tradeoff**: `embedding_scores/spoof_detection_pad_analysis.png`
- **VTU-DTU Face Accuracy Distribution**: `embedding_scores/VTU-DTU/face_accuracy.png`
- **VTU-DTU Voice Accuracy Distribution**: `embedding_scores/VTU-DTU/voice_accuracy.png`

---

## 📜 License

This project is licensed under the MIT License.
