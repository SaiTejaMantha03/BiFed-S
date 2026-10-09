# 🧬 BiFed-S: Multimodal Biometric Federated Learning & Verification Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Modality-Face%20%2B%20Voice%20Fusion-orange.svg)](#)

**BiFed-S** is a comprehensive multimodal biometric verification and presentation attack detection (PAD) framework. It includes data preprocessing, deep feature extraction, class-balanced SMOTE resampling, exploratory data analysis (EDA), and standardized ISO/IEC 30107-3 biometric performance evaluation for Face, Voice, and Combined Multimodal biometrics.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Biometric Benchmark Results](#-biometric-benchmark-results)
  - [1. Verification Performance (ROC-AUC, EER, TAR@FAR)](#1-verification-performance)
  - [2. Presentation Attack Detection (APCER, BPCER, ACER)](#2-presentation-attack-detection)
- [Exploratory Data Analysis & SMOTE Balancing](#-exploratory-data-analysis--smote-balancing)
- [Getting Started](#-getting-started)
- [Pipeline Execution](#-pipeline-execution)
- [Generated Artifacts](#-generated-artifacts)

---

## ✨ Key Features

- **Multimodal Embedding Pipeline**:
  - **Face**: Deep facial feature representation extraction (FaceNet & VGG-Face / 4096-d & 512-d).
  - **Voice**: Speaker verification embeddings extracted using **ECAPA-TDNN** (192-d vectors).
- **Synchronized Train/Test Splitting**: Stratified 80/20 train-test splits keeping face and voice identities synchronized.
- **Class Balancing with SMOTE**: Synthetic Minority Over-sampling (`smote_balancing.ipynb`) to address biometric subject imbalance without data leakage.
- **Comprehensive Verification Benchmark Suite** ([`generate_biometric_reports.py`](file:///Users/saitejamantha/Documents/bifed-s/scripts/generate_biometric_reports.py)):
  - Computes **ROC-AUC**, **EER** (Equal Error Rate), **FAR**, **FRR**, and **TAR @ FAR** (1%, 0.1%, 0.01%).
  - Evaluates individual Face, Voice, and Score-level Multimodal Fusion.
- **Presentation Attack Detection (PAD / Anti-Spoof)**:
  - Standardized evaluation under **ISO/IEC 30107-3**: **APCER**, **BPCER**, and **ACER**.

---

## 📁 Repository Structure

```
BiFed-S/
├── data/                                 # Raw biometric datasets (git-ignored)
│   ├── FaceDB/                           # Face dataset
│   └── VoiceDB/                          # Voice dataset (English filtered)
├── embeddings/                           # Generated embeddings & splits
│   ├── face_embeddings.csv               # Extracted face embeddings
│   ├── voice_embeddings.csv              # Extracted ECAPA-TDNN voice embeddings
│   ├── manifest.json                     # Metadata and identity mapping
│   └── splits/                           # Train/Test splits and raw subsets
├── embedding_scores/                     # Evaluation reports & plots
│   ├── biometric_benchmark_master_report.md  # Master evaluation summary
│   ├── face_verification_report.md           # Face verification metrics
│   ├── voice_verification_report.md          # Voice verification metrics
│   ├── multimodal_verification_report.md     # Combined fusion verification metrics
│   ├── spoof_detection_report.md             # PAD / Anti-spoofing report
│   ├── face_roc_eer_tar.png                  # Face ROC, DET & Score distribution plots
│   ├── voice_roc_eer_tar.png                 # Voice ROC, DET & Score distribution plots
│   ├── multimodal_roc_eer_tar.png            # Fusion ROC, DET & Score distribution plots
│   └── spoof_detection_pad_analysis.png      # APCER / BPCER tradeoff curve
├── scripts/                              # Automated pipeline scripts
│   ├── 02_filter_voice_en.py             # Audio language filter
│   ├── 05_embed_faces.py                 # Face detection & feature extraction
│   ├── 06_embed_voice.py                 # Audio resampling & ECAPA-TDNN extraction
│   ├── 07_validate_embeddings.py         # Integrity validation and manifest builder
│   ├── 08_visualize_embeddings.py        # Cosine similarity distribution & PCA plots
│   ├── 13_train_test_split.py            # Synchronized train/test splitter
│   └── generate_biometric_reports.py     # Master benchmark suite (ROC-AUC, EER, PAD)
├── data_analysis.ipynb                   # Pre-SMOTE vs Post-SMOTE EDA & t-SNE / density plots
├── smote_balancing.ipynb                 # SMOTE dataset balancing workflow
├── requirements.txt                      # Dependencies
└── README.md                             # Documentation
```

---

## 📊 Biometric Benchmark Results

### 1. Verification Performance

Evaluation on test sets across genuine and impostor matching pairs:

| Modality | Features | ROC-AUC | Equal Error Rate (EER) | Optimal Threshold | TAR @ FAR=1% | TAR @ FAR=0.1% | TAR @ FAR=0.01% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Face** | 4096-d | **97.50%** | **8.44%** | `0.2574` | **71.47%** | **50.79%** | **37.30%** |
| **Voice** | 192-d ECAPA | **98.57%** | **4.68%** | `0.2859` | **90.80%** | **80.02%** | **61.37%** |
| **Combined Multimodal Fusion** | Score Fusion | **99.69%** | **2.29%** | `0.3727` | **96.33%** | **86.91%** | **72.87%** |

> **Key Finding**: Multimodal score fusion reduces the Equal Error Rate to **2.29%** and achieves a True Acceptance Rate of **96.33% @ 1% FAR**, significantly outperforming any unimodal biometric system.

---

### 2. Presentation Attack Detection (ISO/IEC 30107-3)

Evaluation against spoofing and presentation attacks:

| Modality PAD | PAD ROC-AUC | EER (APCER = BPCER) | Min ACER | APCER @ BPCER=1% | APCER @ BPCER=5% |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Face Anti-Spoof** | **56.01%** | **45.97%** | **44.57%** | **99.40%** | **95.97%** |
| **Voice Anti-Spoof** | **55.78%** | **46.52%** | **45.77%** | **98.53%** | **92.13%** |
| **Multimodal PAD** | **57.17%** | **45.43%** | **44.88%** | **95.80%** | **89.97%** |

- **APCER**: Attack Presentation Classification Error Rate (proportion of spoof presentations accepted as genuine).
- **BPCER**: Bona Fide Presentation Classification Error Rate (proportion of bona fide presentations rejected).
- **ACER**: Average Classification Error Rate: $\frac{\text{APCER} + \text{BPCER}}{2}$.

---

## 🔬 Exploratory Data Analysis & SMOTE Balancing

- **[`data_analysis.ipynb`](data_analysis.ipynb)**: Detailed statistical analysis, class balance verification, intra/inter-class cosine similarity distributions, and t-SNE / PCA manifold visualizations.
- **[`smote_balancing.ipynb`](smote_balancing.ipynb)**: Applies Synthetic Minority Over-sampling Technique (SMOTE) to rebalance sample counts per identity, outputting balanced training sets (`smote_face_train.csv` and `smote_voice_train.csv`).

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/SaiTejaMantha03/BiFed-S.git
cd BiFed-S

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 🛠️ Pipeline Execution

Run the end-to-end workflow:

```bash
# 1. Filter English voice samples
python scripts/02_filter_voice_en.py

# 2. Extract Face Embeddings
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

## 📑 Generated Artifacts

Detailed markdown reports and high-resolution plots are stored in [`embedding_scores/`](embedding_scores/):
- **[`biometric_benchmark_master_report.md`](embedding_scores/biometric_benchmark_master_report.md)**
- **[`face_verification_report.md`](embedding_scores/face_verification_report.md)**
- **[`voice_verification_report.md`](embedding_scores/voice_verification_report.md)**
- **[`multimodal_verification_report.md`](embedding_scores/multimodal_verification_report.md)**
- **[`spoof_detection_report.md`](embedding_scores/spoof_detection_report.md)**
- **Visuals**:
  - `face_roc_eer_tar.png`
  - `voice_roc_eer_tar.png`
  - `multimodal_roc_eer_tar.png`
  - `spoof_detection_pad_analysis.png`

---

## 📜 License
This project is licensed under the MIT License.
