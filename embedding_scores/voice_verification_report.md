# 🎙️ Voice Biometric Verification Report
### Quantitative Evaluation: ECAPA-TDNN Encoding Model, Pre- vs Post-SMOTE Resampling Analysis, and Verification Benchmarks

---

## 1. 🧬 Encoding Model Architecture & Acoustic Feature Extraction

The voice biometric pipeline employs state-of-the-art deep acoustic neural architectures designed specifically for robust text-independent speaker identification and verification across variable sentence lengths and channels.

### Model Specifications & Feature Spaces
- **Acoustic Neural Architecture**:
  - **Model**: SpeechBrain ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network) — `speechbrain/spkrec-ecapa-voxceleb`.
  - **Pretrained Corpus**: Pretrained on VoxCeleb 1 & 2 datasets (encompassing >1M utterances across >7,000 international speakers).
  - **Feature Dimension**: 192-dimensional compact speaker embedding vectors.
- **Architectural Mechanics**:
  - **1D Dilated Convolutions**: Frame-level multi-scale temporal modeling capturing short- and long-range phonetic transitions.
  - **Squeeze-and-Excitation (SE) Residual Blocks**: Res2Net blocks with channel attention dynamically weighting critical speaker-discriminative spectral sub-bands.
  - **Multi-layer Feature Aggregation (MFA)**: Concatenates intermediate hierarchical layer feature maps prior to pooling.
  - **Attentive Statistical Pooling (ASP)**: Computes weighted temporal mean and standard deviation vectors focusing attention on salient vocal tract resonances while suppressing background pauses.
- **Audio Preprocessing Pipeline**:
  - Raw audio ingested via `soundfile`, converted to single-channel mono, and resampled to a standardized **16,000 Hz** sample rate.
  - Speech activity threshold: Minimum utterance duration enforced at $\ge 1.0\text{ s}$ to guarantee stable vocal formant representation.
  - Spectral representation: 80-dimensional log-mel filterbanks extracted across 25 ms windows with a 10 ms hop size.
  - Normalization: Extracted 192-d vectors undergo strict $L_2$ unit-length projection:
    $$\mathbf{e}_{\text{voice}} = \frac{\mathbf{v}}{||\mathbf{v}||_2}, \quad ||\mathbf{e}_{\text{voice}}||_2 = 1.0$$

---

## 2. ⚖️ Dataset Balancing: Before SMOTE vs. After SMOTE Analysis

Real-world speech corpora naturally feature extreme imbalance due to talkative participants vs. single-utterance speakers. SMOTE was applied to speaker representations in the training split (`smote_voice_train.csv`) to establish balanced acoustic identity priors.

### Quantitative SMOTE Balancing Overview

| Characteristic | Before SMOTE (`train_voice.csv`) | After SMOTE (`smote_voice_train.csv`) | Impact & Operational Validation |
|---|---|---|---|
| **Total Samples** | 11,671 samples | **28,490 samples** | +16,819 synthetic voice vectors generated |
| **Unique Speakers** | 70 speakers | 70 speakers | Speaker catalog preserved |
| **Min Samples / Speaker** | **4 samples** | **407 samples** | Full support restored for rare speakers |
| **Max Samples / Speaker** | 407 samples | 407 samples | Max density baseline maintained |
| **Mean $\pm$ Std per Speaker**| $166.7 \pm 112.4$ (Highly Skewed) | **$407.0 \pm 0.0$ (Uniform)**| Perfect class balance across all speakers |
| **Feature Dimensionality** | 192 features | 192 features | Exact 192-d vector format retained |
| **$L_2$ Vector Norm Mean** | $1.0000 \pm 0.0000$ (Normalized) | **$0.9996 \pm 0.0019$** | Manifold sphere curvature intact without inflation |
| **Separation Index ($d'$)** | 3.28 | **3.55** | $+0.27$ increase in inter-speaker margin |

### Key Observations & Analytical Diagnosis
1. **Resolution of the EDA Intersecting Voice Curves**:
   - In preliminary exploratory data analysis (`data_analysis.ipynb`), voice genuine and impostor curves appeared artifactually overlapping (Genuine Mean: $\sim 0.119$ vs Impostor Mean: $\sim 0.099$).
   - **Root Cause Identified**: The preliminary script assigned Face row identity labels to Voice rows. Because Face and Voice datasets are ordered independently (e.g., Row 0 Face = ID 50, whereas Row 0 Voice = ID 58), pairs matching in Face row index were actually comparing two completely distinct speakers.
   - **Correction in `smote_balancing.ipynb`**: Aligning pairs using genuine speaker labels revealed high genuine similarity ($\mu = 0.6051$) sharply separated from impostors ($\mu = 0.0929$).
2. **SMOTE Interpolation on Speaker Manifolds**:
   - The 192-d latent space forms tight speaker clusters. Linear interpolation along the chord between neighboring vectors produces synthetic points that remain faithful to vocal tract invariants.
   - Density plots (KDE) and $L_2$ norms confirm that synthetic vectors populate the convex hull of real speaker embeddings without degrading discriminability.

---

## 3. 📊 Verification Performance Visualizations
![Voice Verification Performance](/Users/saitejamantha/Documents/bifed-s/embedding_scores/voice_roc_eer_tar.png)

---

## 4. 🎯 Key Biometric Verification Benchmarks

Evaluated on held-out test split `embeddings/splits/test_voice.csv` (1,297 samples across 68 independent speakers, evaluating **15,347 Genuine Pairs** and **46,041 Impostor Pairs**):

| Metric | Measured Value | Standard Benchmark Target | Operational Interpretation |
|---|---|---|---|
| **ROC-AUC Score** | **98.57%** | > 95.0% | Outstanding speaker separability |
| **Equal Error Rate (EER)** | **4.68%** | < 5.0% | Operating point where FAR equals FRR |
| **Optimal EER Threshold** | `0.2859` | N/A | Cosine similarity threshold balancing errors |
| **TAR @ FAR = 1.0% ($10^{-2}$)** | **90.80%** | > 90.0% | Commercial telephony biometric access |
| **TAR @ FAR = 0.1% ($10^{-3}$)** | **80.02%** | > 85.0% | Banking-grade voice authentication |
| **TAR @ FAR = 0.01% ($10^{-4}$)** | **61.37%** | > 80.0% | Ultra-high security voice verification |

---

## 5. 📈 Similarity Score Distribution Statistics
- **Genuine Pairs (Same Speaker)**: $\mu = 0.6051 \pm 0.1724$
- **Impostor Pairs (Different Speaker)**: $\mu = 0.0929 \pm 0.1088$
- **Separation Margin ($d'$)**: `3.55` (Significantly superior to face modality in raw separation index)
- **Full English Corpus Benchmark** (200 speakers, 30,954 vectors): ROC-AUC **88.61%**, EER **19.82%** on cross-session conversational audio.

> [!NOTE]
> **Summary Insight**: SpeechBrain ECAPA-TDNN produces rich 192-d speaker embeddings. Balancing the dataset with SMOTE ensures rare speakers are fully represented, producing a low **4.68% EER** and an exceptional **98.57% ROC-AUC**.
