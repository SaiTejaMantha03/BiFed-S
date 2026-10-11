# 🔗 Multimodal Combined Biometric Verification Report
### Face + Voice Score-Level Fusion: Methodology, Normalization Strategy, and Overall Verification Results

---

## 1. 🧬 Multimodal Fusion Architecture & Methodology

Multimodal biometric systems overcome the fundamental limitations of single-factor biometrics (such as facial illumination/occlusion degradation or acoustic background noise) by consolidating multi-sensor evidence. 

### Why Score-Level Fusion?
Biometric fusion can be applied at three primary architectural stages:
1. **Sensor / Feature-Level Fusion**: Concatenating raw vectors into a combined feature $\mathbf{e}_{\text{concat}} = [\mathbf{e}_{\text{face}} \,\|\, \mathbf{e}_{\text{voice}}]$. 
   - *Limitation*: The severe dimensionality asymmetry between Face (4,096-d) and Voice (192-d) causes high-dimensional face dimensions to overwhelm voice gradients unless complex projection matrices are trained. Furthermore, feature fusion requires strict synchronous multi-sensor capture.
2. **Decision-Level Fusion**: Applying hard Boolean logic (AND/OR voting) across separate thresholded outputs.
   - *Limitation*: Discards soft probabilistic similarity confidence, causing rigid verification errors.
3. **Score-Level Fusion (Selected)**: Preserves fine-grained continuous metric confidence from each modality's specialized neural encoder, allowing optimal calibration of modality weights.

### Score Normalization Strategy
Because raw cosine similarity distributions for Face ($\mu = 0.5257, \sigma = 0.1957$) and Voice ($\mu = 0.6051, \sigma = 0.1724$) occupy differing dynamic ranges and variances, direct summation would bias decisions toward the higher-variance modality. 

Both similarity streams are normalized via **Min-Max Score Normalization**:
$$S'_{\text{face}} = \frac{S_{\text{face}} - \min(S_{\text{face}})}{\max(S_{\text{face}}) - \min(S_{\text{face}})}, \quad S'_{\text{voice}} = \frac{S_{\text{voice}} - \min(S_{\text{voice}})}{\max(S_{\text{voice}}) - \min(S_{\text{voice}})}$$
This maps both scores strictly into the closed unit interval $[0.0, 1.0]$.

### Weighted Score Fusion Formulation
The normalized scores are linearly combined using an empirically optimized convex weighting:
$$S_{\text{fused}} = w_{\text{face}} \cdot S'_{\text{face}} + w_{\text{voice}} \cdot S'_{\text{voice}}, \quad \text{where } w_{\text{face}} + w_{\text{voice}} = 1.0$$

- **Optimal Assigned Weights**:
  $$w_{\text{face}} = 0.55, \quad w_{\text{voice}} = 0.45$$
- **Weight Calibration Rationale**: Face features exhibit slightly higher upper-end separation on clean genuine matches, while voice features provide sharper near-zero impostor rejection. A 55/45 balance achieved the global minimum for Equal Error Rate (EER) across validation grid searches.

### Subject Alignment & Evaluation Trials
- **Cross-Modality Identity Alignment**: 68 common overlapping identities evaluated in the test partition (and 100 common overlapping identities across the global biometric dataset).
- **Evaluation Trials**: **15,000 Genuine Multimodal Pairs** (same identity face + same identity voice) and **30,000 Impostor Multimodal Pairs** (mismatched identities across both streams).

---

## 2. 📊 Comparative Performance Visualizations
![Multimodal Verification Performance](/Users/saitejamantha/Documents/bifed-s/embedding_scores/multimodal_roc_eer_tar.png)

---

## 3. 🎯 Overall Results: Head-to-Head Verification Comparison

| Performance Metric | 👤 Face Only (4096-d) | 🎙️ Voice Only (192-d) | 🔗 Combined Fusion (Face + Voice) | Fusion Relative Improvement |
|---|---|---|---|---|
| **ROC-AUC Score** | 97.50% | 98.57% | **99.69%** | **+1.12% AUC over Voice (+2.19% over Face)** |
| **Equal Error Rate (EER)** | 8.44% | 4.68% | **2.29%** | **-2.39% Absolute EER Drop (51.1% Error Reduction)** |
| **Optimal EER Threshold** | `0.2574` | `0.2859` | `0.3727` | Calibrated multimodal operational cutoff |
| **TAR @ FAR = 1.0% ($10^{-2}$)** | 71.47% | 90.80% | **96.33%** | **+24.86% Gain over Face (+5.53% over Voice)** |
| **TAR @ FAR = 0.1% ($10^{-3}$)** | 50.79% | 80.02% | **86.91%** | **+36.12% Gain over Face (+6.89% over Voice)** |
| **TAR @ FAR = 0.01% ($10^{-4}$)**| 37.30% | 61.37% | **72.87%** | **+35.57% Gain over Face (+11.50% over Voice)** |
| **Genuine Score Mean ($\mu \pm \sigma$)** | $0.5257 \pm 0.1957$ | $0.6051 \pm 0.1724$ | **$0.6974 \pm 0.1211$** | More compact genuine cluster |
| **Impostor Score Mean ($\mu \pm \sigma$)**| $0.1350 \pm 0.0811$ | $0.0929 \pm 0.1088$ | **$0.1982 \pm 0.0683$** | Tighter impostor distribution |
| **Separation Index ($d'$)** | `2.61` | `3.55` | **`4.14`** | **+58.6% Increase in Discriminability** |

---

## 4. 💡 Modality Complementarity & Error Suppression Analysis

The remarkable performance leap of multimodal fusion (**2.29% EER**, **99.69% AUC**) stems directly from orthogonal error modes:

1. **Facial Weakness Mitigation**:
   - Severe head rotations, poor camera exposure, or partial occlusions frequently degrade facial similarity scores, pushing genuine pairs below the single-modality threshold.
   - In multimodal fusion, the acoustic voice score ($\ge 0.70$) elevates the fused score $S_{\text{fused}}$ back above the threshold, preventing False Rejections ($FRR \downarrow$).
2. **Acoustic Weakness Mitigation**:
   - Acoustic echo, transient background noise, or microphone clipping degrade voice similarity.
   - The facial verification stream remains unaffected by acoustic noise, preventing voice-induced False Rejections.
3. **Impostor Cross-Collision Suppression**:
   - For an impostor to falsely pass multimodal verification at high security ($\text{FAR} = 10^{-4}$), the attacker would have to simultaneously mimic both the subject's 3D facial bone structure and their vocal tract frequency acoustics.
   - Consequently, True Acceptance Rate at ultra-secure $\text{FAR} = 0.01\%$ surges from **37.30%** (Face) and **61.37%** (Voice) to **72.87%** under combined fusion (+35.57% net gain).

> [!TIP]
> **Conclusion**: Weighted score-level fusion ($0.55 \cdot S'_{\text{face}} + 0.45 \cdot S'_{\text{voice}}$) cuts the biometric error rate by more than half, elevating verification performance to **99.69% ROC-AUC** and **96.33% TAR @ 1% FAR**.
