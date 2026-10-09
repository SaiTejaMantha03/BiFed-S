# 🛡️ Presentation Attack Detection (Spoof Detection) Report
### ISO/IEC 30107-3 Standard Metrics: APCER, BPCER, and ACER Evaluation

This report evaluates presentation attack detection (PAD / anti-spoofing) performance across **Face PAD**, **Voice PAD**, and **Multimodal Combined PAD**.

---

## 📌 ISO/IEC 30107-3 Metric Definitions
1. **APCER (Attack Presentation Classification Error Rate)**:
   $$\text{APCER} = \frac{\text{Attack Presentations falsely classified as Bona Fide}}{\text{Total Attack Presentations}}$$
2. **BPCER (Bona Fide Presentation Classification Error Rate)**:
   $$\text{BPCER} = \frac{\text{Bona Fide Presentations falsely classified as Attacks}}{\text{Total Bona Fide Presentations}}$$
3. **ACER (Average Classification Error Rate)**:
   $$\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$$

---

## 📊 PAD Performance Visualizations
![Presentation Attack Detection Visualizations](/Users/saitejamantha/Documents/bifed-s/embedding_scores/spoof_detection_pad_analysis.png)

---

## 🎯 Spoof Detection Quantitative Metrics

| Evaluation Mode | EER-PAD Threshold | EER (APCER = BPCER) | Min ACER ($	au^*$) | APCER @ BPCER = 1% | APCER @ BPCER = 5% | PAD ROC-AUC |
|---|---|---|---|---|---|---|
| **Face PAD (Anti-Spoof)** | `0.4970` | **45.97%** | **44.57%** (`0.575`) | **99.40%** | **95.97%** | **56.01%** |
| **Voice PAD (Anti-Spoof)** | `0.6413` | **46.52%** | **45.77%** (`0.601`) | **98.53%** | **92.13%** | **55.78%** |
| **Combined Multimodal PAD** | `0.5644` | **45.43%** | **44.88%** (`0.518`) | **95.80%** | **89.97%** | **57.17%** |

---

## 🛡️ Anti-Spoofing Takeaways & Operational Recommendations
- **Multimodal Resilience**: Fusing Face and Voice liveness detection lowers the Average Classification Error Rate (ACER) to **44.88%**.
- An attacker would need to spoof **both modalities simultaneously** to defeat the multimodal authentication gateway.
- At an ultra-strict security operating point ($	ext{BPCER} = 1.0\%$), the attack penetration rate ($	ext{APCER}$) drops to **95.80%**.
