# 🎙️ Voice Biometric Verification Report
### Quantitative Evaluation: ROC-AUC, EER, FAR, FRR, and TAR@FAR Benchmarks

- **Embedding Space**: 192-dimensional ECAPA-TDNN speaker embeddings
- **Test Set**: `embeddings/splits/test_voice.csv` (1,297 samples across 68 speakers)
- **Evaluation Trials**: 15,347 Genuine Pairs, 46,041 Impostor Pairs

---

## 📊 Performance Visualizations
![Voice Verification Performance](/Users/saitejamantha/Documents/bifed-s/embedding_scores/voice_roc_eer_tar.png)

---

## 🎯 Key Verification Metrics Table

| Metric | Measured Value | Standard Benchmark Target | Operational Interpretation |
|---|---|---|---|
| **ROC-AUC Score** | **98.57%** | > 95.0% | Outstanding speaker separability |
| **Equal Error Rate (EER)** | **4.68%** | < 5.0% | Operating point where FAR equals FRR |
| **Optimal EER Threshold** | `0.2859` | N/A | Cosine similarity threshold balancing errors |
| **TAR @ FAR = 1.0% ($10^{-2}$)** | **90.80%** | > 90.0% | Commercial biometric telephony access |
| **TAR @ FAR = 0.1% ($10^{-3}$)** | **80.02%** | > 85.0% | Banking-grade voice authentication |
| **TAR @ FAR = 0.01% ($10^{-4}$)** | **61.37%** | > 80.0% | High security voice verification |

---

## 📈 Similarity Distribution Statistics
- **Genuine Pairs (Same Speaker)**: $\mu = 0.6051 \pm 0.1724$
- **Impostor Pairs (Different Speaker)**: $\mu = 0.0929 \pm 0.1088$
- **Separation Margin (d')**: `3.55`
