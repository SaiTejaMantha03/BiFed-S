# 👤 Face Biometric Verification Report
### Quantitative Evaluation: ROC-AUC, EER, FAR, FRR, and TAR@FAR Benchmarks

- **Embedding Space**: 4,096-dimensional FaceNet / DeepFace embeddings
- **Test Set**: `embeddings/splits/test_face_4096dim.csv` (1,297 samples across 68 identities)
- **Evaluation Trials**: 15,347 Genuine Pairs, 46,041 Impostor Pairs

---

## 📊 Performance Visualizations
![Face Verification Performance](/Users/saitejamantha/Documents/bifed-s/embedding_scores/face_roc_eer_tar.png)

---

## 🎯 Key Verification Metrics Table

| Metric | Measured Value | Standard Benchmark Target | Operational Interpretation |
|---|---|---|---|
| **ROC-AUC Score** | **97.50%** | > 95.0% | Excellent global discriminative separability |
| **Equal Error Rate (EER)** | **8.44%** | < 10.0% | Operating point where FAR equals FRR |
| **Optimal EER Threshold** | `0.2574` | N/A | Cosine similarity threshold balancing errors |
| **TAR @ FAR = 1.0% ($10^{-2}$)** | **71.47%** | > 85.0% | High security operational setting |
| **TAR @ FAR = 0.1% ($10^{-3}$)** | **50.79%** | > 75.0% | Banking & border control threshold |
| **TAR @ FAR = 0.01% ($10^{-4}$)** | **37.30%** | > 65.0% | Ultra-high security military threshold |

---

## 📈 Similarity Distribution Statistics
- **Genuine Pairs (Same Identity)**: $\mu = 0.5257 \pm 0.1957$
- **Impostor Pairs (Different Identity)**: $\mu = 0.1350 \pm 0.0811$
- **Separation Margin (d')**: `2.61` (High discriminative index)
