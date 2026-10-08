#!/usr/bin/env python3
"""
Step 8: Visualize embedding accuracy, compute EER/AUC, and generate validation graphs.
Usage: python scripts/08_visualize_embeddings.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

# Set dark/modern plot style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings')
PLOT_DIR = os.path.join(EMB_DIR, 'plots')


def compute_pair_similarities(embeddings_dict, max_pairs_per_type=10000):
    """
    Sample positive (genuine, same-person) and negative (impostor, cross-person) pair similarities.
    Returns: genuine_sims, impostor_sims
    """
    persons = list(embeddings_dict.keys())
    genuine_sims = []
    impostor_sims = []

    # 1. Genuine pairs (same person)
    for pid, embs in embeddings_dict.items():
        n = len(embs)
        if n < 2:
            continue
        # Normalize vectors just in case
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embs = embs / norms

        # Cosine similarity matrix for this person's embeddings
        sim_matrix = np.dot(norm_embs, norm_embs.T)
        triu_indices = np.triu_indices(n, k=1)
        sims = sim_matrix[triu_indices]
        genuine_sims.extend(sims)

    # 2. Impostor pairs (different persons)
    np.random.seed(42)
    # Randomly pick pairs of different persons
    for _ in range(max_pairs_per_type):
        p1, p2 = np.random.choice(persons, size=2, replace=False)
        e1 = embeddings_dict[p1][np.random.randint(len(embeddings_dict[p1]))]
        e2 = embeddings_dict[p2][np.random.randint(len(embeddings_dict[p2]))]

        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        sim = np.dot(e1, e2) / (n1 * n2 if n1 > 0 and n2 > 0 else 1.0)
        impostor_sims.append(sim)

    # Limit genuine pairs to max_pairs_per_type for balance
    if len(genuine_sims) > max_pairs_per_type:
        genuine_sims = list(np.random.choice(genuine_sims, size=max_pairs_per_type, replace=False))

    return np.array(genuine_sims), np.array(impostor_sims)


def calculate_eer(genuine_sims, impostor_sims):
    """Calculate Equal Error Rate (EER) and optimal threshold."""
    y_true = np.concatenate([np.ones_like(genuine_sims), np.zeros_like(impostor_sims)])
    y_scores = np.concatenate([genuine_sims, impostor_sims])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # EER is point where FPR == FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    optimal_threshold = thresholds[eer_idx]
    roc_auc = auc(fpr, tpr)

    return eer, optimal_threshold, roc_auc, fpr, tpr


def plot_modality_evaluation(npz_path, name, filename_prefix):
    """Generate evaluation plots (Distribution, ROC, t-SNE clusters) for a dataset."""
    if not os.path.exists(npz_path):
        print(f"⚠️  {name} embeddings file not found: {npz_path}")
        return None

    data = np.load(npz_path)
    embeddings_dict = {k: data[k] for k in data.keys()}

    if len(embeddings_dict) == 0:
        print("   ⚠️  No identities found in this embedding file yet (generation in progress).")
        return None

    # Calculate similarities and metrics
    genuine_sims, impostor_sims = compute_pair_similarities(embeddings_dict)
    eer, threshold, roc_auc, fpr, tpr = calculate_eer(genuine_sims, impostor_sims)

    print(f"   Genuine Pairs:  {len(genuine_sims):,} (Mean Cosine Sim: {np.mean(genuine_sims):.4f})")
    print(f"   Impostor Pairs: {len(impostor_sims):,} (Mean Cosine Sim: {np.mean(impostor_sims):.4f})")
    print(f"   ✨ ROC-AUC Score: {roc_auc*100:.2f}%")
    print(f"   ✨ Equal Error Rate (EER): {eer*100:.2f}% at Threshold = {threshold:.4f}")

    # Create 3-panel Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Cosine Similarity Distribution
    sns.kdeplot(genuine_sims, ax=axes[0], color='#2ecc71', fill=True, alpha=0.4, label=f'Genuine (Same ID, µ={np.mean(genuine_sims):.2f})')
    sns.kdeplot(impostor_sims, ax=axes[0], color='#e74c3c', fill=True, alpha=0.4, label=f'Impostor (Diff ID, µ={np.mean(impostor_sims):.2f})')
    axes[0].axvline(threshold, color='#3498db', linestyle='--', linewidth=2, label=f'Opt Threshold ({threshold:.2f})')
    axes[0].set_title(f'{name} — Cosine Similarity Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Density')
    axes[0].legend(loc='upper left')

    # Panel 2: ROC Curve
    axes[1].plot(fpr, tpr, color='#9b59b6', lw=2.5, label=f'ROC Curve (AUC = {roc_auc*100:.2f}%)')
    axes[1].plot([0, 1], [0, 1], color='#7f8c8d', linestyle=':', lw=1.5)
    axes[1].plot([0, 1], [1, 0], color='#e67e22', linestyle='--', lw=1, label=f'EER Line ({eer*100:.2f}%)')
    axes[1].set_title(f'{name} — ROC Verification Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate (FPR)')
    axes[1].set_ylabel('True Positive Rate (TPR)')
    axes[1].legend(loc='lower right')

    # Panel 3: PCA / Embedding Clusters (Top 10 Identities)
    top_pids = sorted(embeddings_dict.keys(), key=lambda k: len(embeddings_dict[k]), reverse=True)[:10]
    X_sample = []
    y_sample = []
    for pid in top_pids:
        X_sample.extend(embeddings_dict[pid])
        y_sample.extend([pid] * len(embeddings_dict[pid]))

    X_sample = np.array(X_sample)
    if len(X_sample) > 0:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_sample)

        scatter = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=pd_factorize(y_sample), cmap='tab10', alpha=0.8, edgecolors='none', s=25)
        axes[2].set_title(f'{name} — PCA 2D Clustering (Top 10 IDs)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Principal Component 1')
        axes[2].set_ylabel('Principal Component 2')

    plt.tight_layout()
    out_img = os.path.join(PLOT_DIR, f'{filename_prefix}_accuracy.png')
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()

    print(f"   🖼️ Saved plot to: {out_img}")
    return {
        'modality': name,
        'auc': float(roc_auc),
        'eer': float(eer),
        'optimal_threshold': float(threshold),
        'mean_genuine_sim': float(np.mean(genuine_sims)),
        'mean_impostor_sim': float(np.mean(impostor_sims)),
        'plot_file': out_img
    }


def pd_factorize(labels):
    """Helper to convert string labels into categorical integers."""
    unique = list(dict.fromkeys(labels))
    mapping = {k: i for i, k in enumerate(unique)}
    return np.array([mapping[k] for k in labels])


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("=" * 60)
    print("📈 Biometric Embedding Accuracy Evaluation & Visualization")
    print("=" * 60)

    face_stats = plot_modality_evaluation(
        os.path.join(EMB_DIR, 'face_embeddings.npz'),
        'Face (FaceNet)',
        'face'
    )

    voice_stats = plot_modality_evaluation(
        os.path.join(EMB_DIR, 'voice_embeddings.npz'),
        'Voice (ECAPA-TDNN)',
        'voice'
    )

    print(f"\n{'='*60}")
    print("🎉 Accuracy Report Summary")
    print(f"{'='*60}")
    if face_stats:
        print(f"  Face Verification Accuracy (AUC): {face_stats['auc']*100:.2f}% | EER: {face_stats['eer']*100:.2f}%")
    if voice_stats:
        print(f"  Voice Verification Accuracy (AUC): {voice_stats['auc']*100:.2f}% | EER: {voice_stats['eer']*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
