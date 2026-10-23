"""
Module 05 — Évaluation : métriques avancées, matrices de confusion,
courbes ROC, KS statistic, et courbes Lift/Gain.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import FIG_MODELS, RESULTS_DIR
from config.params import DPI, PALETTE
from utils.logger import log


# ═══════════════════════════════════════════════════════════════════
# 5.1 — Matrices de confusion
# ═══════════════════════════════════════════════════════════════════
def plot_confusion_matrices(all_predictions, y_test):
    """Génère les matrices de confusion pour tous les modèles."""
    n_models = len(all_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for i, (name, preds) in enumerate(all_predictions.items()):
        cm = confusion_matrix(y_test, preds["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Non-défaut", "Défaut"],
                    yticklabels=["Non-défaut", "Défaut"],
                    cbar=False, annot_kws={"size": 14})
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Prédit")
        axes[i].set_ylabel("Réel")

    plt.suptitle("Matrices de confusion", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_MODELS, "confusion_matrices.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : confusion_matrices.png")


# ═══════════════════════════════════════════════════════════════════
# 5.2 — Courbes ROC superposées
# ═══════════════════════════════════════════════════════════════════
def plot_roc_curves(all_predictions, y_test):
    """Trace les courbes ROC de tous les modèles sur un seul graphe."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = PALETTE[:len(all_predictions)]

    for (name, preds), color in zip(all_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, preds["y_prob"])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{name} (AUC = {auc_score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (TPR)", fontsize=12)
    ax.set_title("Courbes ROC comparatives", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_MODELS, "roc_curves.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : roc_curves.png")


# ═══════════════════════════════════════════════════════════════════
# 5.3 — Courbes Precision-Recall
# ═══════════════════════════════════════════════════════════════════
def plot_precision_recall_curves(all_predictions, y_test):
    """Trace les courbes Precision-Recall."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = PALETTE[:len(all_predictions)]

    for (name, preds), color in zip(all_predictions.items(), colors):
        precision, recall, _ = precision_recall_curve(y_test, preds["y_prob"])
        ap = average_precision_score(y_test, preds["y_prob"])
        ax.plot(recall, precision, color=color, linewidth=2.2,
                label=f"{name} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Courbes Precision-Recall", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_MODELS, "precision_recall_curves.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : precision_recall_curves.png")


# ═══════════════════════════════════════════════════════════════════
# 5.4 — KS Statistic (Kolmogorov-Smirnov)
# ═══════════════════════════════════════════════════════════════════
def compute_ks_statistic(y_true, y_prob):
    """Calcule la statistique KS et retourne le point de séparation optimal."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    ks_max = ks_values.max()
    ks_idx = ks_values.argmax()
    ks_threshold = thresholds[ks_idx] if ks_idx < len(thresholds) else 0.5

    return {
        "KS": round(ks_max, 4),
        "Threshold": round(ks_threshold, 4),
        "fpr": fpr,
        "tpr": tpr,
        "ks_values": ks_values,
        "ks_idx": ks_idx,
    }


def plot_ks_statistic(all_predictions, y_test):
    """Trace la courbe KS pour le meilleur modèle."""
    # Trouver le meilleur modèle par AUC
    best_name = None
    best_auc = 0
    for name, preds in all_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds["y_prob"])
        auc_val = auc(fpr, tpr)
        if auc_val > best_auc:
            best_auc = auc_val
            best_name = name

    preds = all_predictions[best_name]
    ks_result = compute_ks_statistic(y_test, preds["y_prob"])

    fig, ax = plt.subplots(figsize=(9, 6))

    # CDF des deux classes
    prob_0 = preds["y_prob"][y_test == 0]
    prob_1 = preds["y_prob"][y_test == 1]

    bins = np.linspace(0, 1, 50)
    ax.hist(prob_0, bins=bins, density=True, alpha=0.5, color=PALETTE[0],
            label="Non-défaut", edgecolor="white")
    ax.hist(prob_1, bins=bins, density=True, alpha=0.5, color=PALETTE[3],
            label="Défaut", edgecolor="white")

    # Ligne KS
    ax.axvline(ks_result["Threshold"], color="green", linestyle="--", linewidth=2,
               label=f'KS = {ks_result["KS"]:.4f} (seuil = {ks_result["Threshold"]:.3f})')

    ax.set_xlabel("Probabilité prédite", fontsize=12)
    ax.set_ylabel("Densité", fontsize=12)
    ax.set_title(f"Distribution des scores — {best_name}\n"
                 f"Statistique KS = {ks_result['KS']:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_MODELS, "ks_statistic.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info(f"✓ Figure : ks_statistic.png (KS = {ks_result['KS']:.4f})")

    # Sauvegarder KS pour tous les modèles
    ks_all = []
    for name, preds in all_predictions.items():
        ks = compute_ks_statistic(y_test, preds["y_prob"])
        ks_all.append({"Modèle": name, "KS": ks["KS"], "Seuil_optimal": ks["Threshold"]})
    ks_df = pd.DataFrame(ks_all)
    ks_df.to_csv(os.path.join(RESULTS_DIR, "ks_statistics.csv"), index=False)
    log.info(f"KS Statistics :\n{ks_df.to_string(index=False)}")

    return ks_result


# ═══════════════════════════════════════════════════════════════════
# 5.5 — Courbes Lift & Gain
# ═══════════════════════════════════════════════════════════════════
def plot_lift_gain_curves(all_predictions, y_test):
    """Trace les courbes de Lift et Gain cumulatif."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = PALETTE[:len(all_predictions)]

    for (name, preds), color in zip(all_predictions.items(), colors):
        y_prob = preds["y_prob"]

        # Tri par probabilité décroissante
        sorted_idx = np.argsort(-y_prob)
        y_sorted = np.array(y_test)[sorted_idx]

        # Gain cumulatif
        n = len(y_sorted)
        total_events = y_sorted.sum()
        cum_events = np.cumsum(y_sorted)
        pct_population = np.arange(1, n + 1) / n
        gain = cum_events / total_events

        # Courbe de Gain
        axes[0].plot(pct_population, gain, color=color, linewidth=2, label=name)

        # Courbe de Lift
        lift = gain / pct_population
        axes[1].plot(pct_population, lift, color=color, linewidth=2, label=name)

    # Gain — ligne de référence
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Aléatoire")
    axes[0].set_xlabel("% Population", fontsize=11)
    axes[0].set_ylabel("% Défauts capturés (Gain)", fontsize=11)
    axes[0].set_title("Courbe de Gain cumulatif", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(alpha=0.3)

    # Lift — ligne de référence
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Aléatoire")
    axes[1].set_xlabel("% Population", fontsize=11)
    axes[1].set_ylabel("Lift", fontsize=11)
    axes[1].set_title("Courbe de Lift", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_MODELS, "lift_gain_curves.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : lift_gain_curves.png")


# ═══════════════════════════════════════════════════════════════════
# 5.6 — Comparaison visuelle des modèles
# ═══════════════════════════════════════════════════════════════════
def plot_model_comparison(metrics_df):
    """Barplot comparatif des métriques par modèle."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    df_plot = metrics_df.set_index("Modèle")[metric_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_plot.index))
    width = 0.15

    for i, col in enumerate(metric_cols):
        bars = ax.bar(x + i * width, df_plot[col], width, label=col, color=PALETTE[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7, rotation=45)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_plot.index, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparaison des performances des modèles", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim([0, 1.15])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_MODELS, "model_comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : model_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# Pipeline principal d'évaluation
# ═══════════════════════════════════════════════════════════════════
def run(all_predictions, all_metrics, y_test):
    """Exécute l'évaluation complète."""
    log.info("=" * 60)
    log.info("ÉTAPE 5 — ÉVALUATION DES MODÈLES")
    log.info("=" * 60)

    metrics_df = pd.DataFrame(all_metrics).sort_values("AUC-ROC", ascending=False)

    # 5.1 Matrices de confusion
    log.info("5.1 Matrices de confusion...")
    plot_confusion_matrices(all_predictions, y_test)

    # 5.2 Courbes ROC
    log.info("5.2 Courbes ROC...")
    plot_roc_curves(all_predictions, y_test)

    # 5.3 Precision-Recall
    log.info("5.3 Courbes Precision-Recall...")
    plot_precision_recall_curves(all_predictions, y_test)

    # 5.4 KS Statistic
    log.info("5.4 Statistique KS...")
    ks_result = plot_ks_statistic(all_predictions, y_test)

    # 5.5 Lift / Gain
    log.info("5.5 Courbes Lift & Gain...")
    plot_lift_gain_curves(all_predictions, y_test)

    # 5.6 Comparaison visuelle
    log.info("5.6 Comparaison des modèles...")
    plot_model_comparison(metrics_df)

    log.info("ÉTAPE 5 TERMINÉE ✓")
    return metrics_df, ks_result


if __name__ == "__main__":
    print("Ce module est appelé depuis main.py")
