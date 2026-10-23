"""
Module 01 — Exploration : statistiques descriptives.
Analyse exploratoire complète du jeu de données de scoring crédit.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import RAW_DATA, FIG_EXPLORATION, RESULTS_DIR
from config.params import TARGET, DROP_COLUMNS, FIGSIZE, DPI, PALETTE
from utils.data_loader import load_csv
from utils.logger import log


def run():
    """Exécute l'analyse exploratoire complète."""
    log.info("=" * 60)
    log.info("ÉTAPE 1 — EXPLORATION DES DONNÉES")
    log.info("=" * 60)

    df = load_csv(RAW_DATA)

    # ── 1.1 Informations générales ────────────────────────────────
    log.info(f"Dimensions : {df.shape}")
    log.info(f"Colonnes   : {list(df.columns)}")
    log.info(f"Types :\n{df.dtypes}")
    log.info(f"Valeurs manquantes :\n{df.isnull().sum()}")

    # ── 1.2 Statistiques descriptives ─────────────────────────────
    desc = df.describe().T
    desc["CV"] = (desc["std"] / desc["mean"]).round(3)
    desc.to_csv(os.path.join(RESULTS_DIR, "stats_descriptives.csv"))
    log.info(f"Statistiques descriptives :\n{desc}")

    # ── 1.3 Distribution de la cible ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Barplot
    target_counts = df[TARGET].value_counts()
    ax = axes[0]
    bars = ax.bar(["Non-défaut (0)", "Défaut (1)"],
                  target_counts.values,
                  color=[PALETTE[0], PALETTE[3]], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, target_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Distribution de la variable cible", fontsize=13, fontweight="bold")
    ax.set_ylabel("Nombre de clients")
    ax.spines[["top", "right"]].set_visible(False)

    # Pie chart
    axes[1].pie(target_counts.values, labels=["Non-défaut", "Défaut"],
                colors=[PALETTE[0], PALETTE[3]], autopct="%1.1f%%",
                startangle=90, explode=(0, 0.08), shadow=False,
                textprops={"fontsize": 12})
    axes[1].set_title("Proportion défaut / non-défaut", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_EXPLORATION, "distribution_cible.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : distribution_cible.png")

    # ── 1.4 Distributions des variables numériques ────────────────
    features = [c for c in df.columns if c not in DROP_COLUMNS + [TARGET]]
    n_feat = len(features)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        ax.hist(df[col], bins=30, color=PALETTE[i % len(PALETTE)],
                edgecolor="white", alpha=0.85, linewidth=0.8)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.2, label=f"μ={df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="green", linestyle="-.", linewidth=1.2, label=f"med={df[col].median():.1f}")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Masquer les axes inutilisés
    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution des variables explicatives", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_EXPLORATION, "distributions_variables.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : distributions_variables.png")

    # ── 1.5 Boxplots par classe cible ─────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        data_0 = df[df[TARGET] == 0][col]
        data_1 = df[df[TARGET] == 1][col]
        bp = ax.boxplot([data_0, data_1], patch_artist=True, widths=0.6)
        # Étiquettes posées séparément : compatible avec toutes les versions de Matplotlib
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Non-défaut", "Défaut"])
        bp["boxes"][0].set_facecolor(PALETTE[0])
        bp["boxes"][1].set_facecolor(PALETTE[3])
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots par classe de défaut", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_EXPLORATION, "boxplots_par_classe.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : boxplots_par_classe.png")

    # ── 1.6 Matrice de corrélation ────────────────────────────────
    corr_cols = [c for c in df.columns if c not in ["client_id"]]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matrice de corrélation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_EXPLORATION, "matrice_correlation.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : matrice_correlation.png")

    # ── 1.7 Corrélation avec la cible ─────────────────────────────
    corr_target = df[corr_cols].corr()[TARGET].drop(TARGET).sort_values()
    corr_target.to_csv(os.path.join(RESULTS_DIR, "correlation_cible.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[3] if v > 0 else PALETTE[0] for v in corr_target.values]
    ax.barh(corr_target.index, corr_target.values, color=colors, edgecolor="white")
    ax.set_title("Corrélation des variables avec défaut_paiement", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (idx, val) in enumerate(corr_target.items()):
        ax.text(val + 0.02 * np.sign(val), i, f"{val:.3f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_EXPLORATION, "correlation_cible.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : correlation_cible.png")

    log.info("ÉTAPE 1 TERMINÉE ✓")
    return df


if __name__ == "__main__":
    run()
