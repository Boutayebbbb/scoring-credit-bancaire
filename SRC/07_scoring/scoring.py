"""
Module 07 — Scoring : construction de la grille de scoring (PDO),
segmentation des clients par niveau de risque,
et proposition de politique de décision.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os, sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import MODELS_DIR, RESULTS_DIR, FIG_SCORING
from config.params import (
    PDO, BASE_SCORE, BASE_ODDS, RISK_THRESHOLDS, DPI, PALETTE
)
from utils.logger import log


# ═══════════════════════════════════════════════════════════════════
# 7.1 — Construction de la Scorecard (PDO method)
# ═══════════════════════════════════════════════════════════════════
def build_scorecard(y_prob):
    """
    Convertit les probabilités en scores bancaires via la méthode PDO.

    Formule :
        Score = Offset + Factor × ln(odds)
        où Factor = PDO / ln(2)
        et Offset = Base_Score - Factor × ln(Base_Odds)
    """
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)

    # Éviter prob = 0 ou 1
    prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    odds = (1 - prob_clipped) / prob_clipped
    scores = offset + factor * np.log(odds)

    log.info(f"  Scorecard PDO : Factor={factor:.2f}, Offset={offset:.2f}")
    log.info(f"  Scores : min={scores.min():.0f}, max={scores.max():.0f}, "
             f"mean={scores.mean():.0f}, median={np.median(scores):.0f}")

    return scores, factor, offset


# ═══════════════════════════════════════════════════════════════════
# 7.2 — Segmentation par niveau de risque
# ═══════════════════════════════════════════════════════════════════
def segment_risk(scores):
    """Segmente les clients en niveaux de risque basés sur leur score."""
    risk_labels = []
    for s in scores:
        assigned = False
        for level, (low, high) in RISK_THRESHOLDS.items():
            if low <= s <= high:
                risk_labels.append(level)
                assigned = True
                break
        if not assigned:
            risk_labels.append("élevé")

    return np.array(risk_labels)


# ═══════════════════════════════════════════════════════════════════
# 7.3 — Visualisations du scoring
# ═══════════════════════════════════════════════════════════════════
def plot_score_distribution(scores, y_true, risk_segments):
    """Visualise la distribution des scores par classe et par segment."""

    # ── Distribution des scores par classe réelle ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    scores_0 = scores[y_true == 0]
    scores_1 = scores[y_true == 1]

    bins = np.linspace(scores.min() - 10, scores.max() + 10, 40)

    axes[0].hist(scores_0, bins=bins, alpha=0.65, color=PALETTE[0],
                 label=f"Non-défaut (n={len(scores_0)})", edgecolor="white")
    axes[0].hist(scores_1, bins=bins, alpha=0.65, color=PALETTE[3],
                 label=f"Défaut (n={len(scores_1)})", edgecolor="white")
    axes[0].set_xlabel("Score de crédit", fontsize=11)
    axes[0].set_ylabel("Nombre de clients", fontsize=11)
    axes[0].set_title("Distribution des scores par classe", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].spines[["top", "right"]].set_visible(False)

    # ── Répartition par segment de risque ─────────────────────────
    segments, counts = np.unique(risk_segments, return_counts=True)
    seg_order = ["faible", "modéré", "élevé"]
    ordered_counts = [counts[list(segments).index(s)] if s in segments else 0 for s in seg_order]
    seg_colors = [PALETTE[0], PALETTE[2], PALETTE[3]]

    bars = axes[1].bar(seg_order, ordered_counts, color=seg_colors, edgecolor="white", width=0.6)
    for bar, count in zip(bars, ordered_counts):
        pct = count / len(scores) * 100
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{count}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Niveau de risque", fontsize=11)
    axes[1].set_ylabel("Nombre de clients", fontsize=11)
    axes[1].set_title("Segmentation par niveau de risque", fontsize=13, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SCORING, "score_distribution.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : score_distribution.png")

    # ── Taux de défaut par segment ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    segment_df = pd.DataFrame({"Score": scores, "Risque": risk_segments, "Defaut": y_true})
    default_rate = segment_df.groupby("Risque")["Defaut"].mean()
    default_rate = default_rate.reindex(seg_order)

    bars = ax.bar(default_rate.index, default_rate.values * 100, color=seg_colors,
                  edgecolor="white", width=0.6)
    for bar, val in zip(bars, default_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Niveau de risque", fontsize=11)
    ax.set_ylabel("Taux de défaut (%)", fontsize=11)
    ax.set_title("Taux de défaut par segment de risque", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SCORING, "default_rate_by_segment.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : default_rate_by_segment.png")


# ═══════════════════════════════════════════════════════════════════
# 7.4 — Politique de décision
# ═══════════════════════════════════════════════════════════════════
def generate_decision_policy(scores, y_true, risk_segments):
    """Génère une politique de décision recommandée."""
    segment_df = pd.DataFrame({
        "Score": scores,
        "Risque": risk_segments,
        "Defaut": y_true
    })

    policy = []
    for seg in ["faible", "modéré", "élevé"]:
        subset = segment_df[segment_df["Risque"] == seg]
        if len(subset) == 0:
            continue
        default_rate = subset["Defaut"].mean()
        n_clients = len(subset)

        if seg == "faible":
            decision = "APPROUVER — Crédit accordé sans conditions"
            taux_max = "Taux standard"
        elif seg == "modéré":
            decision = "APPROUVER AVEC CONDITIONS — Garanties supplémentaires requises"
            taux_max = "Taux majoré (+1-2%)"
        else:
            decision = "REFUSER ou ÉTUDE APPROFONDIE — Risque trop élevé"
            taux_max = "Non applicable"

        policy.append({
            "Segment": seg.capitalize(),
            "Score_range": f"{RISK_THRESHOLDS[seg][0]}–{RISK_THRESHOLDS[seg][1]}",
            "N_clients": n_clients,
            "Taux_defaut": f"{default_rate*100:.1f}%",
            "Décision": decision,
            "Taux_intérêt": taux_max,
        })

    policy_df = pd.DataFrame(policy)
    policy_df.to_csv(os.path.join(RESULTS_DIR, "decision_policy.csv"), index=False)
    log.info(f"\nPolitique de décision :\n{policy_df.to_string(index=False)}")

    return policy_df


# ═══════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════
def run(y_prob, y_test):
    """Exécute le scoring complet."""
    log.info("=" * 60)
    log.info("ÉTAPE 7 — SCORING & POLITIQUE DE DÉCISION")
    log.info("=" * 60)

    y_true = np.array(y_test)

    # 7.1 Construction de la scorecard
    log.info("7.1 Construction de la scorecard (PDO)...")
    scores, factor, offset = build_scorecard(y_prob)

    # 7.2 Segmentation
    log.info("7.2 Segmentation par risque...")
    risk_segments = segment_risk(scores)

    # Sauvegarder scorecard
    scorecard_df = pd.DataFrame({
        "Score": scores.round(0).astype(int),
        "Probabilite_defaut": y_prob.round(4),
        "Risque": risk_segments,
        "Defaut_reel": y_true,
    })
    scorecard_df.to_csv(os.path.join(RESULTS_DIR, "scorecard_table.csv"), index=False)

    # 7.3 Visualisations
    log.info("7.3 Visualisations du scoring...")
    plot_score_distribution(scores, y_true, risk_segments)

    # 7.4 Politique de décision
    log.info("7.4 Politique de décision...")
    policy_df = generate_decision_policy(scores, y_true, risk_segments)

    # Résumé
    log.info(f"\n{'═' * 50}")
    log.info(f"RÉSUMÉ SCORING :")
    for seg in ["faible", "modéré", "élevé"]:
        n = (risk_segments == seg).sum()
        pct = n / len(risk_segments) * 100
        log.info(f"  {seg.capitalize():>8s} : {n:>4d} clients ({pct:.1f}%)")
    log.info(f"{'═' * 50}")

    log.info("ÉTAPE 7 TERMINÉE ✓")
    return scores, risk_segments, policy_df


if __name__ == "__main__":
    print("Ce module est appelé depuis main.py")
