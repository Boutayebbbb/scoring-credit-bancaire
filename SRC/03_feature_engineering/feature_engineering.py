"""
Module 03 — Feature Engineering : création de ratios financiers,
binning intelligent, et calcul WoE/IV.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import CLEANED_DATA, FEATURED_DATA, WOE_DIR, FIG_PREPROCESSING, RESULTS_DIR
from config.params import TARGET, FIGSIZE, DPI, PALETTE
from utils.data_loader import load_csv, save_csv
from utils.logger import log


# ═══════════════════════════════════════════════════════════════════
# 3.1 — Création de ratios financiers
# ═══════════════════════════════════════════════════════════════════
def create_ratios(df):
    """Crée des ratios financiers pertinents pour le scoring."""
    df = df.copy()

    # Ratio dette / revenu mensuel
    df["ratio_dette_revenu"] = np.where(
        df["revenu_mensuel"] > 0,
        df["dettes_totales"] / (df["revenu_mensuel"] * 12),
        0
    )

    # Ratio montant crédit / revenu annuel
    df["ratio_credit_revenu"] = np.where(
        df["revenu_mensuel"] > 0,
        df["montant_credit"] / (df["revenu_mensuel"] * 12),
        0
    )

    # Ratio montant crédit / ancienneté emploi
    df["ratio_credit_anciennete"] = np.where(
        df["anciennete_emploi"] > 0,
        df["montant_credit"] / (df["anciennete_emploi"] + 1),
        df["montant_credit"]
    )

    # Score de stabilité (ancienneté normalisée × revenu)
    df["score_stabilite"] = (
        df["anciennete_emploi"] / (df["anciennete_emploi"].max() + 1)
    ) * (
        df["revenu_mensuel"] / (df["revenu_mensuel"].max() + 1)
    )

    # Charge financière totale
    df["charge_financiere"] = df["dettes_totales"] + df["montant_credit"]

    # Indicateur incidents élevés
    df["incidents_eleves"] = (df["incidents_paiement"] >= 4).astype(int)

    log.info(f"  → {6} nouvelles variables créées")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3.2 — Binning des variables continues
# ═══════════════════════════════════════════════════════════════════
def bin_variables(df):
    """Applique un binning optimal sur les variables clés."""
    df = df.copy()

    # Binning de l'âge
    df["age_bin"] = pd.cut(df["age"],
                           bins=[0, 30, 40, 50, 60, 100],
                           labels=["<30", "30-40", "40-50", "50-60", ">60"])

    # Binning du revenu
    df["revenu_bin"] = pd.cut(df["revenu_mensuel"],
                              bins=[0, 10000, 20000, 30000, 50000],
                              labels=["Faible", "Moyen", "Bon", "Élevé"])

    # Binning du ratio d'endettement
    df["endettement_bin"] = pd.cut(df["ratio_endettement"],
                                   bins=[-0.01, 0.33, 0.50, 1.0, 10],
                                   labels=["Sain", "Modéré", "Risqué", "Critique"])

    # Binning ancienneté
    df["anciennete_bin"] = pd.cut(df["anciennete_emploi"],
                                  bins=[-1, 5, 15, 25, 100],
                                  labels=["Nouveau", "Intermédiaire", "Confirmé", "Senior"])

    log.info("  → Variables binnées : age, revenu, endettement, ancienneté")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3.3 — Weight of Evidence (WoE) et Information Value (IV)
# ═══════════════════════════════════════════════════════════════════
def calculate_woe_iv(df, feature, target):
    """Calcule le WoE et IV pour une variable catégorielle / binnée."""
    crosstab = pd.crosstab(df[feature], df[target])
    if crosstab.shape[1] < 2:
        return pd.DataFrame(), 0

    crosstab.columns = ["Non_defaut", "Defaut"]
    crosstab["Total"] = crosstab.sum(axis=1)

    total_non_defaut = crosstab["Non_defaut"].sum()
    total_defaut = crosstab["Defaut"].sum()

    # Distribution des événements et non-événements
    crosstab["Pct_non_defaut"] = crosstab["Non_defaut"] / total_non_defaut
    crosstab["Pct_defaut"] = crosstab["Defaut"] / total_defaut

    # Éviter log(0)
    epsilon = 1e-6
    crosstab["Pct_non_defaut"] = crosstab["Pct_non_defaut"].clip(lower=epsilon)
    crosstab["Pct_defaut"] = crosstab["Pct_defaut"].clip(lower=epsilon)

    # WoE
    crosstab["WoE"] = np.log(crosstab["Pct_non_defaut"] / crosstab["Pct_defaut"])

    # IV par bin
    crosstab["IV"] = (crosstab["Pct_non_defaut"] - crosstab["Pct_defaut"]) * crosstab["WoE"]

    total_iv = crosstab["IV"].sum()

    crosstab["Variable"] = feature
    crosstab = crosstab.reset_index()

    return crosstab, total_iv


def run():
    """Exécute le feature engineering complet."""
    log.info("=" * 60)
    log.info("ÉTAPE 3 — FEATURE ENGINEERING")
    log.info("=" * 60)

    df = load_csv(CLEANED_DATA)

    # 3.1 Ratios financiers
    log.info("3.1 Création des ratios financiers...")
    df = create_ratios(df)

    # 3.2 Binning
    log.info("3.2 Binning des variables...")
    df = bin_variables(df)

    # 3.3 WoE / IV
    log.info("3.3 Calcul du WoE et de l'Information Value...")
    bin_cols = ["age_bin", "revenu_bin", "endettement_bin", "anciennete_bin"]
    iv_results = []
    all_woe = []

    for col in bin_cols:
        woe_table, iv = calculate_woe_iv(df, col, TARGET)
        iv_results.append({"Variable": col, "IV": round(iv, 4)})
        all_woe.append(woe_table)
        log.info(f"  {col}: IV = {iv:.4f}")

    # Interprétation de l'IV
    iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False)
    iv_df["Pouvoir_predictif"] = iv_df["IV"].apply(
        lambda x: "Très fort" if x > 0.5 else
                  "Fort" if x > 0.3 else
                  "Moyen" if x > 0.1 else
                  "Faible" if x > 0.02 else "Inutile"
    )
    iv_df.to_csv(os.path.join(RESULTS_DIR, "information_value.csv"), index=False)
    log.info(f"\nTableau IV :\n{iv_df.to_string(index=False)}")

    # Sauvegarder les tables WoE
    if all_woe:
        woe_combined = pd.concat(all_woe, ignore_index=True)
        woe_combined.to_csv(os.path.join(WOE_DIR, "woe_tables.csv"), index=False)
        log.info("✓ Tables WoE sauvegardées")

    # ── Visualisation IV ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[3] if v > 0.3 else PALETTE[2] if v > 0.1 else PALETTE[0]
              for v in iv_df["IV"]]
    bars = ax.barh(iv_df["Variable"], iv_df["IV"], color=colors, edgecolor="white")
    ax.set_title("Information Value (IV) par variable binnée", fontsize=13, fontweight="bold")
    ax.set_xlabel("IV")
    ax.axvline(0.1, color="orange", linestyle="--", linewidth=1, label="Seuil moyen (0.1)")
    ax.axvline(0.3, color="red", linestyle="--", linewidth=1, label="Seuil fort (0.3)")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, iv_df["IV"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PREPROCESSING, "information_value.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : information_value.png")

    # ── Sauvegarder le dataset enrichi (sans les bins catégoriels) ─
    cols_to_drop_bins = bin_cols  # on garde les features numériques
    df_featured = df.drop(columns=cols_to_drop_bins, errors="ignore")
    save_csv(df_featured, FEATURED_DATA)

    log.info("ÉTAPE 3 TERMINÉE ✓")
    return df_featured


if __name__ == "__main__":
    run()
