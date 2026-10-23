"""
Module 02 — Prétraitement : nettoyage des données.
Suppression des colonnes inutiles, gestion des valeurs aberrantes,
détection et traitement des outliers.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import RAW_DATA, CLEANED_DATA, FIG_PREPROCESSING, RESULTS_DIR
from config.params import TARGET, DROP_COLUMNS, FIGSIZE, DPI, PALETTE
from utils.data_loader import load_csv, save_csv
from utils.logger import log


def detect_outliers_iqr(df, column, factor=1.5):
    """Détecte les outliers avec la méthode IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper


def run():
    """Exécute le nettoyage complet des données."""
    log.info("=" * 60)
    log.info("ÉTAPE 2 — PRÉTRAITEMENT DES DONNÉES")
    log.info("=" * 60)

    df = load_csv(RAW_DATA)

    # ── 2.1 Suppression des colonnes à risque de leakage ─────────
    log.info(f"Suppression des colonnes : {DROP_COLUMNS}")
    df_clean = df.drop(columns=DROP_COLUMNS, errors="ignore")
    log.info(f"Colonnes restantes : {list(df_clean.columns)}")

    # ── 2.2 Vérification des valeurs manquantes ──────────────────
    missing = df_clean.isnull().sum()
    log.info(f"Valeurs manquantes :\n{missing}")

    # ── 2.3 Détection des outliers (IQR) ─────────────────────────
    features = [c for c in df_clean.columns if c != TARGET]
    outlier_report = []

    for col in features:
        outliers, lower, upper = detect_outliers_iqr(df_clean, col)
        n_outliers = len(outliers)
        pct = n_outliers / len(df_clean) * 100
        outlier_report.append({
            "Variable": col,
            "N_outliers": n_outliers,
            "Pct_outliers": round(pct, 2),
            "Borne_inf": round(lower, 2),
            "Borne_sup": round(upper, 2),
            "Min": df_clean[col].min(),
            "Max": df_clean[col].max(),
        })
        if n_outliers > 0:
            log.info(f"  {col}: {n_outliers} outliers ({pct:.1f}%) — bornes [{lower:.1f}, {upper:.1f}]")

    outlier_df = pd.DataFrame(outlier_report)
    outlier_df.to_csv(os.path.join(RESULTS_DIR, "outlier_report.csv"), index=False)
    log.info("✓ Rapport outliers sauvegardé")

    # ── 2.4 Traitement des outliers (capping/winsorization) ──────
    df_capped = df_clean.copy()
    for col in features:
        _, lower, upper = detect_outliers_iqr(df_clean, col)
        before = df_capped[col].describe()
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        after = df_capped[col].describe()
        n_clipped = (df_clean[col] != df_capped[col]).sum()
        if n_clipped > 0:
            log.info(f"  {col}: {n_clipped} valeurs cappées")

    # ── 2.5 Visualisation avant/après capping ─────────────────────
    fig, axes = plt.subplots(2, len(features), figsize=(22, 8))
    if len(features) == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(features):
        # Avant
        axes[0][i].boxplot(df_clean[col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor=PALETTE[0], alpha=0.7))
        axes[0][i].set_title(f"{col}\n(avant)", fontsize=9, fontweight="bold")
        axes[0][i].spines[["top", "right"]].set_visible(False)

        # Après
        axes[1][i].boxplot(df_capped[col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor=PALETTE[2], alpha=0.7))
        axes[1][i].set_title(f"{col}\n(après)", fontsize=9, fontweight="bold")
        axes[1][i].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Traitement des outliers (Winsorization IQR)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PREPROCESSING, "outliers_avant_apres.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : outliers_avant_apres.png")

    # ── 2.6 Vérification des doublons ─────────────────────────────
    n_dup = df_capped.duplicated().sum()
    log.info(f"Doublons détectés : {n_dup}")
    if n_dup > 0:
        df_capped = df_capped.drop_duplicates()
        log.info(f"  → {n_dup} doublons supprimés")

    # ── 2.7 Sauvegarde ────────────────────────────────────────────
    save_csv(df_capped, CLEANED_DATA)
    log.info("ÉTAPE 2 TERMINÉE ✓")
    return df_capped


if __name__ == "__main__":
    run()
