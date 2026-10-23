"""
═══════════════════════════════════════════════════════════════════
    SCORING CRÉDIT BANCAIRE — Pipeline principal
    Projet Data Mining — ESISA 2025/2026
    Auteurs : BOUTAYEB Marouane, Ahmed DIOURI
    Encadrant : Pr. Mohamed EL FAR
═══════════════════════════════════════════════════════════════════

Usage :
    python main.py          → Exécute le pipeline complet
"""
import sys
import os
import time
import importlib
import warnings
warnings.filterwarnings("ignore")

# Ajouter SRC au path pour permettre les imports
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SRC")
sys.path.insert(0, SRC_DIR)

from config.paths import *
from utils.logger import log


def ensure_directories():
    """Crée tous les répertoires nécessaires."""
    dirs = [
        os.path.join(DATA_DIR, "raw"),
        os.path.join(DATA_DIR, "processed"),
        os.path.join(DATA_DIR, "woe"),
        FIG_EXPLORATION, FIG_PREPROCESSING, FIG_MODELS,
        FIG_INTERPRETABILITY, FIG_SCORING,
        MODELS_DIR, RESULTS_DIR, REP_DIR, PPTX_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def import_module(folder, module_name):
    """Importe un module depuis les sous-dossiers numérotés de SRC."""
    return importlib.import_module(f"{folder}.{module_name}")


def run_pipeline():
    """Exécute le pipeline complet de A à Z."""
    start_time = time.time()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   SCORING CRÉDIT BANCAIRE — Pipeline Data Mining        ║")
    log.info("║   ESISA — 4ème année ISIIA — 2025/2026                 ║")
    log.info("║   BOUTAYEB Marouane & Ahmed DIOURI                     ║")
    log.info("║   Encadrant : Pr. Mohamed EL FAR                       ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    ensure_directories()

    # ── ÉTAPE 1 : Exploration ─────────────────────────────────────
    mod = import_module("01_exploration", "stats_descriptives")
    mod.run()

    # ── ÉTAPE 2 : Prétraitement ───────────────────────────────────
    mod = import_module("02_preprocessing", "cleaning")
    mod.run()

    # ── ÉTAPE 3 : Feature Engineering ─────────────────────────────
    mod = import_module("03_feature_engineering", "feature_engineering")
    mod.run()

    # ── ÉTAPE 4 : Modélisation ────────────────────────────────────
    mod = import_module("04_modelisation", "train")
    all_models, all_metrics, all_predictions, X_test, y_test = mod.run()

    # ── ÉTAPE 5 : Évaluation ─────────────────────────────────────
    mod = import_module("05_evaluation", "evaluation")
    mod.run(all_predictions, all_metrics, y_test)

    # ── ÉTAPE 6 : Interprétabilité ────────────────────────────────
    mod = import_module("06_interpretability", "interpretability")
    mod.run(all_models, X_test, y_test)

    # ── ÉTAPE 7 : Scoring & Politique de décision ─────────────────
    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics).sort_values("AUC-ROC", ascending=False)
    best_name = metrics_df.iloc[0]["Modèle"]
    y_prob = all_predictions[best_name]["y_prob"]

    mod = import_module("07_scoring", "scoring")
    mod.run(y_prob, y_test)

    # ── Résumé final ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info(f"║   PIPELINE TERMINÉ EN {elapsed:.1f}s                             ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"  → Figures    : {FIG_DIR}")
    log.info(f"  → Modèles    : {MODELS_DIR}")
    log.info(f"  → Résultats  : {RESULTS_DIR}")


if __name__ == "__main__":
    run_pipeline()
