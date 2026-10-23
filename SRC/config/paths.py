"""
Configuration centralisée — chemins et paramètres du projet.
Projet : Scoring Crédit Bancaire
Auteurs : BOUTAYEB Marouane, Ahmed DIOURI
Encadrant : Pr. Mohamed EL FAR
"""
import os

# ── Racine du projet ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Données ───────────────────────────────────────────────────────
DATA_DIR       = os.path.join(PROJECT_ROOT, "DATA")
RAW_DATA       = os.path.join(DATA_DIR, "raw", "12_banque_scoring_credit.csv")
CLEANED_DATA   = os.path.join(DATA_DIR, "processed", "cleaned.csv")
FEATURED_DATA  = os.path.join(DATA_DIR, "processed", "featured.csv")
WOE_DIR        = os.path.join(DATA_DIR, "woe")

# ── Figures ───────────────────────────────────────────────────────
FIG_DIR             = os.path.join(PROJECT_ROOT, "FIG")
FIG_EXPLORATION     = os.path.join(FIG_DIR, "exploration")
FIG_PREPROCESSING   = os.path.join(FIG_DIR, "preprocessing")
FIG_MODELS          = os.path.join(FIG_DIR, "models")
FIG_INTERPRETABILITY = os.path.join(FIG_DIR, "interpretability")
FIG_SCORING         = os.path.join(FIG_DIR, "scoring")

# ── Modèles sauvegardés ──────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "MODELS")

# ── Résultats ─────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_ROOT, "RESULTS")

# ── Rapport & Présentation ────────────────────────────────────────
REP_DIR  = os.path.join(PROJECT_ROOT, "REP")
PPTX_DIR = os.path.join(PROJECT_ROOT, "PPTX")
