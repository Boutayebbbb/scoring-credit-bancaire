"""
Paramètres du pipeline de scoring crédit.
"""

# ── Variable cible ────────────────────────────────────────────────
TARGET = "defaut_paiement"

# ── Colonnes à exclure (data leakage ou identifiant) ─────────────
DROP_COLUMNS = ["client_id", "credit_accorde"]

# ── Seed de reproductibilité ─────────────────────────────────────
RANDOM_STATE = 42

# ── Split train/test ─────────────────────────────────────────────
TEST_SIZE = 0.2

# ── Validation croisée stratifiée ─────────────────────────────────
CV_FOLDS = 5

# ── Gestion du déséquilibre ───────────────────────────────────────
SMOTE_SAMPLING_STRATEGY = 0.5   # ratio minorité/majorité après SMOTE
SMOTE_K_NEIGHBORS = 5

# ── Modèles à entraîner ──────────────────────────────────────────
MODELS = {
    "Régression Logistique": {
        "type": "logistic_regression",
        "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    },
    "Arbre de Décision": {
        "type": "decision_tree",
        "params": {"max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5}
    },
    "Random Forest": {
        "type": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5}
    },
    "SVM": {
        "type": "svm",
        "params": {"C": 1.0, "kernel": "rbf", "probability": True}
    },
}

# ── Grilles de recherche d'hyperparamètres ────────────────────────
HYPERPARAMETER_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "decision_tree": {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [3, 5, 10],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_leaf": [3, 5, 10],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
    },
}

# ── Scorecard ─────────────────────────────────────────────────────
PDO = 20          # Points to Double the Odds
BASE_SCORE = 600  # Score de base
BASE_ODDS = 50    # Odds de référence (1:50)

# ── Seuils de décision ───────────────────────────────────────────
RISK_THRESHOLDS = {
    "faible":   (650, 999),
    "modéré":   (550, 650),
    "élevé":    (0,   550),
}

# ── Visualisations ────────────────────────────────────────────────
FIGSIZE = (10, 6)
DPI = 150
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
