"""
Module 04 — Modélisation : factory pattern, entraînement unifié,
gestion du déséquilibre (SMOTE), et tuning des hyperparamètres.
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import FEATURED_DATA, MODELS_DIR, RESULTS_DIR, FIG_MODELS
from config.params import (
    TARGET, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    MODELS, HYPERPARAMETER_GRIDS,
    SMOTE_SAMPLING_STRATEGY, SMOTE_K_NEIGHBORS,
    DPI, PALETTE
)
from utils.data_loader import load_csv
from utils.logger import log


# ═══════════════════════════════════════════════════════════════════
# 4.1 — Model Factory
# ═══════════════════════════════════════════════════════════════════
def create_model(model_type: str, params: dict):
    """Crée un modèle à partir de son type et ses paramètres."""
    factories = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
    }
    if model_type not in factories:
        raise ValueError(f"Modèle inconnu : {model_type}")

    model_params = {**params, "random_state": RANDOM_STATE}
    return factories[model_type](**model_params)


# ═══════════════════════════════════════════════════════════════════
# 4.2 — Préparation des données
# ═══════════════════════════════════════════════════════════════════
def prepare_data(df):
    """Sépare features/target, applique le scaling et SMOTE."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"  Split : {len(X_train)} train / {len(X_test)} test")
    log.info(f"  Distribution train : {dict(y_train.value_counts())}")
    log.info(f"  Distribution test  : {dict(y_test.value_counts())}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # SMOTE sur le train uniquement
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    log.info(f"  SMOTE : {len(X_train_scaled)} → {len(X_train_resampled)} échantillons")
    log.info(f"  Distribution après SMOTE : {dict(pd.Series(y_train_resampled).value_counts())}")

    # Sauvegarder le scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler


# ═══════════════════════════════════════════════════════════════════
# 4.3 — Entraînement et validation croisée
# ═══════════════════════════════════════════════════════════════════
def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    """Entraîne un modèle et retourne les métriques."""
    # Validation croisée stratifiée
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
    log.info(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraînement final
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilités (si disponible)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)

    # Métriques
    metrics = {
        "Modèle": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC": round(roc_auc_score(y_test, y_prob), 4),
        "CV_AUC_mean": round(cv_scores.mean(), 4),
        "CV_AUC_std": round(cv_scores.std(), 4),
    }

    return model, metrics, y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════
# 4.4 — Hyperparameter tuning
# ═══════════════════════════════════════════════════════════════════
def tune_hyperparameters(model_type, X_train, y_train):
    """GridSearchCV pour trouver les meilleurs hyperparamètres."""
    if model_type not in HYPERPARAMETER_GRIDS:
        return None, {}

    model = create_model(model_type, {})
    param_grid = HYPERPARAMETER_GRIDS[model_type]

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        model, param_grid, cv=skf, scoring="roc_auc",
        n_jobs=-1, verbose=0, refit=True
    )
    grid_search.fit(X_train, y_train)

    log.info(f"    Meilleurs paramètres : {grid_search.best_params_}")
    log.info(f"    Meilleur AUC-CV      : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


# ═══════════════════════════════════════════════════════════════════
# 4.5 — Pipeline principal
# ═══════════════════════════════════════════════════════════════════
def run():
    """Exécute l'entraînement de tous les modèles."""
    log.info("=" * 60)
    log.info("ÉTAPE 4 — MODÉLISATION")
    log.info("=" * 60)

    df = load_csv(FEATURED_DATA)

    # Préparation
    log.info("4.1 Préparation des données (scaling + SMOTE)...")
    X_train, X_test, y_train, y_test = prepare_data(df)[:4]

    # Entraînement de chaque modèle
    all_metrics = []
    all_models = {}
    all_predictions = {}

    for name, config in MODELS.items():
        log.info(f"\n{'─' * 40}")
        log.info(f"Modèle : {name}")
        log.info(f"{'─' * 40}")

        # Tuning
        log.info(f"  Tuning des hyperparamètres...")
        best_model, best_params = tune_hyperparameters(
            config["type"], X_train, y_train
        )

        if best_model is None:
            model = create_model(config["type"], config["params"])
        else:
            model = best_model

        # Entraînement + évaluation
        model, metrics, y_pred, y_prob = train_and_evaluate(
            name, model, X_train, X_test, y_train, y_test
        )

        all_metrics.append(metrics)
        all_models[name] = model
        all_predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}

        # Sauvegarder le modèle
        model_filename = config["type"] + ".joblib"
        joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
        log.info(f"  ✓ Modèle sauvegardé : {model_filename}")

    # ── Tableau comparatif ────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("AUC-ROC", ascending=False)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
    log.info(f"\n{'═' * 60}")
    log.info(f"TABLEAU COMPARATIF DES MODÈLES :")
    log.info(f"{'═' * 60}")
    log.info(f"\n{metrics_df.to_string(index=False)}")

    # Identifier et sauvegarder le meilleur modèle
    best_name = metrics_df.iloc[0]["Modèle"]
    best_model = all_models[best_name]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    log.info(f"\n★ Meilleur modèle : {best_name} (AUC = {metrics_df.iloc[0]['AUC-ROC']})")

    # Sauvegarder les prédictions du meilleur modèle
    best_preds = all_predictions[best_name]
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": best_preds["y_pred"],
        "y_prob": best_preds["y_prob"]
    })
    pred_df.to_csv(os.path.join(RESULTS_DIR, "best_model_predictions.csv"), index=False)

    log.info("ÉTAPE 4 TERMINÉE ✓")
    return all_models, all_metrics, all_predictions, X_test, y_test


if __name__ == "__main__":
    run()
