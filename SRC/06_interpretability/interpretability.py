"""
Module 06 — Interprétabilité : SHAP values, importance des variables,
et Partial Dependence Plots.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import os, sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.inspection import PartialDependenceDisplay

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import MODELS_DIR, FIG_INTERPRETABILITY, RESULTS_DIR
from config.params import DPI, PALETTE
from utils.logger import log


# ═══════════════════════════════════════════════════════════════════
# 6.1 — Feature Importance (modèles tree-based)
# ═══════════════════════════════════════════════════════════════════
def plot_feature_importance(models, feature_names):
    """Affiche l'importance des features pour les modèles compatibles."""
    importance_data = {}

    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importance_data[name] = pd.Series(imp, index=feature_names).sort_values(ascending=True)
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
            importance_data[name] = pd.Series(imp, index=feature_names).sort_values(ascending=True)

    if not importance_data:
        log.info("  Aucun modèle compatible pour feature importance")
        return

    n = len(importance_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for i, (name, imp) in enumerate(importance_data.items()):
        colors = [PALETTE[3] if v > imp.quantile(0.75) else PALETTE[0] for v in imp.values]
        axes[i].barh(imp.index, imp.values, color=colors, edgecolor="white")
        axes[i].set_title(f"Importance — {name}", fontsize=12, fontweight="bold")
        axes[i].spines[["top", "right"]].set_visible(False)
        for j, (idx, val) in enumerate(imp.items()):
            axes[i].text(val + imp.max() * 0.01, j, f"{val:.4f}", va="center", fontsize=8)

    plt.suptitle("Importance des variables par modèle", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_INTERPRETABILITY, "feature_importance.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : feature_importance.png")

    # Sauvegarder
    for name, imp in importance_data.items():
        imp.to_csv(os.path.join(RESULTS_DIR, f"feature_importance_{name.replace(' ', '_')}.csv"))


# ═══════════════════════════════════════════════════════════════════
# 6.2 — SHAP Values
# ═══════════════════════════════════════════════════════════════════
def compute_shap_values(best_model, X_test, model_name="best_model"):
    """Calcule et visualise les SHAP values."""
    log.info("  Calcul des SHAP values (peut prendre quelques secondes)...")

    # Choisir l'explainer adapté
    if hasattr(best_model, "predict_proba"):
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Classe positive
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # Classe positive (3D array)
        except Exception:
            # Fallback pour les modèles non-tree
            explainer = shap.KernelExplainer(best_model.predict_proba, X_test.iloc[:100])
            shap_values = explainer.shap_values(X_test.iloc[:200])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            X_test = X_test.iloc[:200]
    else:
        explainer = shap.KernelExplainer(best_model.predict, X_test.iloc[:100])
        shap_values = explainer.shap_values(X_test.iloc[:200])
        X_test = X_test.iloc[:200]

    # SHAP Summary Plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("SHAP Summary Plot — Impact des variables", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_INTERPRETABILITY, "shap_summary.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : shap_summary.png")

    # SHAP Bar Plot (importance moyenne)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP — Importance moyenne absolue", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_INTERPRETABILITY, "shap_bar.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("✓ Figure : shap_bar.png")

    # Sauvegarder SHAP values
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_mean = shap_df.abs().mean().sort_values(ascending=False)
    shap_mean.to_csv(os.path.join(RESULTS_DIR, "shap_importance.csv"))

    return shap_values


# ═══════════════════════════════════════════════════════════════════
# 6.3 — Partial Dependence Plots
# ═══════════════════════════════════════════════════════════════════
def plot_partial_dependence(best_model, X_test, top_features):
    """Trace les PDP pour les variables les plus importantes."""
    try:
        n_features = min(4, len(top_features))
        features_to_plot = top_features[:n_features]

        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4.5))
        if n_features == 1:
            axes = [axes]

        display = PartialDependenceDisplay.from_estimator(
            best_model, X_test, features_to_plot,
            kind="average", ax=axes
        )
        fig.suptitle("Partial Dependence Plots — Variables clés", fontsize=14, fontweight="bold", y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_INTERPRETABILITY, "partial_dependence.png"), dpi=DPI, bbox_inches="tight")
        plt.close()
        log.info("✓ Figure : partial_dependence.png")
    except Exception as e:
        log.warning(f"  PDP non disponible pour ce modèle : {e}")


# ═══════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════
def run(all_models, X_test, y_test):
    """Exécute l'analyse d'interprétabilité complète."""
    log.info("=" * 60)
    log.info("ÉTAPE 6 — INTERPRÉTABILITÉ")
    log.info("=" * 60)

    feature_names = X_test.columns.tolist()

    # 6.1 Feature Importance
    log.info("6.1 Feature Importance...")
    plot_feature_importance(all_models, feature_names)

    # 6.2 SHAP (sur le meilleur modèle)
    log.info("6.2 SHAP Values...")
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    shap_values = compute_shap_values(best_model, X_test)

    # 6.3 PDP
    log.info("6.3 Partial Dependence Plots...")
    # Identifier les top features
    # Le fichier a une ligne d'en-tête : on la lit comme en-tête (et non comme une variable)
    shap_imp = pd.read_csv(os.path.join(RESULTS_DIR, "shap_importance.csv"), index_col=0)
    top_features = shap_imp.index[:4].tolist()
    # Essayer PDP avec un modèle tree-based
    for name, model in all_models.items():
        if hasattr(model, "feature_importances_"):
            plot_partial_dependence(model, X_test, top_features)
            break

    log.info("ÉTAPE 6 TERMINÉE ✓")


if __name__ == "__main__":
    print("Ce module est appelé depuis main.py")
