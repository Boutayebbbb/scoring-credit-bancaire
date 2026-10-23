"""
Utilitaires de chargement et sauvegarde des données.
"""
import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    """Charge un fichier CSV et affiche les infos de base."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path)
    print(f"[DATA] Chargé : {path}")
    print(f"       → {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df

def save_csv(df: pd.DataFrame, path: str):
    """Sauvegarde un DataFrame en CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[DATA] Sauvegardé : {path} ({df.shape[0]} lignes)")
