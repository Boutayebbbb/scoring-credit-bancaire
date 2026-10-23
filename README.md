# Scoring crédit des clients bancaires

Système complet de **scoring crédit** : prédiction du risque de défaut de paiement de 3 000 clients,
interprétation des décisions (SHAP) et construction d'une **grille de score bancaire** (méthode PDO)
avec une politique de décision en 3 segments.

## Résultats

| Modèle | AUC-ROC | F1-score | Recall | KS |
| --- | --- | --- | --- | --- |
| **Random Forest** | **0,9971** | **0,9462** | **1,000** | **0,9928** |
| SVM | 0,9955 | 0,8462 | 1,000 | 0,9766 |
| Régression logistique | 0,9955 | 0,8317 | 0,955 | 0,9532 |
| Arbre de décision | 0,9849 | 0,8571 | 0,955 | 0,9503 |

Évaluation sur un test stratifié de 600 clients ; validation croisée 5-fold (AUC moyenne du Random Forest : 0,9987 ± 0,0007).

**Grille de score (PDO = 20, score de base 600)** : `Score = 487,12 + 28,85 × ln((1 − p) / p)`

| Segment | Score | Clients (test) | Taux de défaut | Décision |
| --- | --- | --- | --- | --- |
| Risque faible | 650 – 999 | 449 | 0,0 % | Approuver |
| Risque modéré | 550 – 650 | 66 | 0,0 % | Approuver avec garanties |
| Risque élevé | 0 – 550 | 85 | 51,8 % | Refuser ou étude approfondie |

## Analyse critique

- **Fuite de données détectée et corrigée** : la variable `credit_accorde` avait une corrélation de −1,0
  avec la cible ; elle a été exclue des variables explicatives.
- **Données synthétiques** : l'Information Value du revenu (10,69) dépasse très largement le seuil de 0,5
  au-delà duquel une IV est jugée suspecte. Le revenu mensuel seul, sans aucun modèle, atteint déjà une
  AUC de 0,984 : la cible est presque entièrement déterminée par le revenu. Les scores très élevés
  reflètent donc la structure de ce jeu de données pédagogique ; sur des données bancaires réelles, une
  AUC de 0,75 à 0,85 est habituellement considérée comme très bonne.

## Méthodologie

1. **Exploration** : statistiques descriptives, distributions, corrélations avec la cible.
2. **Prétraitement** : suppression des variables non prédictives ou de fuite, plafonnement des outliers (IQR).
3. **Feature engineering** : 6 ratios financiers (dette/revenu, crédit/revenu, score de stabilité…),
   Weight of Evidence et Information Value.
4. **Modélisation** : split stratifié 80/20, standardisation et **SMOTE appliqués sur le train uniquement**,
   GridSearchCV et validation croisée stratifiée, 4 modèles créés via un *Model Factory*.
5. **Évaluation** : matrices de confusion, ROC, Precision-Recall, statistique KS, courbes Lift et Gain.
6. **Interprétabilité** : importance des variables, valeurs SHAP, Partial Dependence Plots.
7. **Scoring** : grille PDO, segmentation des clients et politique de décision.

## Structure du projet

```
├── main.py                      Point d'entrée : exécute tout le pipeline
├── SRC/
│   ├── 01_exploration/          Analyse exploratoire
│   ├── 02_preprocessing/        Nettoyage et outliers
│   ├── 03_feature_engineering/  Ratios financiers, WoE / IV
│   ├── 04_modelisation/         Model Factory, SMOTE, GridSearchCV
│   ├── 05_evaluation/           ROC, KS, Lift / Gain
│   ├── 06_interpretability/     SHAP, importance des variables, PDP
│   ├── 07_scoring/              Grille PDO et segmentation
│   ├── config/                  Chemins et paramètres
│   └── utils/                   Logger, chargement des données
├── DATA/                        Données brutes, nettoyées et tables WoE
├── FIG/                         Figures générées
├── MODELS/                      Modèles entraînés (.joblib)
├── RESULTS/                     Résultats (CSV)
├── REP/                         Rapport (PDF et Word)
└── PPTX/                        Présentation
```

## Exécution

```bash
pip install -r requirements.txt
python main.py
```

Le pipeline complet s'exécute en environ 1 minute et régénère les figures, les modèles et les résultats.

## Technologies

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · SHAP · Matplotlib · Seaborn · joblib

## Contexte

Projet de Data Mining (ESISA Fès, 4ᵉ année, 2025-2026), encadré par Pr. Mohamed El Far.
Les données sont un jeu pédagogique fourni dans le cadre du cours.
Projet réalisé en binôme avec Ahmed Diouri ; j'ai réalisé l'ensemble du développement et de l'analyse.

**Auteur :** Marouane Boutayeb
