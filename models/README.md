# Modèles et artefacts

## Artefacts (`models/`)

| Fichier                  | Rôle                                                 |
| ------------------------ | ---------------------------------------------------- |
| `lgbm_model.joblib`      | Modèle LightGBM (score brut de défaut)               |
| `calibrator.joblib`      | Calibrateur : convertit le score brut en probabilité |
| `income_imputer.joblib`  | Imputer MissForest du revenu mensuel                 |
| `dependents_mean.joblib` | Moyenne du nombre de personnes à charge (train)      |

Ces fichiers sont versionnés pour permettre de lancer la démonstration sans les données. Ils dépendent des versions de Python (3.13) et des bibliothèques (scikit-learn 1.8.0, LightGBM 4.6.0, joblib 1.5.3). Avec d'autres versions, le chargement peut échouer ou donner des avertissements : l'image Docker utilise ces versions.

---

## 1. LightGBM

LightGBM est un algorithme de **gradient boosting** : les arbres de décision sont construits séquentiellement, chacun corrigeant les erreurs des précédents, avec un taux d'apprentissage qui pondère chaque contribution. Il construit les arbres feuille par feuille (*leaf-wise*) et utilise des histogrammes pour accélérer l'entraînement.

Dans ce projet :
* Cible : `SeriousDlqin2yrs` (1 = défaut dans les 2 ans).
* Le paramètre `is_unbalance=True` compense la rareté des défauts (6,7 %). Conséquence : les scores sont gonflés et ne sont **pas** des probabilités, d'où la calibration (section 2).
* Hyperparamètres : recherche Optuna en validation croisée à 5 plis sur le train (notebook 04), valeurs dans `src/models/train.py`. Une nouvelle recherche sur le pipeline corrigé n'a pas amélioré le résultat (AUC CV 0,8657 contre 0,8656 ± 0,0032).

Limites : un ensemble de centaines d'arbres n'est pas lisible directement (d'où SHAP), et le surapprentissage est possible si les paramètres sont mal réglés.

---

## 2. Calibration des probabilités (`src/models/calibrate.py`)

Avec `is_unbalance=True`, le score moyen est de 0,32 alors que le taux de défaut réel est de 0,067.

* Un **Platt scaling** (régression logistique sur le logit du score brut) est appris sur des prédictions **out-of-fold** du train (5 plis) : chaque client est prédit par un modèle qui ne l'a pas vu.
* Le test n'est jamais utilisé pour la calibration.
* Résultat sur le test : probabilité moyenne 6,73 % pour un taux réel de 6,68 %, Brier 0,141 → 0,049. L'AUC est inchangée, car la transformation est monotone.

Limite : le calibrateur est appris avec des modèles entraînés sur 80 % du train, alors que le modèle final l'est sur 100 %. L'écart n'a pas été mesuré.

---

## 3. Seuil de décision (`src/config.py`)

Un dossier est refusé si `p > marge / (perte + marge)` :
* perte si un futur défaillant est accordé : 58 % du montant prêté ;
* marge perdue si un bon client est refusé : 8 % du montant prêté ;
* seuil résultant : ≈ 12,1 %.

Ces deux valeurs sont des **hypothèses illustratives, non sourcées**. Le seuil n'est pas ajusté sur le jeu de test. La table de sensibilité est dans le `README.md` racine.

---

## 4. Imputation du revenu

Voir `data/README.md` (MissForest, ajusté sur le train uniquement).

---

## 5. Explicabilité : SHAP

L'API utilise `shap.TreeExplainer` sur le LightGBM. Pour chaque client, elle renvoie les 3 variables qui augmentent le plus le risque et les 3 qui le diminuent le plus.

* Les valeurs sont exprimées en **échelle de score brut** (log-odds) du LightGBM, pas en probabilité calibrée. Les signes et le classement sont fiables, mais les valeurs ne s'additionnent pas avec la probabilité affichée.
* SHAP décrit le comportement du modèle, pas des causes. C'est une explication post-hoc : elle ne constitue pas une démonstration de conformité réglementaire.
* `MonthlyIncome_is_missing` est une variable technique, masquée dans le dashboard.