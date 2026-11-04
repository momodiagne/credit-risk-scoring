# Credit Risk Scoring

Projet personnel : modèle de scoring de crédit qui estime la probabilité de défaut de paiement à 2 ans, entraîné sur le jeu de données public **Give Me Some Credit** (Kaggle), exposé par une API FastAPI et un dashboard Streamlit.

---

## Ce que fait le projet

1. **Estime la probabilité de défaut** d'un emprunteur à partir de son profil financier (LightGBM, probabilités calibrées).
2. **Propose une décision** (Accordé / Refusé) avec un seuil dérivé d'un rapport de coûts. Les hypothèses de coûts sont illustratives (voir Limites).
3. **Explique chaque décision** en affichant les principaux facteurs favorables et défavorables (valeurs SHAP).

---

## Résultats

Évaluation sur un jeu de test tenu à l'écart : 30 000 clients, dont 2 005 défauts (6,7 %).

| Métrique                                          | Valeur          |
| ------------------------------------------------- | --------------- |
| AUC-ROC                                           | 0,870           |
| Gini                                              | 0,741           |
| KS                                                | 0,583           |
| Brier, score brut → calibré                       | 0,141 → 0,049   |
| Probabilité moyenne prédite / taux de défaut réel | 6,73 % / 6,68 % |

Au seuil de décision retenu (≈ 12,1 %) : 14,4 % des dossiers sont refusés, **65 % des défauts sont détectés** et **30 %** des refus sont de vrais défauts.

Comparaison des modèles de base (AUC sur le test, notebook 04) : régression logistique 0,827, Random Forest 0,866, LightGBM 0,867. L'écart entre Random Forest et LightGBM n'est pas significatif sur un seul découpage.

### Sensibilité au rapport de coûts

Le seuil dépend du rapport entre le coût d'un défaut accordé et celui d'un bon client refusé. Retenu : 7,25.

| Rapport  | Seuil      | Dossiers refusés | Défauts détectés | Précision  |
| -------- | ---------- | ---------------- | ---------------- | ---------- |
| 2        | 33,3 %     | 5,0 %            | 36,7 %           | 48,6 %     |
| 3        | 25,0 %     | 7,3 %            | 46,4 %           | 42,7 %     |
| 5        | 16,7 %     | 10,6 %           | 57,5 %           | 36,3 %     |
| **7,25** | **12,1 %** | **14,4 %**       | **65,2 %**       | **30,2 %** |
| 10       | 9,1 %      | 19,1 %           | 71,7 %           | 25,1 %     |
| 20       | 4,8 %      | 31,9 %           | 85,3 %           | 17,9 %     |

---

## Choix méthodologiques

* **Séparation train/test avant toute imputation.** Les imputers sont ajustés sur le train uniquement, puis appliqués au test et à l'API.
* **Données manquantes.** Le revenu mensuel est imputé par MissForest (`IterativeImputer` avec Random Forest) et un indicateur `MonthlyIncome_is_missing` est conservé. Les personnes à charge, l'âge et les retards sont imputés par la moyenne. Voir `data/README.md`.
* **Features.** `WeightedPastDue`, `HasAnyPastDue` et `IncomePerPerson` sont définies une seule fois (`src/features/shared.py`) et partagées par l'entraînement et l'API. Un test vérifie la cohérence.
* **Modèle.** LightGBM. Les hyperparamètres viennent d'une recherche Optuna en validation croisée sur le train. Une nouvelle recherche sur le pipeline corrigé n'a pas amélioré le résultat (AUC CV 0,8657 contre 0,8656 ± 0,0032), les paramètres existants sont conservés.
* **Calibration.** `is_unbalance=True` gonfle les scores (score moyen de 0,32 pour un taux de défaut réel de 0,067). Un Platt scaling, appris sur des prédictions out-of-fold du train, les convertit en probabilités. Il ne change pas l'AUC.
* **Seuil.** Refus si `p > marge / (perte + marge)`, avec les hypothèses de `src/config.py` (perte de 58 % du montant prêté en cas de défaut, marge perdue de 8 % pour un bon client refusé). Le seuil n'est pas ajusté sur le jeu de test.

---

## Limites

* **Les hypothèses de coûts sont illustratives et non sourcées.** Le jeu de données ne contient aucune donnée de coût. Le coût moyen calculé par `evaluate.py` (2,15 % du montant prêté, contre 3,88 % si on accorde tout et 7,47 % si on refuse tout) dépend de ces hypothèses et n'est pas une validation indépendante. Les métriques qui n'en dépendent pas sont l'AUC, le KS et la calibration.
* **Jeu de données public.** Il n'est pas représentatif d'un portefeuille réel. Le calibrateur reflète son taux de défaut (6,7 %) et devrait être recalibré sur un autre portefeuille.
* **Un seul découpage train/test**, sans intervalle de confiance sur les métriques du test.
* **Explicabilité.** SHAP fournit une explication post-hoc du modèle, exprimée en échelle de score brut (log-odds) et non en probabilité. Ce n'est pas une démonstration de conformité réglementaire (ACPR, RGPD).
* **Équité non évaluée.** L'âge est utilisé comme variable, ce qui est sensible pour un usage de crédit.
* **`MonthlyIncome_is_missing`** : dans ces données, un revenu non renseigné est associé à un risque un peu plus bas. C'est une particularité du jeu de données, pas une règle métier. Cette variable n'est pas affichée dans le dashboard.
* **Optuna** : la validation croisée est faite sur un train déjà imputé, donc une fuite mineure entre plis est possible.
* **Calibrateur** appris avec des modèles entraînés sur 80 % du train, alors que le modèle final l'est sur 100 %. L'écart n'a pas été mesuré.
* **Reproductibilité** : les dépendances directes sont épinglées, pas les dépendances transitives. Le build Docker n'est pas garanti identique à l'octet.
* C'est un démonstrateur : pas de monitoring ni de ré-entraînement planifié.

---

## Structure du dépôt

```
credit-risk-scoring/
├── api/             # API REST FastAPI (prédiction + explications SHAP)
├── dashboard/       # Application Streamlit
├── data/            # Données brutes et transformées (non versionnées)
├── models/          # Artefacts pré-entraînés (LightGBM, imputers, calibrateur)
├── notebooks/       # Exploration et expérimentations (EDA, imputation, features, modèles)
├── reports/         # Métriques et graphiques générés (ROC, calibration, SHAP)
├── src/             # Pipeline de référence (preprocess, features, train, calibrate, evaluate)
├── tests/           # Tests pytest (API, cohérence des features)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Démarrage rapide (Docker)

Les modèles pré-entraînés sont inclus dans `models/` : **aucun téléchargement de données n'est nécessaire** pour la démonstration.

```bash
docker compose up --build -d
```

* Dashboard Streamlit : http://localhost:8501
* Documentation de l'API (Swagger) : http://localhost:8000/docs

```bash
docker compose down
```

---

## Réentraîner le modèle

Nécessite `cs-training.csv` (Kaggle, *Give Me Some Credit*), à placer dans `data/raw/`.

```bash
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train
python -m src.models.calibrate
python -m src.models.evaluate
python -m src.models.explain
```

Lancer sans Docker :

```bash
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

## Tests

```bash
python -m pytest tests -v
```