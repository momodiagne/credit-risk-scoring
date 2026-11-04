# Code source (pipeline de référence)

```
src/
├── config.py            # Hypothèses de coûts et seuil de décision
├── data/
│   ├── load.py          # Lecture du CSV brut
│   └── preprocess.py    # Nettoyage, split train/test, imputation
├── features/
│   ├── shared.py        # Définition des features, partagée avec l'API
│   └── build_features.py
└── models/
    ├── train.py         # Entraînement LightGBM
    ├── calibrate.py     # Calibration des probabilités (Platt scaling)
    ├── evaluate.py      # Métriques, coût, sensibilité, graphiques
    └── explain.py       # Importance globale SHAP
```

## Ordre d'exécution

Depuis la racine du dépôt, avec `data/raw/cs-training.csv` en place :

```bash
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train
python -m src.models.calibrate
python -m src.models.evaluate
python -m src.models.explain
```

| Étape            | Produit                                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------------------- |
| `preprocess`     | `data/processed/cs-{train,test}-clean.csv`, `models/income_imputer.joblib`, `models/dependents_mean.joblib` |
| `build_features` | `data/processed/cs-{train,test}-features.csv`                                                               |
| `train`          | `models/lgbm_model.joblib`                                                                                  |
| `calibrate`      | `models/calibrator.joblib`                                                                                  |
| `evaluate`       | `reports/metrics.json`, courbes ROC et de calibration, matrice de confusion                                 |
| `explain`        | graphiques SHAP dans `reports/`                                                                             |

## Règle à retenir

Les formules des features et la liste des variables de MissForest sont définies **une seule fois** dans `features/shared.py`. L'API les importe : toute modification se fait là, pas dans `api/`.