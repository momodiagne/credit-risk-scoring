# Notebooks d'exploration

Notebooks d'exploration et d'expérimentation. **Le pipeline de référence est dans `src/`** : les notebooks justifient les choix, ils ne produisent pas les artefacts livrés.

## Liste

* `01_eda.ipynb` : exploration des données brutes.
* `02_missing_data_comparison.ipynb` : comparaison de méthodes d'imputation.
* `03_feature_engineering.ipynb` : création et sélection des features, sur le train uniquement.
* `04_modeling.ipynb` : comparaison de modèles (régression logistique, Random Forest, LightGBM) et recherche d'hyperparamètres avec Optuna.

## Exécution

Les notebooks 03 et 04 lisent des fichiers produits par `src/`. Depuis la racine du dépôt :

```bash
python -m src.data.preprocess          # nécessaire au notebook 03
python -m src.features.build_features  # nécessaire au notebook 04
```

## À savoir

* Les sorties des cellules ne sont pas versionnées (`nbstripout`). Il faut exécuter les notebooks pour voir les graphiques et les résultats.
* Dans le notebook 04, le rapport de classification est calculé à un seuil de 0,5 sur des scores bruts, uniquement pour comparer les modèles. Le seuil de production et les probabilités calibrées sont traités dans `src/models/calibrate.py` et `src/models/evaluate.py`.
* La recherche Optuna du notebook 04 peut durer plusieurs minutes.