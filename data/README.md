# Données et imputation des valeurs manquantes

Les données ne sont pas versionnées. Pour réentraîner le modèle, placer `cs-training.csv` (Kaggle, *Give Me Some Credit*) dans `data/raw/`.

---

## 1. Problème

La variable `MonthlyIncome` est manquante pour environ 20 % des clients : refus de communiquer, revenu irrégulier, omission de saisie. Quelques autres colonnes contiennent des valeurs manquantes (`NumberOfDependents`) ou des codes spéciaux (`age = 0`, valeurs 96 et 98 dans les colonnes de retards).

---

## 2. Pipeline de préparation (`src/data/preprocess.py`)

Ordre des opérations :

1. **Nettoyage des anomalies** (aucune statistique apprise) : `age = 0` et les codes 96 / 98 des trois colonnes de retards deviennent des valeurs manquantes.
2. **Séparation train / test** 80 / 20, stratifiée sur la cible (`random_state=42`).
3. **Ajustement des imputers sur le train uniquement** :
   * moyenne pour `age`, `NumberOfDependents` et les trois colonnes de retards ;
   * MissForest pour `MonthlyIncome`.
4. **Application** de ces imputers au train et au test.
5. L'indicateur `MonthlyIncome_is_missing` est créé **avant** l'imputation, et le revenu imputé est borné à 0 minimum.

Sorties : `data/processed/cs-train-clean.csv` et `cs-test-clean.csv`. L'étape suivante (`src/features/build_features.py`) produit `cs-train-features.csv` et `cs-test-features.csv`.

Artefacts sauvegardés dans `models/` et réutilisés par l'API : `income_imputer.joblib` et `dependents_mean.joblib`.

---

## 3. Pourquoi pas une simple moyenne ou médiane pour le revenu

Remplacer tous les revenus manquants par une même valeur ignore le profil du client (âge, charges, crédits en cours) et crée un pic artificiel dans la distribution. Le notebook `02_missing_data_comparison` compare plusieurs méthodes d'imputation.

---

## 4. MissForest (`IterativeImputer` avec Random Forest)

MissForest traite l'imputation comme un problème de prédiction : le revenu est prédit à partir des autres variables du client.

1. **Initialisation** : les revenus manquants sont remplacés temporairement par la moyenne de la colonne.
2. **Apprentissage** : un Random Forest est entraîné sur les clients dont le revenu est connu, avec 9 variables explicatives (âge, utilisation du crédit revolving, endettement, nombre de crédits ouverts, prêts immobiliers, personnes à charge, et les trois colonnes de retards).
3. **Prédiction** : il estime le revenu des clients dont la valeur manque.
4. **Itération** : les étapes 2 et 3 sont répétées, au maximum 5 fois.

Paramètres : 50 arbres, profondeur maximale 10, `max_features="sqrt"`.

---

## 5. Prévention de la fuite de données

* Le découpage train / test est fait **avant** toute imputation.
* Les imputers (moyennes et MissForest) sont ajustés sur le train seul, puis appliqués au test.
* En production, l'API applique le même `income_imputer.joblib` et la même moyenne de `NumberOfDependents` que ceux de l'entraînement.

Point restant : la validation croisée d'Optuna est réalisée sur un train déjà imputé, donc l'imputation n'est pas refaite à l'intérieur de chaque pli. Une fuite mineure entre plis est possible.

---

## 6. Avantages et limites

**Avantages**
* L'estimation tient compte du profil du client au lieu d'une valeur unique.
* L'indicateur `MonthlyIncome_is_missing` conserve l'information « revenu non renseigné ».

**Limites**
* Le revenu imputé est une prédiction : il ajoute de l'incertitude qui n'est pas propagée au modèle.
* Calcul plus long qu'une moyenne (l'imputer est sauvegardé pour ne pas être recalculé).
* Dans ces données, un revenu non renseigné est associé à un risque un peu plus bas. C'est une particularité du jeu de données, pas une règle métier.