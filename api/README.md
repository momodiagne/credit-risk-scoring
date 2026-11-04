# API de scoring (FastAPI)

Reçoit le profil d'un client et renvoie sa probabilité de défaut, une décision et les principaux facteurs explicatifs.

## Lancer

```bash
uvicorn api.main:app --reload
```

Documentation interactive : http://localhost:8000/docs. Les artefacts de `models/` doivent exister (ils sont versionnés).

## Points d'accès

* `GET /` : informations générales.
* `GET /health` : état du service et du modèle.
* `POST /predict` : prédiction pour un client.

## Réponse de `/predict`

| Champ                                     | Contenu                                                                          |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| `probabilite_defaut`                      | Probabilité de défaut à 2 ans, calibrée                                          |
| `decision`                                | « Refusé » si la probabilité atteint le seuil, sinon « Accordé »                 |
| `niveau_risque`                           | Faible (moins de la moitié du seuil), Modéré (jusqu'au seuil), Élevé (au-dessus) |
| `seuil_applique`                          | Seuil de `src/config.py` (≈ 0,12)                                                |
| `facteurs_risque` / `facteurs_favorables` | 3 variables chacun, avec valeur du client et impact SHAP                         |

Les impacts SHAP sont en échelle de score brut du modèle, pas en probabilité.

## Traitement des entrées

* Revenu absent : imputé par l'imputer MissForest de l'entraînement, avec l'indicateur `MonthlyIncome_is_missing = 1`.
* Nombre de personnes à charge absent : remplacé par la moyenne du train.
* Les entrées sont validées (âge, montants et ratios non négatifs, compteurs de retards limités). Une entrée invalide renvoie une erreur 422. Les codes spéciaux 96 et 98 du jeu GMSC sont rejetés.
* Une erreur interne renvoie un message générique (500) ; le détail reste dans les logs.

## Fichiers

* `main.py` : application, décision et niveau de risque.
* `model_loader.py` : chargement des artefacts, préparation des variables, prédiction. Utilise les fonctions de `src/features/shared.py`.
* `schemas.py` : structure et validation des données échangées.
* `entrypoint.sh` : démarrage dans Docker. Lance le pipeline d'entraînement seulement si un artefact manque.

## Tests

```bash
python -m pytest tests -v
```