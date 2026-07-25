# Modèles & Artefacts d'Évaluation

Ce dossier regroupe les fichiers de calcul sauvegardés qui permettent à l'application de répondre instantanément lors des demandes d'évaluation.

## Fichiers inclus

* `lgbm_model.joblib` : Modèle de scoring calculant la probabilité de risque.
* `income_imputer.joblib` : Outil statistique estimant le revenu d'un client lorsque celui-ci n'est pas renseigné.
* `dependents_mean.joblib` : Valeur moyenne de référence pour le nombre de personnes à charge.
