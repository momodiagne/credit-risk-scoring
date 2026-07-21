#!/bin/sh
# api/entrypoint.sh
# Lance le pipeline d'entraînement seulement si le modèle n'existe pas encore,
# puis démarre l'API. Évite de ré-entraîner à chaque redémarrage de conteneur.

set -e

if [ ! -f "models/lgbm_model.joblib" ] || [ ! -f "models/income_imputer.joblib" ] || [ ! -f "models/dependents_mean.joblib" ]; then
    echo "Un ou plusieurs artefacts manquants : exécution du pipeline d'entraînement..."
    python -m src.data.preprocess
    python -m src.features.build_features
    python -m src.models.train
else
    echo "Tous les artefacts existent dans models/ : pipeline ignoré."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000