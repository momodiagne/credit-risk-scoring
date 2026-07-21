"""
api/main.py

API REST FastAPI pour le Scoring de Crédit et l'Explicabilité SHAP en Temps Réel.

Lancer le serveur de dev avec :
    uvicorn api.main:app --reload
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from api.model_loader import model_service
from api.schemas import ClientDataRequest, FeatureImpact, PredictionResponse

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API de prédiction de risque de crédit bancaire avec explicabilité SHAP.",
    version="1.0.0",
)

SEUIL_DECISION = 0.30  # Seuil conservateur : au-dessus de 30% de probabilité de défaut, on refuse


@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de Scoring de Crédit",
        "documentation": "/docs",
        "healthcheck": "/health",
    }


@app.get("/health")
def healthcheck():
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(request: ClientDataRequest):
    try:
        # 1. Préparation des variables
        X_client = model_service.prepare_features(request)

        # 2. Prédiction de la probabilité de défaut
        proba = float(model_service.model.predict_proba(X_client)[0, 1])

        # 3. Prise de décision & Niveau de risque
        decision = "Refusé" if proba >= SEUIL_DECISION else "Accordé"

        if proba < 0.15:
            niveau_risque = "Faible"
        elif proba < 0.30:
            niveau_risque = "Modéré"
        else:
            niveau_risque = "Élevé"

        # 4. Calcul de l'explicabilité SHAP
        shap_values = model_service.explainer(X_client)
        impacts = shap_values.values[0]

        impact_df = pd.DataFrame(
            {
                "feature": X_client.columns,
                "valeur_client": X_client.iloc[0].values,
                "impact_shap": impacts,
            }
        )

        # Top 3 facteurs augmentant le risque (SHAP > 0)
        top_risque = (
            impact_df[impact_df["impact_shap"] > 0]
            .sort_values(by="impact_shap", ascending=False)
            .head(3)
        )

        # Top 3 facteurs réduisant le risque (SHAP < 0)
        top_favorables = (
            impact_df[impact_df["impact_shap"] < 0]
            .sort_values(by="impact_shap", ascending=True)
            .head(3)
        )

        facteurs_risque = [
            FeatureImpact(
                feature=row["feature"],
                valeur_client=float(row["valeur_client"]),
                impact_shap=float(row["impact_shap"]),
            )
            for _, row in top_risque.iterrows()
        ]

        facteurs_favorables = [
            FeatureImpact(
                feature=row["feature"],
                valeur_client=float(row["valeur_client"]),
                impact_shap=float(row["impact_shap"]),
            )
            for _, row in top_favorables.iterrows()
        ]

        return PredictionResponse(
            probabilite_defaut=round(proba, 4),
            decision=decision,
            niveau_risque=niveau_risque,
            seuil_applique=SEUIL_DECISION,
            facteurs_risque=facteurs_risque,
            facteurs_favorables=facteurs_favorables,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
