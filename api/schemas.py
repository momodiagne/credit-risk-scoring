"""
api/schemas.py

Schémas de validation Pydantic pour les requêtes et réponses de l'API de scoring.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ClientDataRequest(BaseModel):
    """Données brutes fournies par le conseiller ou l'application pour un client."""

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0.0, description="Taux d'utilisation du crédit revolving (ex: 0.05 pour 5%)"
    )
    age: int = Field(..., ge=18, le=120, description="Âge du client")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        0, ge=0, description="Nombre de retards de 30 à 59 jours"
    )
    DebtRatio: float = Field(..., ge=0.0, description="Taux d'endettement (ou dette si revenu manquant)")
    MonthlyIncome: Optional[float] = Field(
        None, ge=0.0, description="Revenu mensuel du client (peut être null)"
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., ge=0, description="Nombre de crédits et prêts ouverts"
    )
    NumberOfTimes90DaysLate: int = Field(
        0, ge=0, description="Nombre de retards de 90 jours ou plus"
    )
    NumberRealEstateLoansOrLines: int = Field(
        0, ge=0, description="Nombre de prêts immobiliers"
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        0, ge=0, description="Nombre de retards de 60 à 89 jours"
    )
    NumberOfDependents: Optional[float] = Field(
        0.0, ge=0.0, description="Nombre de personnes à charge"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.05,
                "age": 42,
                "NumberOfTime30_59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.25,
                "MonthlyIncome": 3500.0,
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60_89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2.0,
            }
        }
    }


class FeatureImpact(BaseModel):
    """Impact d'une variable spécifique sur la décision (SHAP)."""

    feature: str
    valeur_client: float
    impact_shap: float


class PredictionResponse(BaseModel):
    """Réponse détaillée retournée par l'API."""

    probabilite_defaut: float = Field(..., description="Probabilité de défaut (entre 0.0 et 1.0)")
    decision: str = Field(..., description="Accordé ou Refusé")
    niveau_risque: str = Field(..., description="Faible, Modéré, ou Élevé")
    seuil_applique: float = Field(..., description="Seuil de décision appliqué")
    facteurs_risque: List[FeatureImpact] = Field(
        ..., description="Top variables qui augmentent le risque"
    )
    facteurs_favorables: List[FeatureImpact] = Field(
        ..., description="Top variables qui réduisent le risque"
    )
