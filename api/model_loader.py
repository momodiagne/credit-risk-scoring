"""
api/model_loader.py

Chargement unique du modèle LightGBM, du calibrateur et de l'explainer SHAP,
et transformation des données clientes entrantes.

Les features sont calculées avec les MÊMES fonctions que l'entraînement
(src/features/shared.py) pour éviter tout décalage entre entraînement et production.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from api.schemas import ClientDataRequest
from src.features.shared import (
    MISSFOREST_FEATURES,
    add_financial_features,
    add_past_due_features,
)
from src.models.calibrate import CALIBRATOR_PATH, calibrate_proba

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.joblib"
INCOME_IMPUTER_PATH = PROJECT_ROOT / "models" / "income_imputer.joblib"
DEPENDENTS_MEAN_PATH = PROJECT_ROOT / "models" / "dependents_mean.joblib"


class ModelService:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.income_imputer = None
        self.dependents_mean = None
        self.calibrator = None
        self._load_model()

    def _load_model(self):
        for path in (MODEL_PATH, INCOME_IMPUTER_PATH, DEPENDENTS_MEAN_PATH, CALIBRATOR_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Fichier introuvable : {path}. "
                    "Exécutez 'python -m src.data.preprocess', "
                    "'python -m src.features.build_features', "
                    "'python -m src.models.train' puis 'python -m src.models.calibrate'."
                )

        self.model = joblib.load(MODEL_PATH)
        self.income_imputer = joblib.load(INCOME_IMPUTER_PATH)
        self.dependents_mean = joblib.load(DEPENDENTS_MEAN_PATH)
        self.calibrator = joblib.load(CALIBRATOR_PATH)
        self.explainer = shap.TreeExplainer(self.model)

    def _impute_income(self, base_row: dict) -> float:
        """Applique l'imputer MissForest déjà ajusté sur une ligne cliente."""
        cols = MISSFOREST_FEATURES + ["MonthlyIncome"]
        X_row = pd.DataFrame([base_row])[cols]
        imputed = self.income_imputer.transform(X_row)
        return max(float(imputed[0, -1]), 0.0)  # même clip(lower=0) qu'à l'entraînement

    def predict_default_proba(self, X: pd.DataFrame) -> float:
        """Probabilité de défaut corrigée (calibrée) pour un client."""
        raw = self.model.predict_proba(X)[:, 1]
        return float(calibrate_proba(self.calibrator, raw)[0])

    def prepare_features(self, req: ClientDataRequest) -> pd.DataFrame:
        """Transforme la requête brute en DataFrame avec les features du modèle."""
        dependents = (
            self.dependents_mean
            if req.NumberOfDependents is None
            else float(req.NumberOfDependents)
        )

        base_row = {
            "RevolvingUtilizationOfUnsecuredLines": req.RevolvingUtilizationOfUnsecuredLines,
            "age": req.age,
            "NumberOfTime30-59DaysPastDueNotWorse": req.NumberOfTime30_59DaysPastDueNotWorse,
            "DebtRatio": req.DebtRatio,
            "NumberOfOpenCreditLinesAndLoans": req.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate": req.NumberOfTimes90DaysLate,
            "NumberRealEstateLoansOrLines": req.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": req.NumberOfTime60_89DaysPastDueNotWorse,
            "NumberOfDependents": dependents,
            "MonthlyIncome": np.nan if req.MonthlyIncome is None else float(req.MonthlyIncome),
        }

        income_is_missing = 1 if req.MonthlyIncome is None else 0
        base_row["MonthlyIncome"] = self._impute_income(base_row)
        base_row["MonthlyIncome_is_missing"] = income_is_missing

        # Mêmes fonctions que l'entraînement
        df = pd.DataFrame([base_row])
        df = add_past_due_features(df)
        df = add_financial_features(df)

        # Ordre des colonnes attendu par le modèle
        expected_order = getattr(self.model, "feature_name_", None)
        if expected_order is not None:
            df = df[list(expected_order)]

        return df


# Instance globale réutilisée par FastAPI
model_service = ModelService()