"""
api/model_loader.py

Gestion du chargement unique du modèle LightGBM et de l'explainer SHAP en mémoire,
ainsi que la transformation des données clientes entrantes.
"""

from pathlib import Path
import joblib
import pandas as pd
import shap
from api.schemas import ClientDataRequest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.joblib"

# Moyenne par défaut pour le revenu si non renseigné (issues du jeu d'entraînement)
DEFAULT_MONTHLY_INCOME_MEAN = 6670.0
DEFAULT_DEPENDENTS_MEAN = 0.75


class ModelService:
    def __init__(self):
        self.model = None
        self.explainer = None
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_PATH}. Exécutez 'python -m src.models.train' d'abord."
            )
        self.model = joblib.load(MODEL_PATH)
        self.explainer = shap.TreeExplainer(self.model)

    def prepare_features(self, req: ClientDataRequest) -> pd.DataFrame:
        """
        Transforme la requête Pydantic brute en DataFrame avec les 14 features
        exactes attendues par le modèle.
        """
        # 1. Gestion des valeurs manquantes & drapeaux
        income_is_missing = 1 if req.MonthlyIncome is None else 0
        income = (
            DEFAULT_MONTHLY_INCOME_MEAN
            if req.MonthlyIncome is None
            else float(req.MonthlyIncome)
        )
        dependents = (
            DEFAULT_DEPENDENTS_MEAN
            if req.NumberOfDependents is None
            else float(req.NumberOfDependents)
        )

        # 2. Renommage des colonnes Pydantic (remplacement des underscores en tirets pour les retards)
        n_30_59 = req.NumberOfTime30_59DaysPastDueNotWorse
        n_60_89 = req.NumberOfTime60_89DaysPastDueNotWorse
        n_90 = req.NumberOfTimes90DaysLate

        # 3. Calcul des variables agrégées / enginées
        weighted_past_due = (n_30_59 * 1) + (n_60_89 * 2) + (n_90 * 3)
        total_past_due = n_30_59 + n_60_89 + n_90
        has_any_past_due = 1 if total_past_due > 0 else 0
        income_per_person = income / (dependents + 1.0)

        data = {
            "RevolvingUtilizationOfUnsecuredLines": [req.RevolvingUtilizationOfUnsecuredLines],
            "age": [req.age],
            "NumberOfTime30-59DaysPastDueNotWorse": [n_30_59],
            "DebtRatio": [req.DebtRatio],
            "MonthlyIncome": [income],
            "NumberOfOpenCreditLinesAndLoans": [req.NumberOfOpenCreditLinesAndLoans],
            "NumberOfTimes90DaysLate": [n_90],
            "NumberRealEstateLoansOrLines": [req.NumberRealEstateLoansOrLines],
            "NumberOfTime60-89DaysPastDueNotWorse": [n_60_89],
            "NumberOfDependents": [dependents],
            "MonthlyIncome_is_missing": [income_is_missing],
            "WeightedPastDue": [weighted_past_due],
            "HasAnyPastDue": [has_any_past_due],
            "IncomePerPerson": [income_per_person],
        }

        return pd.DataFrame(data)


# Instance globale réutilisée par FastAPI
model_service = ModelService()
