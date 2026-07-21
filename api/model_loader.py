"""
api/model_loader.py

Gestion du chargement unique du modèle LightGBM et de l'explainer SHAP en mémoire,
ainsi que la transformation des données clientes entrantes.

L'imputation de MonthlyIncome réutilise l'IterativeImputer (MissForest) entraîné
dans src/data/preprocess.py pour éviter tout écart
entre l'imputation faite à l'entraînement et celle faite en serving.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from api.schemas import ClientDataRequest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.joblib"
INCOME_IMPUTER_PATH = PROJECT_ROOT / "models" / "income_imputer.joblib"
DEPENDENTS_MEAN_PATH = PROJECT_ROOT / "models" / "dependents_mean.joblib"

# Doit être identique à MISSFOREST_FEATURES dans src/data/preprocess.py :
# ce sont les colonnes sur lesquelles l'imputer MonthlyIncome a été entraîné,
# dans le même ordre.
MISSFOREST_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


class ModelService:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.income_imputer = None
        self.dependents_mean = None
        self._load_model()

    def _load_model(self):
        for path in (MODEL_PATH, INCOME_IMPUTER_PATH, DEPENDENTS_MEAN_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Fichier introuvable : {path}. "
                    "Exécutez 'python -m src.data.preprocess' puis "
                    "'python -m src.models.train' d'abord."
                )

        self.model = joblib.load(MODEL_PATH)
        self.income_imputer = joblib.load(INCOME_IMPUTER_PATH)
        self.dependents_mean = joblib.load(DEPENDENTS_MEAN_PATH)
        self.explainer = shap.TreeExplainer(self.model)

    def _impute_income(self, base_row: dict) -> float:
        """
        Réplique l'imputation MissForest de src/data/preprocess.py pour une
        seule ligne cliente : construit le même jeu de features (avec NaN sur
        MonthlyIncome si manquant) et applique l'imputer fitted.
        """
        row = {col: base_row[col] for col in MISSFOREST_FEATURES}
        row["MonthlyIncome"] = base_row["MonthlyIncome"]  # peut être NaN

        cols = MISSFOREST_FEATURES + ["MonthlyIncome"]
        X_row = pd.DataFrame([row])[cols]

        imputed = self.income_imputer.transform(X_row)
        income_value = float(imputed[0, -1])
        return max(income_value, 0.0)  # même clip(lower=0) qu'à l'entraînement

    def prepare_features(self, req: ClientDataRequest) -> pd.DataFrame:
        """
        Transforme la requête Pydantic brute en DataFrame avec les 14 features
        exactes attendues par le modèle.
        """
        # 1. Renommage des colonnes Pydantic (underscores -> tirets)
        n_30_59 = req.NumberOfTime30_59DaysPastDueNotWorse
        n_60_89 = req.NumberOfTime60_89DaysPastDueNotWorse
        n_90 = req.NumberOfTimes90DaysLate

        # 2. Dependents : imputation par la moyenne d'entraînement, comme en train
        dependents_is_missing = req.NumberOfDependents is None
        dependents = (
            self.dependents_mean
            if dependents_is_missing
            else float(req.NumberOfDependents)
        )

        # 3. Income : imputation par le MissForest entraîné, pas une constante
        income_is_missing = 1 if req.MonthlyIncome is None else 0
        base_row = {
            "RevolvingUtilizationOfUnsecuredLines": req.RevolvingUtilizationOfUnsecuredLines,
            "age": req.age,
            "NumberOfTime30-59DaysPastDueNotWorse": n_30_59,
            "DebtRatio": req.DebtRatio,
            "NumberOfOpenCreditLinesAndLoans": req.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate": n_90,
            "NumberRealEstateLoansOrLines": req.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": n_60_89,
            "NumberOfDependents": dependents,
            "MonthlyIncome": (
                np.nan if req.MonthlyIncome is None else float(req.MonthlyIncome)
            ),
        }
        income = self._impute_income(base_row)

        # 4. Features engineered — calculées APRES imputation, sur les valeurs
        #    imputées, exactement comme dans src/features/build_features.py
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

        df = pd.DataFrame(data)

        # Sécurise l'ordre des colonnes attendu par le modèle (évite un
        # décalage silencieux si ce dict est réordonné un jour)
        expected_order = getattr(self.model, "feature_name_", None)
        if expected_order is not None:
            df = df[list(expected_order)]

        return df


# Instance globale réutilisée par FastAPI
model_service = ModelService()