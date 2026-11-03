"""Vérifie que l'API construit les features comme le pipeline d'entraînement."""

import pandas as pd

from api.model_loader import ModelService
from api.schemas import ClientDataRequest
from src.features.shared import add_financial_features, add_past_due_features


def test_api_features_match_training_features():
    service = ModelService()

    req = ClientDataRequest(
        RevolvingUtilizationOfUnsecuredLines=0.4,
        age=35,
        NumberOfTime30_59DaysPastDueNotWorse=1,
        DebtRatio=0.3,
        MonthlyIncome=4000.0,
        NumberOfOpenCreditLinesAndLoans=5,
        NumberOfTimes90DaysLate=1,
        NumberRealEstateLoansOrLines=1,
        NumberOfTime60_89DaysPastDueNotWorse=2,
        NumberOfDependents=2.0,
    )
    api_features = service.prepare_features(req)

    # Même client, passé par les fonctions de l'entraînement
    training_row = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": 0.4,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.3,
        "MonthlyIncome": 4000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 2,
        "NumberOfDependents": 2.0,
        "MonthlyIncome_is_missing": 0,
    }])
    expected = add_financial_features(add_past_due_features(training_row))
    expected = expected[list(api_features.columns)]

    pd.testing.assert_frame_equal(
        api_features.reset_index(drop=True).astype(float),
        expected.reset_index(drop=True).astype(float),
    )