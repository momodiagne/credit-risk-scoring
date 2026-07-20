"""
src/data/preprocess.py

Pipeline de nettoyage et d'imputation du dataset GMSC.

Lancer avec :
    python -m src.data.preprocess
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from src.data.load import load_raw

PROCESSED_PATH = Path("data/processed/cs-training-clean.csv")

SENTINEL_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_VALUES = [96, 98]

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

MISSFOREST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 10,
    "max_features": "sqrt",
    "max_iter": 5,
}


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["age"] == 0, "age"] = np.nan
    for col in SENTINEL_COLUMNS:
        df.loc[df[col].isin(SENTINEL_VALUES), col] = np.nan
    return df


def impute_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    imputer = SimpleImputer(strategy="mean")
    df[column] = imputer.fit_transform(df[[column]]).ravel()
    return df


def impute_monthly_income(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = MISSFOREST_FEATURES + ["MonthlyIncome"]

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=MISSFOREST_PARAMS["n_estimators"],
            max_depth=MISSFOREST_PARAMS["max_depth"],
            max_features=MISSFOREST_PARAMS["max_features"],
            random_state=42,
            n_jobs=-1,
        ),
        max_iter=MISSFOREST_PARAMS["max_iter"],
        random_state=42,
    )

    imputed = imputer.fit_transform(df[cols])
    df["MonthlyIncome"] = imputed[:, -1]
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(lower=0)
    return df


def impute_all(df: pd.DataFrame) -> pd.DataFrame:
    df = impute_mean(df, "age")
    df = impute_mean(df, "NumberOfDependents")
    for col in SENTINEL_COLUMNS:
        df = impute_mean(df, col)
    df = impute_monthly_income(df)
    return df


def validate(df: pd.DataFrame) -> None:
    assert df.isna().sum().sum() == 0, "Il reste des valeurs manquantes"
    assert (df["age"] > 0).all(), "Il reste des ages <= 0"
    assert (df["MonthlyIncome"] >= 0).all(), "Il reste des revenus négatifs"
    for col in SENTINEL_COLUMNS:
        assert not df[col].isin(SENTINEL_VALUES).any(), f"Sentinelles dans {col}"


def run_pipeline() -> pd.DataFrame:
    df = load_raw()
    df = clean_anomalies(df)
    df = impute_all(df)
    validate(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    return df


if __name__ == "__main__":
    run_pipeline()