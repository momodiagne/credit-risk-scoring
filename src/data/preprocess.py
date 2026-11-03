"""
src/data/preprocess.py

Nettoyage, séparation train/test, puis imputation du dataset GMSC.
Les imputers sont ajustés sur le train uniquement, puis appliqués au test.

Lancer avec :
    python -m src.data.preprocess
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from src.features.shared import MISSFOREST_FEATURES
from src.data.load import TARGET, load_raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CLEAN_PATH = PROCESSED_DIR / "cs-train-clean.csv"
TEST_CLEAN_PATH = PROCESSED_DIR / "cs-test-clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
INCOME_IMPUTER_PATH = MODEL_DIR / "income_imputer.joblib"
DEPENDENTS_MEAN_PATH = MODEL_DIR / "dependents_mean.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42

SENTINEL_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_VALUES = [96, 98]

INCOME_COLUMNS = MISSFOREST_FEATURES + ["MonthlyIncome"]
MEAN_IMPUTED_COLUMNS = ["age", "NumberOfDependents", *SENTINEL_COLUMNS]

MISSFOREST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 10,
    "max_features": "sqrt",
    "max_iter": 5,
}


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les valeurs aberrantes par NaN (aucune statistique apprise ici)."""
    df = df.copy()
    df.loc[df["age"] == 0, "age"] = np.nan
    for col in SENTINEL_COLUMNS:
        df.loc[df[col].isin(SENTINEL_VALUES), col] = np.nan
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_imputers(train: pd.DataFrame) -> tuple[SimpleImputer, IterativeImputer]:
    """Ajuste les imputers sur le train uniquement."""
    mean_imputer = SimpleImputer(strategy="mean").fit(train[MEAN_IMPUTED_COLUMNS])

    train_filled = train.copy()
    train_filled[MEAN_IMPUTED_COLUMNS] = mean_imputer.transform(
        train[MEAN_IMPUTED_COLUMNS]
    )

    income_imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=MISSFOREST_PARAMS["n_estimators"],
            max_depth=MISSFOREST_PARAMS["max_depth"],
            max_features=MISSFOREST_PARAMS["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        max_iter=MISSFOREST_PARAMS["max_iter"],
        random_state=RANDOM_STATE,
    ).fit(train_filled[INCOME_COLUMNS])

    return mean_imputer, income_imputer


def apply_imputers(
    df: pd.DataFrame, mean_imputer: SimpleImputer, income_imputer: IterativeImputer
) -> pd.DataFrame:
    """Applique les imputers déjà ajustés (identique pour train et test)."""
    df = df.copy()

    # Le flag doit être calculé AVANT l'imputation
    df["MonthlyIncome_is_missing"] = df["MonthlyIncome"].isna().astype(int)

    df[MEAN_IMPUTED_COLUMNS] = mean_imputer.transform(df[MEAN_IMPUTED_COLUMNS])

    imputed = income_imputer.transform(df[INCOME_COLUMNS])
    df["MonthlyIncome"] = np.clip(imputed[:, -1], 0, None)
    return df


def validate(df: pd.DataFrame) -> None:
    assert df.isna().sum().sum() == 0, "Il reste des valeurs manquantes"
    assert (df["age"] > 0).all(), "Il reste des ages <= 0"
    assert (df["MonthlyIncome"] >= 0).all(), "Il reste des revenus négatifs"
    for col in SENTINEL_COLUMNS:
        assert not df[col].isin(SENTINEL_VALUES).any(), f"Sentinelles dans {col}"


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = clean_anomalies(load_raw())
    train, test = split_train_test(df)

    mean_imputer, income_imputer = fit_imputers(train)
    train = apply_imputers(train, mean_imputer, income_imputer)
    test = apply_imputers(test, mean_imputer, income_imputer)

    validate(train)
    validate(test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(income_imputer, INCOME_IMPUTER_PATH)
    dependents_idx = MEAN_IMPUTED_COLUMNS.index("NumberOfDependents")
    joblib.dump(float(mean_imputer.statistics_[dependents_idx]), DEPENDENTS_MEAN_PATH)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CLEAN_PATH, index=False)
    test.to_csv(TEST_CLEAN_PATH, index=False)
    return train, test


if __name__ == "__main__":
    run_pipeline()