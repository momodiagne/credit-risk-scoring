"""
src/features/build_features.py

Création des features retenues après l'analyse exploratoire (notebook 03).

Lancer avec :
    python -m src.features.build_features
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import load_raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cs-training-clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "cs-training-features.csv"


def load_clean() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_PATH)


def add_missingness_flags(df: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MonthlyIncome_is_missing"] = df_raw["MonthlyIncome"].isna().astype(int)
    return df


def add_past_due_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["WeightedPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1 +
        df["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
        df["NumberOfTimes90DaysLate"] * 3
    )
    df["HasAnyPastDue"] = (
        (df["NumberOfTime30-59DaysPastDueNotWorse"] +
         df["NumberOfTime60-89DaysPastDueNotWorse"] +
         df["NumberOfTimes90DaysLate"]) > 0
    ).astype(int)
    return df


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    return df


def build_features() -> pd.DataFrame:
    df_raw = load_raw()
    df = load_clean()

    df = add_missingness_flags(df, df_raw)
    df = add_past_due_features(df)
    df = add_financial_features(df)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)

    return df


if __name__ == "__main__":
    build_features()
