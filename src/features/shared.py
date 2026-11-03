"""
src/features/shared.py

Définitions partagées entre l'entraînement (src/) et l'API (api/).
Toute modification de feature se fait ICI, une seule fois.
"""

import pandas as pd

# Colonnes sur lesquelles l'imputer MonthlyIncome est ajusté, dans cet ordre
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

COL_30_59 = "NumberOfTime30-59DaysPastDueNotWorse"
COL_60_89 = "NumberOfTime60-89DaysPastDueNotWorse"
COL_90 = "NumberOfTimes90DaysLate"


def add_past_due_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["WeightedPastDue"] = df[COL_30_59] * 1 + df[COL_60_89] * 2 + df[COL_90] * 3
    df["HasAnyPastDue"] = ((df[COL_30_59] + df[COL_60_89] + df[COL_90]) > 0).astype(int)
    return df


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    return df