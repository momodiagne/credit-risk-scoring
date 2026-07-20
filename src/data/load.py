"""
src/data/load.py

Chargement du dataset Give Me Some Credit (GMSC) et diagnostic du missingness.

Télécharger le dataset :
    https://www.kaggle.com/competitions/GiveMeSomeCredit/data

Placer cs-training.csv dans data/raw/
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"

TARGET = "SeriousDlqin2yrs"

def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Charge le dataset brut et nettoie la colonne d'index parasite."""
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un résumé du taux de valeurs manquantes par colonne,
    trié par pourcentage décroissant.
    """
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df) * 100).round(2)

    report = (
        pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
        .query("n_missing > 0")
        .sort_values("pct_missing", ascending=False)
    )
    return report


def class_balance(df: pd.DataFrame, target: str = TARGET) -> pd.Series:
    """Retourne la répartition des classes de la variable cible."""
    counts = df[target].value_counts()
    pct = (df[target].value_counts(normalize=True) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


if __name__ == "__main__":
    df = load_raw()

    print(f"Shape : {df.shape}")
    print(f"\nColonnes : {list(df.columns)}")

    print("\nRépartition de la cible (SeriousDlqin2yrs)")
    print(class_balance(df))

    print("\nRapport de missingness")
    report = missingness_report(df)
    print(report)

    if report.empty:
        print("Aucune valeur manquante détectée.")