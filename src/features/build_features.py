"""
src/features/build_features.py

Création des features retenues après l'analyse exploratoire (notebook 03).
Les formules sont dans src/features/shared.py (partagées avec l'API).

Lancer avec :
    python -m src.features.build_features
"""

from pathlib import Path

import pandas as pd

from src.features.shared import add_financial_features, add_past_due_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = {
    "cs-train-clean.csv": "cs-train-features.csv",
    "cs-test-clean.csv": "cs-test-features.csv",
}


def build_features() -> None:
    for clean_name, features_name in SPLITS.items():
        df = pd.read_csv(PROCESSED_DIR / clean_name)
        df = add_past_due_features(df)
        df = add_financial_features(df)
        df.to_csv(PROCESSED_DIR / features_name, index=False)


if __name__ == "__main__":
    build_features()