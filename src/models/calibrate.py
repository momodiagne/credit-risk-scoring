"""
src/models/calibrate.py

Corrige les scores du modèle pour qu'ils deviennent de vraies probabilités.

Lancer avec :
    python -m src.models.calibrate
"""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.models.train import BEST_PARAMS, MODEL_DIR, TARGET, TRAIN_FEATURES_PATH

CALIBRATOR_PATH = MODEL_DIR / "calibrator.joblib"
EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate_proba(calibrator, raw_proba) -> np.ndarray:
    """Transforme un score brut du modèle en probabilité corrigée."""
    return calibrator.predict_proba(_logit(raw_proba))[:, 1]


def fit_calibrator() -> None:
    df = pd.read_csv(TRAIN_FEATURES_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    raw = cross_val_predict(
        lgb.LGBMClassifier(**BEST_PARAMS), X, y, cv=cv, method="predict_proba"
    )[:, 1]

    calibrator = LogisticRegression(C=1e6)
    calibrator.fit(_logit(raw), y)
    corrected = calibrate_proba(calibrator, raw)

    print(f"Taux de défaut réel          : {y.mean():.4f}")
    print(f"Score moyen avant correction : {raw.mean():.4f}")
    print(f"Score moyen après correction : {corrected.mean():.4f}")
    print(f"Erreur (Brier) avant -> après : "
          f"{brier_score_loss(y, raw):.4f} -> {brier_score_loss(y, corrected):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    print(f"Sauvegardé dans : {CALIBRATOR_PATH}")


if __name__ == "__main__":
    fit_calibrator()