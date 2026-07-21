"""
src/models/train.py

Entraînement et sauvegarde du modèle final LightGBM avec les hyperparamètres optimisés.

Lancer avec :
    python -m src.models.train
"""

from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "cs-training-features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_model.joblib"

TARGET = "SeriousDlqin2yrs"

# Hyperparamètres issus de l'optimisation Optuna (Notebook 04)
BEST_PARAMS = {
    "n_estimators": 467,
    "max_depth": 5,
    "learning_rate": 0.015979916201671447,
    "num_leaves": 77,
    "min_child_samples": 100,
    "reg_alpha": 4.435718419647249,
    "reg_lambda": 0.6610535955913199,
    "subsample": 0.8236130817018508,
    "colsample_bytree": 0.5627027462431399,
    "is_unbalance": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FEATURES_PATH}. "
            "Veuillez exécuter 'python -m src.features.build_features' d'abord."
        )
    return pd.read_csv(FEATURES_PATH)


def train_model() -> None:
    print("Chargement des données d'entraînement")
    df = load_features()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entraînement du modèle LightGBM")
    model = lgb.LGBMClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    # Évaluation rapide
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC sur le jeu de test : {auc:.4f}")

    # Sauvegarde du modèle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé avec succès dans : {MODEL_PATH}")


if __name__ == "__main__":
    train_model()