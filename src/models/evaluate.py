"""
src/models/evaluate.py

Évaluation approfondie du modèle sauvegardé (AUC, Gini, KS, Matrice de confusion).

Lancer avec :
    python -m src.models.evaluate
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "cs-training-features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "SeriousDlqin2yrs"
THRESHOLD = 0.5  # Seuil de décision par défaut (ou optimisé)


def load_data_and_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Exécutez 'python -m src.models.train' d'abord."
        )
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return model, X_test, y_test


def compute_ks_statistic(y_true, y_proba):
    """Calcule la statistique Kolmogorov-Smirnov (KS)."""
    proba_non_defaut = y_proba[y_true == 0]
    proba_defaut = y_proba[y_true == 1]
    res = ks_2samp(proba_non_defaut, proba_defaut)
    return res.statistic


def evaluate():
    print("Chargement du modèle et des données de test")
    model, X_test, y_test = load_data_and_model()

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    # 1. Métriques principales
    auc = roc_auc_score(y_test, y_proba)
    gini = 2 * auc - 1
    ks_stat = compute_ks_statistic(y_test, y_proba)

    metrics = {
        "AUC-ROC": round(float(auc), 4),
        "Gini": round(float(gini), 4),
        "KS-Statistic": round(float(ks_stat), 4),
    }

    print("Métriques Réglementaires")
    for k, v in metrics.items():
        print(f"{k:<15} : {v}")

    print("\nRapport de Classification")
    report = classification_report(y_test, y_pred, target_names=["Pas de défaut", "Défaut"])
    print(report)

    # 2. Sauvegarde des métriques en JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 3. Sauvegarde de la Matrice de Confusion
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Pas de défaut", "Défaut"], ax=ax, cmap="Blues"
    )
    plt.title(f"Matrice de Confusion (Seuil = {THRESHOLD})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # 4. Sauvegarde de la Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"LightGBM (AUC = {auc:.4f}, Gini = {gini:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC — Évaluation du Modèle")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    print(f"\nÉvaluation terminée. Rapports et graphiques sauvegardés dans : {REPORTS_DIR}")


if __name__ == "__main__":
    evaluate()
