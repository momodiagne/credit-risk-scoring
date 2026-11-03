"""
src/models/evaluate.py

Évaluation du modèle calibré sur le jeu de test (jamais utilisé pour
l'entraînement ni la calibration).

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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from src.config import DECISION_THRESHOLD, LOSS_GIVEN_DEFAULT, NET_MARGIN_GOOD_LOAN
from src.models.calibrate import CALIBRATOR_PATH, calibrate_proba

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.joblib"
TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "cs-test-features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "SeriousDlqin2yrs"


def load_resources():
    for path in (MODEL_PATH, CALIBRATOR_PATH, TEST_FEATURES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    df = pd.read_csv(TEST_FEATURES_PATH)
    return model, calibrator, df.drop(columns=[TARGET]), df[TARGET].to_numpy()


def compute_ks_statistic(y_true, y_proba) -> float:
    return ks_2samp(y_proba[y_true == 0], y_proba[y_true == 1]).statistic


def average_cost(y_true, refuse) -> float:
    """Coût moyen par client, en % du montant prêté (hypothèses de config.py)."""
    faux_negatifs = int(((~refuse) & (y_true == 1)).sum())  # défaut accordé
    faux_positifs = int((refuse & (y_true == 0)).sum())     # bon client refusé
    return (faux_negatifs * LOSS_GIVEN_DEFAULT + faux_positifs * NET_MARGIN_GOOD_LOAN) / len(y_true)


def evaluate() -> None:
    model, calibrator, X_test, y_test = load_resources()

    raw = model.predict_proba(X_test)[:, 1]
    y_proba = calibrate_proba(calibrator, raw)
    refuse = y_proba >= DECISION_THRESHOLD
    y_pred = refuse.astype(int)

    # 1. Métriques
    auc = roc_auc_score(y_test, y_proba)
    metrics = {
        "AUC-ROC": round(float(auc), 4),
        "Gini": round(float(2 * auc - 1), 4),
        "KS": round(float(compute_ks_statistic(y_test, y_proba)), 4),
        "Brier (score brut)": round(float(brier_score_loss(y_test, raw)), 4),
        "Brier (calibré)": round(float(brier_score_loss(y_test, y_proba)), 4),
        "Seuil de décision": round(float(DECISION_THRESHOLD), 4),
    }
    print("\nMétriques sur le jeu de test")
    for name, value in metrics.items():
        print(f"{name:<20}: {value}")
    print(f"{'Taux de défaut réel':<20}: {y_test.mean():.4f}")
    print(f"{'Proba moyenne prédite':<20}: {y_proba.mean():.4f}")

    # 2. Ce que fait le seuil retenu
    print(f"\nSeuil retenu ({DECISION_THRESHOLD:.4f})")
    print(classification_report(y_test, y_pred, target_names=["Pas de défaut", "Défaut"]))

    # 3. Coût moyen : modèle contre stratégies simples (en % du montant prêté)
    cost_model = average_cost(y_test, refuse)
    cost_accept_all = average_cost(y_test, np.zeros(len(y_test), dtype=bool))
    cost_refuse_all = average_cost(y_test, np.ones(len(y_test), dtype=bool))
    metrics["Coût moyen modèle (% prêté)"] = round(100 * cost_model, 2)
    metrics["Coût moyen tout accorder (% prêté)"] = round(100 * cost_accept_all, 2)
    metrics["Coût moyen tout refuser (% prêté)"] = round(100 * cost_refuse_all, 2)
    print("Coût moyen par client (% du montant prêté)")
    print(f"Modèle + seuil    : {100 * cost_model:.2f} %")
    print(f"Tout accorder     : {100 * cost_accept_all:.2f} %")
    print(f"Tout refuser      : {100 * cost_refuse_all:.2f} %")

    # 4. Sensibilité au rapport de coûts FN:FP
    ratio_retenu = LOSS_GIVEN_DEFAULT / NET_MARGIN_GOOD_LOAN
    print(f"\nSensibilité au rapport de coûts (retenu : {ratio_retenu:.2f})")
    print(f"{'ratio':>6} | {'seuil':>6} | {'refus':>6} | {'défauts détectés':>16} | {'précision':>9}")
    n_defauts = int((y_test == 1).sum())
    for ratio in (2, 3, 5, ratio_retenu, 10, 20):
        seuil = 1 / (1 + ratio)
        pred = y_proba >= seuil
        tp = int((pred & (y_test == 1)).sum())
        precision = tp / max(int(pred.sum()), 1)
        print(f"{ratio:>6.2f} | {seuil:>6.3f} | {pred.mean():>6.1%} | {tp / n_defauts:>16.1%} | {precision:>9.1%}")

    # 5. Sauvegarde des résultats et graphiques
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Pas de défaut", "Défaut"], ax=ax, cmap="Blues"
    )
    plt.title(f"Matrice de confusion (seuil = {DECISION_THRESHOLD:.3f})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"LightGBM calibré (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC (jeu de test)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, "o-", label="Modèle calibré")
    plt.plot([0, mean_pred.max()], [0, mean_pred.max()], "k--", label="Calibration parfaite")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Taux de défaut observé")
    plt.title("Courbe de calibration (jeu de test)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=300)
    plt.close()

    print(f"\nRésultats et graphiques sauvegardés dans : {REPORTS_DIR}")


if __name__ == "__main__":
    evaluate()