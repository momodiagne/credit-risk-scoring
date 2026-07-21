"""
src/models/explain.py

Explicabilité du modèle LightGBM avec SHAP (global et individuel).

Lancer avec :
    python -m src.models.explain
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "cs-training-features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "SeriousDlqin2yrs"


def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Exécutez 'python -m src.models.train' d'abord."
        )
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    return model, X


def get_tree_explainer(model):
    """Crée un explainer SHAP adapté aux modèles à base d'arbres (TreeExplainer)."""
    return shap.TreeExplainer(model)


def generate_global_explanation(explainer, X_sample: pd.DataFrame) -> None:
    """Génère et sauvegarde le graphique d'importance globale SHAP (Beeswarm plot)."""
    print("Calcul des valeurs SHAP globales")
    shap_values = explainer(X_sample)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "shap_summary.png"

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Explicabilité Globale SHAP", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Graphique SHAP global sauvegardé dans : {output_path}")


def explain_single_client(explainer, client_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'impact de chaque variable sur le score d'un client unique.
    Utile pour l'API et le Dashboard.
    """
    shap_values = explainer(client_features)
    values = shap_values.values[0]

    impact_df = pd.DataFrame(
        {
            "feature": client_features.columns,
            "valeur_client": client_features.iloc[0].values,
            "impact_shap": values,
        }
    ).sort_values(by="impact_shap", ascending=False)

    return impact_df


def main():
    model, X = load_resources()

    # Échantillon de 2000 clients pour accélérer le calcul global
    X_sample = X.sample(n=2000, random_state=42)

    explainer = get_tree_explainer(model)
    generate_global_explanation(explainer, X_sample)

    # Exemple sur le tout premier client
    client_1 = X.iloc[[0]]
    print("\nExplication pour le Client #1 ")
    impact_client_1 = explain_single_client(explainer, client_1)
    print(impact_client_1.head(5))


if __name__ == "__main__":
    main()
