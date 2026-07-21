"""
run_pipeline.py

Pipeline End-to-End : exécute toute la chaîne en une commande.

Lancer avec :
    python run_pipeline.py
"""

from src.data.preprocess import run_pipeline as run_preprocessing
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate as evaluate_model
from src.models.explain import main as explain_model


def main():
    run_preprocessing()
    build_features()
    train_model()
    evaluate_model()
    explain_model()


if __name__ == "__main__":
    main()
