"""
tests/test_api.py

Tests automatisés pour les endpoints de l'API de scoring de crédit (FastAPI).

Lancer tous les tests avec :
    pytest
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_root():
    """Vérifie que la racine '/' réponds avec le message d'accueil."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "documentation" in data


def test_healthcheck():
    """Vérifie que le modèle est bien chargé et que le service est sain."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_client_accorde():
    """Vérifie qu'un client à très faible risque obtient un crédit 'Accordé'."""
    good_client_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.02,
        "age": 52,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.20,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }

    response = client.post("/predict", json=good_client_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["decision"] == "Accordé"
    assert data["probabilite_defaut"] < 0.30
    assert data["niveau_risque"] in ["Faible", "Modéré"]
    assert len(data["facteurs_favorables"]) > 0


def test_predict_client_refuse():
    """Vérifie qu'un client à fort risque (plusieurs retards graves) obtient un crédit 'Refusé'."""
    bad_client_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.95,
        "age": 25,
        "NumberOfTime30_59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.85,
        "MonthlyIncome": 1800.0,
        "NumberOfOpenCreditLinesAndLoans": 10,
        "NumberOfTimes90DaysLate": 3,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60_89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 3.0,
    }

    response = client.post("/predict", json=bad_client_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["decision"] == "Refusé"
    assert data["probabilite_defaut"] >= 0.30
    assert data["niveau_risque"] == "Élevé"
    assert len(data["facteurs_risque"]) > 0


def test_predict_donnees_invalides():
    """Vérifie que la validation Pydantic rejette un âge < 18 ans (HTTP 422)."""
    invalid_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.05,
        "age": 15,  # Âge invalide (< 18)
        "DebtRatio": 0.2,
        "NumberOfOpenCreditLinesAndLoans": 2,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
