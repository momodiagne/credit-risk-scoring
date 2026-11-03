"""Tests de l'API de scoring (nécessite les artefacts dans models/)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import DECISION_THRESHOLD

client = TestClient(app)

CLIENT_SAIN = {
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "age": 42,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.25,
    "MonthlyIncome": 3500.0,
    "NumberOfOpenCreditLinesAndLoans": 4,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2.0,
}

CLIENT_RISQUE = {
    "RevolvingUtilizationOfUnsecuredLines": 1.2,
    "age": 25,
    "NumberOfTime30_59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.9,
    "MonthlyIncome": None,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 3,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60_89DaysPastDueNotWorse": 1,
    "NumberOfDependents": None,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_client_sain_est_accorde():
    r = client.post("/predict", json=CLIENT_SAIN)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "Accordé"
    assert body["niveau_risque"] == "Faible"
    assert 0.0 <= body["probabilite_defaut"] < DECISION_THRESHOLD


def test_client_risque_est_refuse_meme_sans_revenu_ni_dependants():
    r = client.post("/predict", json=CLIENT_RISQUE)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "Refusé"
    assert body["niveau_risque"] == "Élevé"
    assert body["probabilite_defaut"] >= DECISION_THRESHOLD


def test_seuil_renvoye_est_celui_de_la_config():
    body = client.post("/predict", json=CLIENT_SAIN).json()
    assert body["seuil_applique"] == pytest.approx(DECISION_THRESHOLD, abs=1e-4)


def test_reponse_contient_trois_facteurs_de_chaque_cote():
    body = client.post("/predict", json=CLIENT_RISQUE).json()
    assert len(body["facteurs_risque"]) == 3
    assert len(body["facteurs_favorables"]) == 3
    assert all(f["impact_shap"] > 0 for f in body["facteurs_risque"])
    assert all(f["impact_shap"] < 0 for f in body["facteurs_favorables"])


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("age", 10),                       # mineur
        ("age", 200),                      # irréaliste
        ("MonthlyIncome", -100.0),         # revenu négatif
        ("DebtRatio", -1.0),               # ratio négatif
        ("NumberOfTimes90DaysLate", -1),   # compteur négatif
    ],
)
def test_entrees_invalides_rejetees(champ, valeur):
    payload = {**CLIENT_SAIN, champ: valeur}
    assert client.post("/predict", json=payload).status_code == 422


def test_champ_obligatoire_manquant_rejete():
    payload = {k: v for k, v in CLIENT_SAIN.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422

@pytest.mark.parametrize(
    "champ",
    [
        "NumberOfTime30_59DaysPastDueNotWorse",
        "NumberOfTime60_89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ],
)
@pytest.mark.parametrize("valeur", [96, 98])
def test_codes_speciaux_gmsc_rejetes(champ, valeur):
    payload = {**CLIENT_SAIN, champ: valeur}
    assert client.post("/predict", json=payload).status_code == 422


def test_erreur_interne_ne_fuit_pas_de_details(monkeypatch):
    from api.main import model_service

    def boom(_X):
        raise ValueError("détail interne secret")

    monkeypatch.setattr(model_service, "predict_default_proba", boom)
    r = client.post("/predict", json=CLIENT_SAIN)
    assert r.status_code == 500
    assert "secret" not in r.text