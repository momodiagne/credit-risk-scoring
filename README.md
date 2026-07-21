# 💳 Credit Risk Scoring & Explainable AI (XAI)

Système complet de Scoring de Crédit bancaire combinant le modèle **LightGBM**, l'explicabilité **SHAP**, une **API REST FastAPI**, un **Dashboard Streamlit** et une **containerisation Docker**.

---

## 📌 Fonctionnalités

- **Pipeline de Données** : Nettoyage des anomalies (âge=0, codes 96/98) et imputation avancée par Random Forest (**MissForest**) pour le revenu mensuel.
- **Feature Engineering** : Création d'indicateurs de retard de paiement (`WeightedPastDue`, `HasAnyPastDue`) et ratios financiers (`IncomePerPerson`).
- **Modélisation & Optimisation** : LightGBM optimisé par **Optuna** (AUC > 0.86, Gini > 0.72).
- **Explicabilité (XAI)** : Explication des décisions au niveau global et individuel avec **SHAP**.
- **API REST (FastAPI)** : Service haute performance pour servir les prédictions et les facteurs SHAP par HTTP, avec une imputation du revenu cohérente entre entraînement et serving (réutilisation du même MissForest fitted, pas d'une constante).
- **Dashboard Interactive (Streamlit)** : Interface de simulation pour les analystes crédit.
- **Orchestration & DevOps** : Script d'exécution End-to-End (`run_pipeline.py`), suite de tests `pytest`, et déploiement `docker-compose`.

---

## Démarrage Rapide

### 1. Installation de l'environnement

```bash
git clone https://github.com/momodiagne/credit-risk-scoring.git
cd credit-risk-scoring
pip install -r requirements.txt
```

### 2. Modèles pré-entraînés inclus

Les artefacts entraînés (`models/*.joblib` : modèle LightGBM, imputer MissForest du revenu, moyenne de personnes à charge) sont **inclus dans le repo** afin que l'application fonctionne immédiatement après un clone, sans ré-entraînement préalable.

Pour régénérer le pipeline complet depuis zéro (nettoyage, imputation, feature engineering, entraînement) :

```bash
python run_pipeline.py
```

ou étape par étape :

```bash
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train
```

### 3. Démarrage de l'API FastAPI

```bash
uvicorn api.main:app --reload
```
Accédez à la documentation Swagger interactive sur : http://127.0.0.1:8000/docs

### 4. Démarrage du Dashboard Streamlit

```bash
streamlit run dashboard/app.py
```
Accédez à l'application sur : http://localhost:8501

### 5. Démarrage avec Docker Compose

Pour lancer l'ensemble de l'architecture (API + Dashboard) en conteneurs isolés :

```bash
docker-compose up --build
```

Vous pourrez accéder :
- À l'API sur [http://localhost:8000](http://localhost:8000)
- Au Dashboard sur [http://localhost:8501](http://localhost:8501)

Pour arrêter les conteneurs :
```bash
docker-compose down
```

---

## 📁 Structure du Projet

```
credit-risk-scoring/
├── api/                  # API REST FastAPI
│   ├── main.py           # Point d'entrée de l'API
│   └── ...
├── dashboard/            # Application Streamlit
│   └── app.py            # Application de scoring
├── data/                 # Données brutes et traitées (régénérées via le pipeline)
├── models/               # Modèles entraînés et explainers SHAP (inclus dans le repo)
├── reports/               # Rapports d'évaluation
├── notebooks/             # Notebooks Jupyter d'exploration
├── tests/                 # Tests unitaires et d'intégration
├── docker-compose.yml     # Configuration Docker & Orchestration
├── Dockerfile             # Build de l'image Docker
└── run_pipeline.py        # Script d'exécution complet
```
---

## 🧪 Tests

Pour lancer la suite de tests complète :

```bash
pytest
```