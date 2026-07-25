# Credit Risk Scoring System

Système complet d'évaluation du risque de crédit bancaire et d'aide à la décision.

---

## Objectif et Métier

Dans le secteur bancaire, l'octroi d'un prêt nécessite d'évaluer la probabilité qu'un emprunteur se retrouve en situation de défaut de paiement dans les 2 ans.

Ce projet met en place une chaîne décisionnelle de bout en bout qui :
1. **Évalue le risque de défaut** à partir du profil financier de l'emprunteur.
2. **Prend une décision automatisée** (Crédit Accordé ou Refusé) basée sur un seuil de risque conservateur.
3. **Fournit une explicabilité totale** pour chaque décision individuelle, conformément aux exigences réglementaires de transparence bancaire (ACPR / RGPD).

---

## Points Forts Techniques

Ce projet se distingue par une rigueur méthodologique appliquée à chaque étape du pipeline :

* **Imputation avancée sans fuite de données (Data Leakage)** :
  La variable `MonthlyIncome` présente des valeurs manquantes. Plutôt qu'une simple imputation par la moyenne, nous utilisons l'algorithme **MissForest** (`IterativeImputer` par Random Forest). L'imputer est entraîné sur le jeu de données d'apprentissage et sauvegardé pour réutilisation directe lors de l'inférence API, garantissant une cohérence absolue entre entraînement et production.

* **Feature Engineering ciblé** :
  Création d'indicateurs métier stratégiques : score de retards cumulés pondéré (`WeightedPastDue`), indicateur binaire de présence d'impayés (`HasAnyPastDue`), et ratio de revenu disponible par personne à charge (`IncomePerPerson`).

* **Modélisation & Optimisation avec LightGBM et Optuna** :
  Choix de l'algorithme **LightGBM** pour sa rapidité, sa gestion des relations non linéaires et son efficacité sur des classes déséquilibrées. Recherche des hyperparamètres optimisée par l'algorithme bayésien **Optuna** (ROC-AUC > 0.86).

---

## Structure du Dépôt

```
credit-risk-scoring/
├── api/                  # Moteur d'inférence REST FastAPI
├── dashboard/            # Application web Streamlit et styles CSS
├── data/                 # Traitement des données financières (raw & processed)
├── models/               # Artefacts sauvegardés (LightGBM, MissForest, Mean)
├── notebooks/             # Cahiers d'analyse et d'expérimentation (EDA, Modélisation)
├── src/                  # Pipeline Python modulaire (preprocess, features, train)
├── Dockerfile             # Fichier de build des images Docker
├── docker-compose.yml     # Orchestration des services API et Dashboard
├── run_pipeline.py        # Script d'exécution End-to-End du pipeline
└── requirements.txt       # Dépendances du projet
```

---

## Guide de Démarrage Rapide

### Option 1 : Déploiement avec Docker Compose (Recommandé)

Pour builder et lancer l'ensemble des conteneurs isolés en une seule commande :

```bash
docker compose up --build -d
```

Accès aux services :
* **Application Conseiller (Streamlit)** : http://localhost:8501
* **Documentation API Swagger (FastAPI)** : http://localhost:8000/docs

Pour arrêter les conteneurs :
```bash
docker compose down
```

---

### Option 2 : Lancement en Environnement Local

1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optionnel) Ré-entraîner le pipeline complet** :
   ```bash
   python run_pipeline.py
   ```

3. **Lancer le serveur API** :
   ```bash
   uvicorn api.main:app --reload
   ```

4. **Lancer le Dashboard Streamlit** (dans un second terminal) :
   ```bash
   streamlit run dashboard/app.py
   ```