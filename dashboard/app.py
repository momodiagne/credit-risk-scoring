"""
dashboard/app.py

Application Streamlit de Scoring de Crédit et d'Explicabilité SHAP en Direct.

Lancer avec :
    streamlit run dashboard/app.py
"""

import requests
import streamlit as st
import plotly.express as px
import pandas as pd

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


# Configuration de la page
st.set_page_config(
    page_title="Scoring de Crédit Bancaire",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Système de Scoring & Décision de Crédit")
st.caption("Évaluation du risque de défaut d'emprunt et explicabilité SHAP par l'API FastAPI.")

# Formulaire latéral : Données de l'emprunteur
with st.sidebar:
    st.header("👤 Profil de l'Emprunteur")
    st.markdown("---")

    age = st.slider("Âge du client", 18, 100, 42)

    income_unknown = st.checkbox("Revenu inconnu / non communiqué")
    if income_unknown:
        income = None
        st.caption("Le revenu sera estimé automatiquement (imputation MissForest).")
    else:
        income = st.number_input("Revenu Mensuel (€)", min_value=0.0, value=3500.0, step=100.0)

    dependents = st.number_input("Nombre de personnes à charge", min_value=0.0, value=2.0, step=1.0)

    st.markdown("### 📊 Données Financières")
    revolving_util = st.slider("Taux d'utilisation crédit revolving (%)", 0.0, 100.0, 5.0) / 100.0
    debt_ratio = st.slider("Taux d'endettement (Debt Ratio)", 0.0, 2.0, 0.25, step=0.01)
    open_lines = st.number_input("Nombre de crédits ouverts", min_value=0, value=5)
    real_estate = st.number_input("Nombre de prêts immobiliers", min_value=0, value=1)

    st.markdown("### ⚠️ Retards de Paiement")
    n_30_59 = st.number_input("Retards 30-59 jours", min_value=0, value=0)
    n_60_89 = st.number_input("Retards 60-89 jours", min_value=0, value=0)
    n_90 = st.number_input("Retards 90+ jours", min_value=0, value=0)

    submit = st.button("🚀 Évaluer le Dossier", type="primary", use_container_width=True)

# Préparation du payload API
client_payload = {
    "RevolvingUtilizationOfUnsecuredLines": float(revolving_util),
    "age": int(age),
    "NumberOfTime30_59DaysPastDueNotWorse": int(n_30_59),
    "DebtRatio": float(debt_ratio),
    "MonthlyIncome": float(income) if income is not None else None,
    "NumberOfOpenCreditLinesAndLoans": int(open_lines),
    "NumberOfTimes90DaysLate": int(n_90),
    "NumberRealEstateLoansOrLines": int(real_estate),
    "NumberOfTime60_89DaysPastDueNotWorse": int(n_60_89),
    "NumberOfDependents": float(dependents),
}

# Zone principale d'affichage des résultats
if submit or "result" in st.session_state:
    try:
        if submit:
            with st.spinner("Analyse du dossier et calcul SHAP par l'API..."):
                res = requests.post(API_URL, json=client_payload, timeout=20)
                if res.status_code == 200:
                    st.session_state["result"] = res.json()
                else:
                    st.error(f"Erreur de l'API ({res.status_code}) : {res.text}")
                    st.stop()

        result = st.session_state["result"]
        proba = result["probabilite_defaut"]
        decision = result["decision"]
        niveau = result["niveau_risque"]

        # 1. Cartes de décision
        col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

        with col_res1:
            if decision == "Accordé":
                st.success(f"### ✅ CRÉDIT ACCORDÉ\nRisque estimé : **{proba:.1%}** (Niveau {niveau})")
            else:
                st.error(f"### ❌ CRÉDIT REFUSÉ\nRisque estimé : **{proba:.1%}** (Niveau {niveau})")

        with col_res2:
            st.metric("Probabilité de Défaut", f"{proba:.1%}", delta=f"Seuil : {result['seuil_applique']:.0%}", delta_color="inverse")

        with col_res3:
            st.metric("Niveau de Risque", niveau)

        st.divider()

        # 2. Explicabilité SHAP
        st.subheader("🔍 Explicabilité de la Décision (SHAP)")

        col_favorable, col_risque = st.columns(2)

        with col_favorable:
            st.markdown("#### 🟢 Principaux points forts (Réducteurs de risque)")
            fav_df = pd.DataFrame(result["facteurs_favorables"])
            if not fav_df.empty:
                fig_fav = px.bar(
                    fav_df,
                    x="impact_shap",
                    y="feature",
                    orientation="h",
                    text_auto=".3f",
                    color_discrete_sequence=["#22C55E"],
                    labels={"impact_shap": "Impact SHAP (Réduction du risque)", "feature": "Variable"},
                )
                fig_fav.update_layout(yaxis={"categoryorder": "total descending"}, height=300)
                st.plotly_chart(fig_fav, use_container_width=True)
            else:
                st.info("Aucun point fort majeur identifié.")

        with col_risque:
            st.markdown("#### 🔴 Principaux facteurs de risque (Augmentateurs)")
            risk_df = pd.DataFrame(result["facteurs_risque"])
            if not risk_df.empty:
                fig_risk = px.bar(
                    risk_df,
                    x="impact_shap",
                    y="feature",
                    orientation="h",
                    text_auto=".3f",
                    color_discrete_sequence=["#EF4444"],
                    labels={"impact_shap": "Impact SHAP (Augmentation du risque)", "feature": "Variable"},
                )
                fig_risk.update_layout(yaxis={"categoryorder": "total ascending"}, height=300)
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.info("Aucun facteur de risque majeur identifié.")

    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Impossible de contacter l'API. Assurez-vous que le serveur FastAPI est démarré "
            "dans un terminal avec : `uvicorn api.main:app --reload`"
        )
    except requests.exceptions.Timeout:
        st.error("⚠️ L'API met trop de temps à répondre. Réessayez dans quelques instants.")
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Erreur de connexion à l'API : {e}")