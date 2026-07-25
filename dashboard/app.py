"""
Application Streamlit de Scoring et Décision de Crédit.
"""

import os
from pathlib import Path
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
STYLE_PATH = Path(__file__).parent / "style.css"

FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Utilisation crédits revolving",
    "age": "Âge de l'emprunteur",
    "NumberOfTime30_59DaysPastDueNotWorse": "Retards 30-59 jours",
    "DebtRatio": "Ratio d'endettement",
    "MonthlyIncome": "Revenu mensuel",
    "MonthlyIncome_is_missing": "Revenu",
    "NumberOfOpenCreditLinesAndLoans": "Crédits actifs",
    "NumberOfTimes90DaysLate": "Retards 90+ jours",
    "NumberRealEstateLoansOrLines": "Prêts immobiliers",
    "NumberOfTime60_89DaysPastDueNotWorse": "Retards 60-89 jours",
    "NumberOfDependents": "Personnes à charge",
    "WeightedPastDue": "Score retards cumulés",
    "DebtRatio_is_high": "Endettement critique",
    "RevolvingUtilization_is_high": "Plafond crédit atteint",
    "Income_per_Dependent": "Revenu par personne à charge",
    "Total_Past_Due": "Cumul retards de paiement",
}

# Icones SVG
SVG_BANK = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><path d="M3 21h18M3 10h18M5 6l7-3 7 3M4 10v11M20 10v11M8 14v3M12 14v3M16 14v3"/></svg>'
SVG_CHECK = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
SVG_CROSS = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
SVG_TREND_UP = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>'
SVG_TREND_DOWN = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>'


def clean_feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return name.replace("_is_missing", "").replace("_is_high", " élevé").replace("_", " ").capitalize()


def load_css():
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    st.markdown(
        f"""
        <div class="bank-navbar">
            <div class="bank-navbar-title">{SVG_BANK}Système d'Analyse & Décision Crédit</div>
            <div class="bank-navbar-subtitle">Évaluation automatisée de la solvabilité et analyse du risque emprunteur</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_inputs() -> dict:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 4px 0 10px 0;">
                <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: 0.06em; color: var(--text-color);">SCORING CRÉDIT</div>
                <div style="font-size: 0.76rem; opacity: 0.7; font-weight: 500; margin-top: 2px;">MODULE D'ÉVALUATION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.subheader("Saisie du Dossier Client")

        with st.expander("Profil Emprunteur", expanded=True):
            age = st.number_input("Âge du client (ans)", min_value=18, max_value=100, value=42)
            dependents = st.number_input("Personnes à charge", min_value=0.0, max_value=10.0, value=1.0, step=1.0)

        with st.expander("Situation Financière", expanded=True):
            income_unknown = st.checkbox("Revenu non communiqué", value=False)
            if income_unknown:
                income = None
                st.caption("Le revenu sera estimé par le modèle statistique.")
            else:
                income = st.number_input("Revenu Mensuel (€)", min_value=0.0, value=3500.0, step=100.0)

            revolving_util = st.slider("Utilisation crédit revolving (%)", 0.0, 100.0, 15.0, step=1.0) / 100.0
            debt_ratio = st.slider("Taux d'endettement (%)", 0.0, 100.0, 25.0, step=1.0) / 100.0
            open_lines = st.number_input("Crédits actifs en cours", min_value=0, value=4)
            real_estate = st.number_input("Prêts immobiliers en cours", min_value=0, value=1)

        with st.expander("Historique de Paiement", expanded=True):
            n_30_59 = st.number_input("Retards 30-59 jours", min_value=0, value=0)
            n_60_89 = st.number_input("Retards 60-89 jours", min_value=0, value=0)
            n_90 = st.number_input("Retards 90+ jours", min_value=0, value=0)

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Réévaluer le Dossier", type="primary", use_container_width=True)

    return {
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


def fetch_scoring(payload: dict) -> dict:
    res = requests.post(API_URL, json=payload, timeout=5)
    if res.status_code == 200:
        return res.json()
    st.error(f"Service indisponible ({res.status_code}).")
    st.stop()


def render_decision_summary(result: dict):
    proba = result["probabilite_defaut"]
    decision = result["decision"]
    niveau = result["niveau_risque"]
    seuil = result.get("seuil_applique", 0.30)
    badge_class = "badge-low" if niveau == "Faible" else ("badge-medium" if niveau == "Modéré" else "badge-high")

    if decision == "Accordé":
        card_html = f"""
        <div class="decision-card-granted">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3>{SVG_CHECK}Avis Favorable : Crédit Accordé</h3>
                <span class="{badge_class}">Risque {niveau}</span>
            </div>
            <p>L'évaluation financière indique un profil conforme aux critères de l'établissement. Probabilité de risque estimée à <strong>{proba:.1%}</strong> (seuil toléré : <strong>{seuil:.0%}</strong>).</p>
        </div>
        """
    else:
        card_html = f"""
        <div class="decision-card-refused">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3>{SVG_CROSS}Avis Défavorable : Crédit Refusé</h3>
                <span class="{badge_class}">Risque {niveau}</span>
            </div>
            <p>Le dossier franchit le seuil maximal de risque toléré. Probabilité de risque estimée à <strong>{proba:.1%}</strong> (seuil toléré : <strong>{seuil:.0%}</strong>).</p>
        </div>
        """

    st.markdown(card_html, unsafe_allow_html=True)
    st.markdown("**Niveau d'exposition au risque :**")
    st.progress(min(float(proba), 1.0))
    st.caption(f"Probabilité de défaut : {proba:.1%} (Seuil toléré : {seuil:.0%})")

    m1, m2, m3 = st.columns(3)
    m1.metric("Risque Estimé", f"{proba:.1%}")
    m2.metric("Seuil Toléré", f"{seuil:.0%}")
    m3.metric("Classification", niveau)


def render_impact_charts(result: dict):
    st.subheader("Analyse des Critères Décisionnels")
    st.caption("Principaux facteurs d'impact identifiés sur ce dossier.")

    col_fav, col_risk = st.columns(2)
    fav_list = result.get("facteurs_favorables", [])
    risk_list = result.get("facteurs_risque", [])

    with col_fav:
        st.markdown(f"#### {SVG_TREND_UP}Facteurs Favorables", unsafe_allow_html=True)
        if fav_list:
            df_fav = pd.DataFrame(fav_list)
            df_fav["feature_fr"] = df_fav["feature"].map(clean_feature_label)
            df_fav["impact_abs"] = df_fav["impact_shap"].abs()

            fig = px.bar(df_fav, x="impact_abs", y="feature_fr", orientation="h", color_discrete_sequence=["#059669"], labels={"impact_abs": "Poids favorable"})
            fig.update_traces(marker_cornerradius=5)
            fig.update_layout(height=260, margin=dict(l=0, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis={"showgrid": True})
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun facteur favorable prépondérant.")

    with col_risk:
        st.markdown(f"#### {SVG_TREND_DOWN}Facteurs de Vigilance", unsafe_allow_html=True)
        if risk_list:
            df_risk = pd.DataFrame(risk_list)
            df_risk["feature_fr"] = df_risk["feature"].map(clean_feature_label)

            fig = px.bar(df_risk, x="impact_shap", y="feature_fr", orientation="h", color_discrete_sequence=["#dc2626"], labels={"impact_shap": "Poids de risque"})
            fig.update_traces(marker_cornerradius=5)
            fig.update_layout(height=260, margin=dict(l=0, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis={"showgrid": True})
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun facteur de risque majeur identifié.")


def render_summary_table(payload: dict):
    st.subheader("Synthèse des Données Financières")
    income_val = payload["MonthlyIncome"]
    
    summary_data = {
        "Critère d'Évaluation": [
            "Âge du client",
            "Personnes à charge",
            "Revenu mensuel",
            "Taux d'utilisation du crédit revolving",
            "Taux d'endettement",
            "Nombre de crédits actifs",
            "Prêts immobiliers",
            "Retards de paiement (30-59 jours)",
            "Retards de paiement (60-89 jours)",
            "Retards majeurs (90+ jours)",
        ],
        "Valeur Saisie": [
            f"{payload['age']} ans",
            f"{int(payload['NumberOfDependents'])}",
            f"{income_val:,.0f} €".replace(",", " ") if income_val is not None else "Non communiqué (Estimé)",
            f"{payload['RevolvingUtilizationOfUnsecuredLines'] * 100:.1f} %",
            f"{payload['DebtRatio'] * 100:.1f} %",
            f"{payload['NumberOfOpenCreditLinesAndLoans']}",
            f"{payload['NumberRealEstateLoansOrLines']}",
            f"{payload['NumberOfTime30_59DaysPastDueNotWorse']}",
            f"{payload['NumberOfTime60_89DaysPastDueNotWorse']}",
            f"{payload['NumberOfTimes90DaysLate']}",
        ],
    }
    st.table(pd.DataFrame(summary_data))


def main():
    st.set_page_config(page_title="Système d'Analyse du Risque Crédit", page_icon=None, layout="wide", initial_sidebar_state="expanded")
    load_css()
    render_header()
    payload = render_sidebar_inputs()

    try:
        with st.spinner("Traitement du dossier en cours..."):
            result = fetch_scoring(payload)
        render_decision_summary(result)
        st.divider()
        render_impact_charts(result)
        st.divider()
        render_summary_table(payload)
    except requests.exceptions.ConnectionError:
        st.warning("Connecteur décisionnel indisponible. Assurez-vous que le service API est lancé.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'évaluation : {e}")


if __name__ == "__main__":
    main()