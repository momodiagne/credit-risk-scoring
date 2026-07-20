"""
dashboard/eda_app.py

App Streamlit d'exploration du dataset GMSC.

Lancer avec :
    streamlit run dashboard/eda_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

RAW_DATA_PATH = Path("data/raw/cs-training.csv")
LABEL_COLUMN = "SeriousDlqin2yrs"

# Colonnes avec des valeurs suspectes connues, à surveiller à l'oeil
SENTINEL_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_VALUES = [96, 98]


st.set_page_config(page_title="GMSC - Exploration des données", layout="wide")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df) * 100).round(2)
    report = (
        pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
        .query("n_missing > 0")
        .sort_values("pct_missing", ascending=False)
    )
    return report


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Recense les valeurs suspectes connues, sans les modifier."""
    rows = []

    n_age_zero = (df["age"] == 0).sum()
    if n_age_zero > 0:
        rows.append({"colonne": "age", "anomalie": "valeur = 0", "n_lignes": n_age_zero})

    for col in SENTINEL_COLUMNS:
        if col in df.columns:
            n_sentinel = df[col].isin(SENTINEL_VALUES).sum()
            if n_sentinel > 0:
                rows.append(
                    {
                        "colonne": col,
                        "anomalie": f"valeur dans {SENTINEL_VALUES}",
                        "n_lignes": n_sentinel,
                    }
                )

    return pd.DataFrame(rows)


if not RAW_DATA_PATH.exists():
    st.error(
        f"Fichier introuvable : {RAW_DATA_PATH}. "
        "Télécharge cs-training.csv depuis Kaggle et place-le dans data/raw/."
    )
    st.stop()

df = load_data(RAW_DATA_PATH)

st.title("Exploration des données")
st.caption("Analyse exploratoire avant nettoyage et imputation.")

# Vue d'ensemble
col1, col2, col3 = st.columns(3)
col1.metric("Lignes", f"{df.shape[0]:,}")
col2.metric("Colonnes", df.shape[1])
col3.metric("Taux de défaut", f"{df[LABEL_COLUMN].mean() * 100:.2f} %")

st.divider()

# Répartition de la cible
st.subheader("Répartition de la variable cible")
target_counts = df[LABEL_COLUMN].value_counts().rename({0: "Pas de défaut", 1: "Défaut"})
fig_target = px.bar(
    x=target_counts.index,
    y=target_counts.values,
    labels={"x": "Statut", "y": "Nombre de clients"},
    text=target_counts.values,
)
st.plotly_chart(fig_target, use_container_width=True)
st.divider()

# Missingness
st.subheader("Valeurs manquantes")
report = missingness_report(df)

if report.empty:
    st.success("Aucune valeur manquante détectée.")
else:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.dataframe(report, use_container_width=True)
    with col_right:
        fig_missing = px.bar(
            report,
            x=report.index,
            y="pct_missing",
            labels={"x": "Colonne", "pct_missing": "% manquant"},
            text="pct_missing",
        )
        st.plotly_chart(fig_missing, use_container_width=True)

st.divider()

# Anomalies détectées
st.subheader("Valeurs suspectes détectées")
st.caption(
    "Ces valeurs ne sont pas modifiées ici - ce module sert uniquement à les repérer. "
    "Le nettoyage (recodage en NaN) et l'imputation se font dans src/data/imputation.py"
    "pas dans cette app."
)

anomalies = detect_anomalies(df)
if anomalies.empty:
    st.success("Aucune anomalie connue détectée.")
else:
    st.dataframe(anomalies, use_container_width=True)

    st.markdown("#### Âge: un `age = 0` est impossible")
    col_age1, col_age2 = st.columns(2)
    with col_age1:
        fig_age_hist = px.histogram(df, x="age", nbins=50, title="Distribution de age")
        st.plotly_chart(fig_age_hist, use_container_width=True)
    with col_age2:
        fig_age_box = px.box(df, y="age", title="Boxplot de age")
        st.plotly_chart(fig_age_box, use_container_width=True)
    st.caption(
        "Les valeurs hautes (96-110) sont plausibles humainement, même si le boxplot les "
        "marque comme outliers statistiques. Seul age = 0 est une vraie anomalie à corriger."
    )

    st.markdown("Colonnes de retard de paiement - codes suspects 96/98")
    fig_sentinel = plt.figure(figsize=(14, 3.5))
    for i, col in enumerate(SENTINEL_COLUMNS):
        ax = fig_sentinel.add_subplot(1, 3, i + 1)
        df[col].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Nombre de lignes")
    plt.tight_layout()
    st.pyplot(fig_sentinel)
    st.caption(
        "Un grand vide entre les valeurs normales (0-13) et un pic isolé à 96/98 sur les "
        "3 colonnes en même temps, signe d'un code système plutôt que de vrais comptages."
    )

st.divider()

# Distribution par variable
st.subheader("Distribution d'une variable")
numeric_columns = [c for c in df.columns if c != LABEL_COLUMN]
selected_col = st.selectbox("Choisis une colonne", numeric_columns)

col_a, col_b = st.columns(2)
with col_a:
    clip_upper = st.checkbox("Ignorer les valeurs extrêmes (99e percentile)", value=True)

series = df[selected_col].dropna()
if clip_upper:
    upper_bound = series.quantile(0.99)
    series = series[series <= upper_bound]

fig_dist = px.histogram(series, nbins=50, labels={"value": selected_col})
fig_dist.update_layout(showlegend=False)
st.plotly_chart(fig_dist, use_container_width=True)

st.write(series.describe())