# Dashboard conseiller (Streamlit)

Interface de simulation qui interroge l'API de scoring.

## Lancer

L'API doit tourner. L'adresse se règle par la variable d'environnement `API_URL` (par défaut `http://127.0.0.1:8000/predict`).

```bash
streamlit run dashboard/app.py
```

Avec Docker : `docker compose up --build -d` puis http://localhost:8501.

## Contenu

1. **Formulaire latéral** : profil financier du client (case « Revenu non communiqué » possible).
2. **Boîte de décision** : décision, niveau de risque et probabilité de défaut.
3. **Graphiques explicatifs** : facteurs favorables et facteurs de vigilance (valeurs SHAP).
4. **Tableau de synthèse** des critères saisis.

## À savoir

* Le seuil de décision est celui renvoyé par l'API (`src/config.py`).
* La variable technique `MonthlyIncome_is_missing` n'est pas affichée dans les graphiques (constante `TECHNICAL_FEATURES`).
* Les poids affichés sont des valeurs SHAP en échelle de score brut, comparables entre eux mais pas des points de probabilité.

## Fichiers

* `app.py` : application Streamlit.
* `style.css` : feuille de style.