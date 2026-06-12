import streamlit as st
import numpy as np
import pandas as pd
import joblib
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
import os
import plotly.graph_objects as go

# === Configuration de la page (Doit être placé tout en haut) ===
st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="centered")

# === Chargement du scaler ===
scaler = joblib.load(os.path.join("data", "scaler.pkl"))

# === Configuration locale de MLflow ===
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
mlflow.set_tracking_uri("file:./mlruns")

client = MlflowClient()
experiment = client.get_experiment_by_name("credit_scoring")

if experiment is None:
    raise Exception("Expérience 'credit_scoring' introuvable.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

best_run = runs[0]
best_run_id = best_run.info.run_id

# === Solution : Chargement via le chemin relatif direct ===
model_path = "./mlruns/332181450071015150/dde4b570426a44b7a6ccebc1b6d19073/artifacts/model"
model = mlflow.sklearn.load_model(model_path)

# === Design de la page ===
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>💳 Application de Scoring Crédit</h1>
    <p style='text-align: center;'>Prédisez si un client remboursera ou non sa dette.</p>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.header("📌 Informations générales")
    st.markdown("- **Modèle utilisé :** Meilleur modèle MLflow")
    st.markdown(f"- **Run ID :** `{best_run_id}`")
    st.markdown("- **Source :** Base publique de scoring crédit")

# === Expander Description des variables ===
with st.expander("🧾 Description des variables utilisées", expanded=False):
    st.markdown("""
    - **LIMIT_BAL** : Montant du crédit accordé
    - **SEX** : Sexe (`1 = homme`, `2 = femme`)
    - **EDUCATION** : Niveau d'éducation (`1 = universitaire`, `2 = secondaire`, `3 = autre`)
    - **MARRIAGE** : Statut marital (`1 = célibataire`, `2 = marié`, `3 = autre`)
    - **AGE** : Âge du client
    - **PAY_0 à PAY_6** : Statut des 6 derniers paiements (-1 = avance, 0 = à temps, 1 = retard, etc.)
    - **BILL_AMT6** : Montant dû du dernier relevé
    - **PAY_AMT1 à PAY_AMT6** : Paiements effectués sur les 6 derniers mois
    """)

# === Formulaire utilisateur ===
st.subheader("📝 Saisissez les données du client :")

with st.form("credit_form"):
    LIMIT_BAL = st.number_input("💰 Montant du crédit accordé", min_value=0, step=1000)
    SEX = st.selectbox("👤 Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")
    EDUCATION = st.selectbox("🎓 Niveau d'éducation", [1, 2, 3],
                             format_func=lambda x: {1: "Université", 2: "Secondaire", 3: "Autre"}[x])
    MARRIAGE = st.selectbox("💍 Statut marital", [1, 2, 3],
                            format_func=lambda x: {1: "Célibataire", 2: "Marié", 3: "Autre"}[x])
    AGE = st.slider("🎂 Âge", min_value=18, max_value=100, value=35)

    st.markdown("#### 📅 Historique des paiements")
    PAY_0 = st.selectbox("PAY_0", list(range(-1, 9)))
    PAY_2 = st.selectbox("PAY_2", list(range(-1, 9)))
    PAY_3 = st.selectbox("PAY_3", list(range(-1, 9)))
    PAY_4 = st.selectbox("PAY_4", list(range(-1, 9)))
    PAY_5 = st.selectbox("PAY_5", list(range(-1, 9)))
    PAY_6 = st.selectbox("PAY_6", list(range(-1, 9)))

    BILL_AMT6 = st.number_input("📄 Montant dû du dernier relevé", min_value=0, step=100)

    st.markdown("#### 💸 Paiements effectués")
    PAY_AMT1 = st.number_input("PAY_AMT1", min_value=0, step=100)
    PAY_AMT2 = st.number_input("PAY_AMT2", min_value=0, step=100)
    PAY_AMT3 = st.number_input("PAY_AMT3", min_value=0, step=100)
    PAY_AMT4 = st.number_input("PAY_AMT4", min_value=0, step=100)
    PAY_AMT5 = st.number_input("PAY_AMT5", min_value=0, step=100)
    PAY_AMT6 = st.number_input("PAY_AMT6", min_value=0, step=100)

    submitted = st.form_submit_button("🎯 Prédire")

# === Initialiser l'historique dans la session si non existant ===
if "history" not in st.session_state:
    st.session_state["history"] = []

# === Traitement de la Prédiction ===
if submitted:
    input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                            BILL_AMT6,
                            PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # Proba défaut

    # Interprétation du risque
    if proba >= 0.7:
        interpretation = "❌ Risque ÉLEVÉ de non-remboursement"
    elif proba >= 0.4:
        interpretation = "⚠️ Risque MOYEN"
    else:
        interpretation = "✅ Risque FAIBLE"

    # Jauge visuelle
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "gold"},
                {'range': [70, 100], 'color': "crimson"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': proba * 100
            }
        }
    ))

    if prediction == 0:
        st.success("✅ Le client **REMBOURSERA** son crédit.")
        result_text = "REMBOURSERA"
    else:
        st.error("⚠️ Le client **NE REMBOURSERA PAS** son crédit.")
        result_text = "NE REMBOURSERA PAS"

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"### {interpretation}")

    # Stocker le résultat dans la session
    result = {
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": "Homme" if SEX == 1 else "Femme",
        "EDUCATION": {1: "Université", 2: "Secondaire", 3: "Autre"}[EDUCATION],
        "MARRIAGE": {1: "Célibataire", 2: "Marié", 3: "Autre"}[MARRIAGE],
        "AGE": AGE,
        "PREDICTION": result_text
    }
    st.session_state["history"].append(result)

# === Onglet Historique ===
st.subheader("🕒 Historique des prédictions")

if st.session_state["history"]:
    df_history = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_history)

    # Télécharger CSV
    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger l'historique (.csv)",
        data=csv,
        file_name="historique_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Aucune prédiction enregistrée pour l’instant.")
