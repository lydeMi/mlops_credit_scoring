import streamlit as st
import numpy as np
import pandas as pd
import joblib
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
import os

st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="💳",
    layout="centered"
)

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
mlflow.set_tracking_uri("file:./mlruns")


@st.cache_resource
def load_resources():
    scaler_object = joblib.load(os.path.join("data", "scaler.pkl"))
    model_path = "./mlruns/332181450071015150/dde4b570426a44b7a6ccebc1b6d19073/artifacts/model"
    model_object = mlflow.sklearn.load_model(model_path)
    return scaler_object, model_object


scaler, model = load_resources()

st.info(f"Nombre de variables attendu par le scaler : {scaler.n_features_in_}")

if hasattr(scaler, "feature_names_in_"):
    st.write("Colonnes attendues par le scaler :")
    st.write(list(scaler.feature_names_in_))


@st.cache_data
def get_best_run_id():
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("credit_scoring")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC"]
            )
            return runs[0].info.run_id if runs else "N/A"
    except Exception:
        return "dde4b570426a44b7a6ccebc1b6d19073"
    return "N/A"


best_run_id = get_best_run_id()

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>Application de Scoring Crédit</h1>
<p style='text-align: center;'>Prédisez si un client remboursera ou non sa dette.</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Informations générales")
    st.markdown("- **Modèle utilisé :** Meilleur modèle MLflow")
    st.markdown(f"- **Run ID :** `{best_run_id}`")
    st.markdown("- **Source :** Base publique de scoring crédit")

with st.expander("Description des variables utilisées", expanded=False):
    st.markdown("""
    - **LIMIT_BAL** : Montant du crédit accordé
    - **SEX** : Sexe (`1 = homme`, `2 = femme`)
    - **EDUCATION** : Niveau d'éducation (`1 = universitaire`, `2 = secondaire`, `3 = autre`)
    - **MARRIAGE** : Statut marital (`1 = célibataire`, `2 = marié`, `3 = autre`)
    - **AGE** : Âge du client
    - **PAY_0 à PAY_6** : Statut des 6 derniers paiements
    - **BILL_AMT1 à BILL_AMT6** : Montants des relevés mensuels
    - **PAY_AMT1 à PAY_AMT6** : Paiements effectués sur les 6 derniers mois
    """)

st.subheader("Saisissez les données du client :")

with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        LIMIT_BAL = st.number_input("Montant du crédit", min_value=0, step=1000, value=50000)
        SEX = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")

    with col2:
        EDUCATION = st.selectbox(
            "Éducation",
            [1, 2, 3],
            format_func=lambda x: {1: "Université", 2: "Secondaire", 3: "Autre"}[x]
        )
        MARRIAGE = st.selectbox(
            "Statut marital",
            [1, 2, 3],
            format_func=lambda x: {1: "Célibataire", 2: "Marié", 3: "Autre"}[x]
        )

    with col3:
        AGE = st.slider("Âge", min_value=18, max_value=100, value=35)

    st.markdown("#### Historique des statuts de paiement")
    c_pay1, c_pay2, c_pay3, c_pay4, c_pay5, c_pay6 = st.columns(6)

    PAY_0 = c_pay1.selectbox("Mois -1", list(range(-1, 9)), key="p0")
    PAY_2 = c_pay2.selectbox("Mois -2", list(range(-1, 9)), key="p2")
    PAY_3 = c_pay3.selectbox("Mois -3", list(range(-1, 9)), key="p3")
    PAY_4 = c_pay4.selectbox("Mois -4", list(range(-1, 9)), key="p4")
    PAY_5 = c_pay5.selectbox("Mois -5", list(range(-1, 9)), key="p5")
    PAY_6 = c_pay6.selectbox("Mois -6", list(range(-1, 9)), key="p6")

    st.markdown("#### Montant des relevés bancaires")
    cb1, cb2, cb3, cb4, cb5, cb6 = st.columns(6)

    BILL_AMT1 = cb1.number_input("BILL mois -1", min_value=0, step=100, key="b1")
    BILL_AMT2 = cb2.number_input("BILL mois -2", min_value=0, step=100, key="b2")
    BILL_AMT3 = cb3.number_input("BILL mois -3", min_value=0, step=100, key="b3")
    BILL_AMT4 = cb4.number_input("BILL mois -4", min_value=0, step=100, key="b4")
    BILL_AMT5 = cb5.number_input("BILL mois -5", min_value=0, step=100, key="b5")
    BILL_AMT6 = cb6.number_input("BILL mois -6", min_value=0, step=100, key="b6")

    st.markdown("#### Paiements mensuels effectués")
    cp1, cp2, cp3, cp4, cp5, cp6 = st.columns(6)

    PAY_AMT1 = cp1.number_input("PAY mois -1", min_value=0, step=100, key="pa1")
    PAY_AMT2 = cp2.number_input("PAY mois -2", min_value=0, step=100, key="pa2")
    PAY_AMT3 = cp3.number_input("PAY mois -3", min_value=0, step=100, key="pa3")
    PAY_AMT4 = cp4.number_input("PAY mois -4", min_value=0, step=100, key="pa4")
    PAY_AMT5 = cp5.number_input("PAY mois -5", min_value=0, step=100, key="pa5")
    PAY_AMT6 = cp6.number_input("PAY mois -6", min_value=0, step=100, key="pa6")

    submitted = st.form_submit_button("Prédire le risque")


if "history" not in st.session_state:
    st.session_state["history"] = []


if submitted:
    input_data = np.array([[
        LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
        PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
        PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
    ]])

    st.write("Nombre de variables envoyées au scaler :", input_data.shape[1])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if proba >= 0.7:
        interpretation = "Risque ÉLEVÉ de non-remboursement"
    elif proba >= 0.4:
        interpretation = "Risque MOYEN"
    else:
        interpretation = "Risque FAIBLE"

    if prediction == 0:
        st.success("Le client REMBOURSERA son crédit.")
        result_text = "REMBOURSERA"
    else:
        st.error("Le client NE REMBOURSERA PAS son crédit.")
        result_text = "NE REMBOURSERA PAS"

    st.metric(
        label="Probabilité de défaut",
        value=f"{proba * 100:.1f}%"
    )

    st.markdown(f"### {interpretation}")

    st.session_state["history"].append({
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": "Homme" if SEX == 1 else "Femme",
        "EDUCATION": {1: "Université", 2: "Secondaire", 3: "Autre"}[EDUCATION],
        "MARRIAGE": {1: "Célibataire", 2: "Marié", 3: "Autre"}[MARRIAGE],
        "AGE": AGE,
        "PREDICTION": result_text,
        "PROBA_DEFAUT": f"{proba * 100:.1f}%"
    })


st.subheader("Historique des prédictions")

if st.session_state["history"]:
    df_history = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_history, use_container_width=True)

    csv = df_history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Télécharger l'historique (.csv)",
        data=csv,
        file_name="historique_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Aucune prédiction enregistrée pour l’instant.")
