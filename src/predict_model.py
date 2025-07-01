import mlflow
import joblib
import pandas as pd
from mlflow.tracking import MlflowClient
import os

# === Configuration ===
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()
experiment_name = "credit_scoring"

# === Récupérer le meilleur modèle ===
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise Exception(f"Expérience '{experiment_name}' introuvable.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)
best_run = runs[0]
run_id = best_run.info.run_id

print(f"✅ Chargement du modèle du run {run_id} avec une accuracy de {best_run.data.metrics['accuracy']:.4f}")

# === Charger le modèle ===
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# === Charger les données de test ===
BASE_DIR = os.getcwd()  # tu peux adapter si besoin
X_test = joblib.load(os.path.join(BASE_DIR, "data", "X_test.pkl"))

# === Faire des prédictions ===
preds = model.predict(X_test)
df_preds = pd.DataFrame(preds, columns=["Prediction"])

# === Afficher les 20 premières ===
print("🔍 Prédictions sur les 20 premières lignes :")
print(df_preds.head(20))