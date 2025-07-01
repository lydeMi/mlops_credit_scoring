import mlflow
from mlflow.tracking import MlflowClient

# Se connecter au tracking local
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# Nom de l'expérience
experiment_name = "credit_scoring"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    raise Exception(f"Expérience '{experiment_name}' introuvable.")

# Récupérer tous les runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

# Meilleur run
best_run = runs[0]
print("🎯 Meilleur run trouvé !")
print("Run ID :", best_run.info.run_id)
print("Accuracy :", best_run.data.metrics["accuracy"])
print("Modèle :", best_run.data.tags.get("mlflow.runName", "non spécifié"))