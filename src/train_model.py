import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === Chemin racine du projet ===
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Pour .py
except NameError:
    BASE_DIR = os.getcwd()  # Pour notebook

# === Chargement des données ===
X_train = joblib.load(os.path.join(BASE_DIR, "data", "X_train.pkl"))
X_test = joblib.load(os.path.join(BASE_DIR, "data", "X_test.pkl"))
y_train = joblib.load(os.path.join(BASE_DIR, "data", "y_train.pkl"))
y_test = joblib.load(os.path.join(BASE_DIR, "data", "y_test.pkl"))

# === Configuration de MLflow ===
mlflow.set_tracking_uri("file://" + os.path.join(BASE_DIR, "mlruns"))
mlflow.set_experiment("credit_scoring")

with mlflow.start_run(run_name="random_forest_run"):

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Log paramètres, métriques et modèle
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, preds))