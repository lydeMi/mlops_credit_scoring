import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === Détection du chemin racine du projet ===
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # exécution .py
except NameError:
    BASE_DIR = os.getcwd()  # exécution notebook

# === Chargement des données ===
X_train = joblib.load(os.path.join(BASE_DIR, "data", "X_train.pkl"))
X_test = joblib.load(os.path.join(BASE_DIR, "data", "X_test.pkl"))
y_train = joblib.load(os.path.join(BASE_DIR, "data", "y_train.pkl"))
y_test = joblib.load(os.path.join(BASE_DIR, "data", "y_test.pkl"))

# === Paramètres du modèle ===
params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": 42
}

# === MLflow tracking ===
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit_scoring")

with mlflow.start_run(run_name="GradientBoosting"):
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Log paramètres + métriques + modèle
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Accuracy: {acc:.4f}")
    print("📊 Confusion Matrix:\n", cm)
    print("📝 Classification Report:\n", classification_report(y_test, preds))