# train_logistic.py
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

X_train = joblib.load(os.path.join(BASE_DIR, "data", "X_train.pkl"))
X_test = joblib.load(os.path.join(BASE_DIR, "data", "X_test.pkl"))
y_train = joblib.load(os.path.join(BASE_DIR, "data", "y_train.pkl"))
y_test = joblib.load(os.path.join(BASE_DIR, "data", "y_test.pkl"))

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit_scoring")

with mlflow.start_run(run_name="logistic_regression"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc:.4f}\nConfusion Matrix:\n{cm}")
