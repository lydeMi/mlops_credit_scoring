import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Charger les données
df = pd.read_csv("/Users/lydem/Downloads/mlops_credit_scoring/data/credit_Card.csv")

# 2. Sélection des colonnes utiles
selected_features = [
    'LIMIT_BAL',
    'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',  # sociologique
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',  # historique de paiement
    'BILL_AMT6',  # dernier montant dû
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'  # paiements effectués
]

X = df[selected_features]
y = df["default.payment.next.month"]

# 3. Division en train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Sauvegarde
joblib.dump(X_train_scaled, "/Users/lydem/Downloads/mlops_credit_scoring/data/X_train.pkl")
joblib.dump(X_test_scaled, "/Users/lydem/Downloads/mlops_credit_scoring/data/X_test.pkl")
joblib.dump(y_train, "/Users/lydem/Downloads/mlops_credit_scoring/data/y_train.pkl")
joblib.dump(y_test, "/Users/lydem/Downloads/mlops_credit_scoring/data/y_test.pkl")
joblib.dump(scaler, "/Users/lydem/Downloads/mlops_credit_scoring/data/scaler.pkl")

print("✅ Données sélectionnées, standardisées et sauvegardées dans `data/`.")