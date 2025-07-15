# MLOps Credit Scoring App

Une application complète de **scoring crédit** utilisant Python, MLflow, Streamlit, et Scikit-learn.  
Elle permet de prédire si un client remboursera ou non un crédit, à partir de données personnelles et financières.

---

## Objectifs

- Analyser des données de crédit
- Entraîner plusieurs modèles de machine learning
- Suivre les performances avec **MLflow**
- Déployer une interface interactive via **Streamlit**
- Automatiser la sélection du meilleur modèle
- Permettre la saisie manuelle et l’historique des prédictions

---

## Structure du projet

```bash
mlops_credit_scoring/
│
├── data/                  # Données brutes et transformées (X_train.pkl, scaler, etc.)
│   └── credit_data.csv
│
├── notebooks/             # Analyse exploratoire (EDA)
│   └── eda.ipynb
│
├── src/                   # Code source
│   ├── preprocessing.py   # Préparation des données
│   ├── train_model.py     # Entraînement et tracking MLflow
│   └── predict_model.py   # Script de prédiction simple
│
├── app/                   # Application Streamlit
│   └── app.py
│
├── mlruns/                # Dossier MLflow (expériences)
├── requirements.txt       # Liste des dépendances
├── README.md              # Ce fichier
└── .gitignore             # Fichiers à exclure du suivi Git
