import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("road_accident_prediction")

def load_data(data_path="./data/preprocessed"):
    """Charge les données preprocessées"""
    # Adapter selon la structure réelle de tes données
    # Exemple générique
    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))
    
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train_model(X_train, y_train, params=None):
    """Entraîne le modèle avec les paramètres donnés"""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et retourne les métriques"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics

def main():
    """Pipeline principal d'entraînement avec MLflow tracking"""
    
    print("🚀 Démarrage de l'entraînement avec MLflow...")
    
    # Chargement des données
    print("📊 Chargement des données...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Paramètres du modèle
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    # Démarrage du run MLflow
    with mlflow.start_run(run_name="random_forest_training"):
        
        # Log des paramètres
        mlflow.log_params(params)
        
        # Entraînement
        print("🎯 Entraînement du modèle...")
        model = train_model(X_train, y_train, params)
        
        # Évaluation
        print("📈 Évaluation du modèle...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log des métriques
        mlflow.log_metrics(metrics)
        
        # Log du modèle
        mlflow.sklearn.log_model(model, "model")
        
        # Sauvegarde locale du modèle
        model_path = "./models/model.joblib"
        os.makedirs("./models", exist_ok=True)
        joblib.dump(model, model_path)
        print(f"💾 Modèle sauvegardé à {model_path}")
        
        # Affichage des résultats
        print("\n✨ Résultats de l'entraînement:")
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")
        
        # Log du chemin du modèle
        mlflow.log_artifact(model_path)
        
        print(f"\n🔗 Run ID: {mlflow.active_run().info.run_id}")
        print(f"📊 MLflow UI: {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    main()