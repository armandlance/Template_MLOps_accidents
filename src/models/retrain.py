"""
Script de réentraînement automatique du modèle
Peut être déclenché par Airflow ou manuellement
"""
import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import json

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("road_accident_retraining")

class ModelRetrainer:
    """Classe pour gérer le réentraînement du modèle"""
    
    def __init__(self, data_path="./data/processed"):
        self.data_path = data_path
        self.model = None
        self.metrics = {}
        self.model_path = "./models/model.joblib"
        self.backup_path = None
        
    def backup_current_model(self):
        """Sauvegarde le modèle actuel avant réentraînement"""
        if os.path.exists(self.model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_path = f"./models/backups/model_backup_{timestamp}.joblib"
            os.makedirs("./models/backups", exist_ok=True)
            
            import shutil
            shutil.copy(self.model_path, self.backup_path)
            print(f"✅ Modèle actuel sauvegardé: {self.backup_path}")
        else:
            print("⚠️ Aucun modèle existant à sauvegarder")
    
    def load_data(self):
        """Charge les données pour le réentraînement"""
        print("📊 Chargement des données...")
        
        # ADAPTER selon ta structure de données
        try:
            X_train = pd.read_csv(os.path.join(self.data_path, "X_train.csv"))
            y_train = pd.read_csv(os.path.join(self.data_path, "y_train.csv"))
            X_test = pd.read_csv(os.path.join(self.data_path, "X_test.csv"))
            y_test = pd.read_csv(os.path.join(self.data_path, "y_test.csv"))
            
            return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données: {str(e)}")
            raise
    
    def train_model(self, X_train, y_train, params=None):
        """Entraîne un nouveau modèle"""
        print("🎯 Entraînement du nouveau modèle...")
        
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Évalue le nouveau modèle"""
        print("📈 Évaluation du modèle...")
        
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return self.metrics
    
    def compare_with_previous(self):
        """Compare les performances avec le modèle précédent"""
        print("🔍 Comparaison avec le modèle précédent...")
        
        if self.backup_path and os.path.exists(self.backup_path):
            # Charger l'ancien modèle
            old_model = joblib.load(self.backup_path)
            
            # Charger les données de test
            X_test = pd.read_csv(os.path.join(self.data_path, "X_test.csv"))
            y_test = pd.read_csv(os.path.join(self.data_path, "y_test.csv")).values.ravel()
            
            # Évaluer l'ancien modèle
            y_pred_old = old_model.predict(X_test)
            old_accuracy = accuracy_score(y_test, y_pred_old)
            
            new_accuracy = self.metrics['accuracy']
            
            improvement = new_accuracy - old_accuracy
            
            print(f"   Ancien modèle: {old_accuracy:.4f}")
            print(f"   Nouveau modèle: {new_accuracy:.4f}")
            print(f"   Amélioration: {improvement:+.4f}")
            
            return improvement > 0  # True si amélioration
        else:
            print("⚠️ Pas de modèle précédent pour comparaison")
            return True  # Accepter le nouveau modèle
    
    def deploy_model(self):
        """Déploie le nouveau modèle"""
        print("🚀 Déploiement du nouveau modèle...")
        
        joblib.dump(self.model, self.model_path)
        print(f"✅ Nouveau modèle déployé: {self.model_path}")
    
    def log_to_mlflow(self, params):
        """Log le réentraînement dans MLflow"""
        with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log paramètres
            mlflow.log_params(params)
            
            # Log métriques
            mlflow.log_metrics(self.metrics)
            
            # Log modèle
            mlflow.sklearn.log_model(self.model, "model")
            
            # Log info de réentraînement
            mlflow.log_param("retraining_date", datetime.now().isoformat())
            mlflow.log_param("backup_path", self.backup_path or "none")
            
            print(f"✅ Réentraînement loggé dans MLflow")
    
    def retrain(self, force=False):
        """
        Pipeline complet de réentraînement
        
        Args:
            force: Si True, déploie le modèle même s'il n'y a pas d'amélioration
        """
        print("\n" + "="*50)
        print("🔄 DÉBUT DU RÉENTRAÎNEMENT")
        print("="*50 + "\n")
        
        try:
            # 1. Backup du modèle actuel
            self.backup_current_model()
            
            # 2. Chargement des données
            X_train, X_test, y_train, y_test = self.load_data()
            
            # 3. Entraînement
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
            self.train_model(X_train, y_train, params)
            
            # 4. Évaluation
            self.evaluate_model(X_test, y_test)
            
            print("\n📊 Résultats:")
            for metric_name, metric_value in self.metrics.items():
                print(f"   {metric_name}: {metric_value:.4f}")
            
            # 5. Comparaison
            is_better = self.compare_with_previous()
            
            # 6. Décision de déploiement
            if is_better or force:
                self.deploy_model()
                self.log_to_mlflow(params)
                
                print("\n✅ RÉENTRAÎNEMENT RÉUSSI!")
                
                # Créer un rapport
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success',
                    'metrics': self.metrics,
                    'deployed': True
                }
            else:
                print("\n⚠️ Le nouveau modèle n'est pas meilleur. Déploiement annulé.")
                
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'aborted',
                    'metrics': self.metrics,
                    'deployed': False,
                    'reason': 'No improvement'
                }
            
            # Sauvegarder le rapport
            os.makedirs("./logs", exist_ok=True)
            with open('./logs/retraining_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print("\n" + "="*50)
            print("🏁 FIN DU RÉENTRAÎNEMENT")
            print("="*50 + "\n")
            
            return report
            
        except Exception as e:
            print(f"\n❌ ERREUR: {str(e)}")
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            
            with open('./logs/retraining_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            raise

def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Réentraîner le modèle')
    parser.add_argument('--force', action='store_true', 
                       help='Forcer le déploiement même sans amélioration')
    
    args = parser.parse_args()
    
    # Lancer le réentraînement
    retrainer = ModelRetrainer()
    report = retrainer.retrain(force=args.force)
    
    return report

if __name__ == "__main__":
    main()