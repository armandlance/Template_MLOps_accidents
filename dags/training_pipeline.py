from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Configuration par défaut du DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Définition du DAG
dag = DAG(
    'road_accident_training_pipeline',
    default_args=default_args,
    description='Pipeline complet d\'entraînement du modèle',
    schedule_interval='@weekly',  # Réentraînement hebdomadaire
    catchup=False,
    tags=['mlops', 'training', 'road-accident']
)

# Task 1: Import des données brutes
import_data = BashOperator(
    task_id='import_raw_data',
    bash_command='cd /app && python ./src/data/import_raw_data.py',
    dag=dag
)

# Task 2: Prétraitement des données
preprocess_data = BashOperator(
    task_id='preprocess_data',
    bash_command='cd /app && python ./src/data/make_dataset.py',
    dag=dag
)

# Task 3: Construction des features
build_features = BashOperator(
    task_id='build_features',
    bash_command='cd /app && python ./src/features/build_features.py',
    dag=dag
)

# Task 4: Entraînement du modèle avec MLflow
train_model = BashOperator(
    task_id='train_model',
    bash_command='cd /app && python ./src/models/train_model_mlflow.py',
    dag=dag
)

# Task 5: Validation du modèle
def validate_model(**context):
    """Valide que le modèle a été créé correctement"""
    import os
    model_path = '/app/models/model.joblib'
    
    if not os.path.exists(model_path):
        raise ValueError("Le modèle n'a pas été créé!")
    
    file_size = os.path.getsize(model_path)
    print(f"✅ Modèle validé: {model_path} ({file_size} bytes)")
    
    return True

validate = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

# Task 6: Notification (optionnel)
def notify_completion(**context):
    """Notifie la fin du pipeline"""
    print("🎉 Pipeline d'entraînement terminé avec succès!")
    print(f"Run ID: {context['run_id']}")
    print(f"Execution date: {context['execution_date']}")

notify = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag
)

# Définition du workflow
import_data >> preprocess_data >> build_features >> train_model >> validate >> notify