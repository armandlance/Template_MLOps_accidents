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
    schedule='@weekly',  # ✅ CHANGÉ: schedule au lieu de schedule_interval
    catchup=False,
    tags=['mlops', 'training', 'road-accident']
)

# # Task 1: Import des données brutes
# import_data = BashOperator(
#     task_id='import_raw_data',
#     bash_command='cd /app && python ./src/data/import_raw_data.py',
#     dag=dag
# )

# # Task 2: Prétraitement des données
# preprocess_data = BashOperator(
#     task_id='preprocess_data',
#     bash_command='cd /app && python ./src/data/make_dataset.py',
#     dag=dag
# )

# # Task 3: Construction des features
# build_features = BashOperator(
#     task_id='build_features',
#     bash_command='cd /app && python ./src/features/build_features.py',
#     dag=dag
# )

# Task Data X_Train: Existence des données
def data_valid(**context):
    """Vérifie que les données prétraitées existent"""
    data_path_X_train = '/app/data/processed/X_train.csv'
    data_path_X_test = '/app/data/processed/X_test.csv'
    data_path_y_train = '/app/data/processed/y_train.csv'
    data_path_y_test = '/app/data/processed/y_test.csv'

    
    if (not os.path.exists(data_path_X_train) or not os.path.exists(data_path_X_test) or not os.path.exists(data_path_y_train) or not os.path.exists(data_path_y_test)) :
        raise ValueError("Les données prétraitées n'existent pas!")
    
    file_size_X_train = os.path.getsize(data_path_X_train)
    file_size_X_test = os.path.getsize(data_path_X_test)
    file_size_y_train = os.path.getsize(data_path_y_train)
    file_size_y_test = os.path.getsize(data_path_y_test)
    print(f"✅ Données trouvées: {data_path_X_train} ({file_size_X_train} bytes)")
    print(f"✅ Données trouvées: {data_path_X_test} ({file_size_X_test} bytes)")
    print(f"✅ Données trouvées: {data_path_y_train} ({file_size_y_train} bytes)")
    print(f"✅ Données trouvées: {data_path_y_test} ({file_size_y_test} bytes)")    
    
    return True

check_data = PythonOperator(
    task_id='data_valid',
    python_callable=data_valid,
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
# Task 6: Notification de fin d'entraînement
def notify_completion(**context):
    """Notifie la fin de l'entraînement du modèle"""
    print("✅ Entraînement du modèle terminé avec succès!")


notify = PythonOperator(
    task_id='notify_completion',  
    python_callable=notify_completion,
    dag=dag
)

# Définition du workflow
#import_data >> preprocess_data >> build_features >> train_model >> validate >> notify
check_data >> train_model >> validate >> notify