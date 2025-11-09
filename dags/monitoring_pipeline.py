"""
DAG Airflow pour le monitoring automatique et détection de drift
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Configuration
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['your-email@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG de monitoring
dag = DAG(
    'monitoring_drift_detection',
    default_args=default_args,
    description='Pipeline de monitoring et détection de drift',
    schedule_interval='@daily',  # Tous les jours
    catchup=False,
    tags=['mlops', 'monitoring', 'drift']
)

# Task 1: Exécuter le monitoring
run_monitoring = BashOperator(
    task_id='run_drift_detection',
    bash_command='cd /app && python ./src/monitoring/monitor.py',
    dag=dag
)

# Task 2: Vérifier si drift détecté
def check_drift_alerts(**context):
    """Vérifie si des alertes de drift ont été générées"""
    import os
    import json
    
    alert_file = '/app/logs/drift_alerts.json'
    
    if not os.path.exists(alert_file):
        print("✅ Aucune alerte de drift")
        return False
    
    # Lire les alertes récentes (dernières 24h)
    from datetime import datetime, timedelta
    
    with open(alert_file, 'r') as f:
        alerts = [json.loads(line) for line in f]
    
    recent_alerts = [
        alert for alert in alerts
        if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(days=1)
    ]
    
    if recent_alerts:
        print(f"🚨 {len(recent_alerts)} alerte(s) détectée(s) dans les dernières 24h")
        return True
    else:
        print("✅ Pas d'alerte récente")
        return False

check_drift = PythonOperator(
    task_id='check_drift_alerts',
    python_callable=check_drift_alerts,
    dag=dag
)

# Task 3: Déclencher réentraînement si drift détecté
def trigger_retraining_if_needed(**context):
    """Déclenche le réentraînement si drift détecté"""
    ti = context['ti']
    drift_detected = ti.xcom_pull(task_ids='check_drift_alerts')
    
    if drift_detected:
        print("🔄 Drift détecté! Déclenchement du réentraînement...")
        
        # Déclencher le DAG de training
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator
        
        # Note: Cette partie nécessite d'être adaptée selon ta config
        print("⚠️ Réentraînement à déclencher manuellement ou via API Airflow")
        
        return "retraining_needed"
    else:
        print("✅ Pas de réentraînement nécessaire")
        return "no_action"

trigger_retraining = PythonOperator(
    task_id='trigger_retraining_if_needed',
    python_callable=trigger_retraining_if_needed,
    dag=dag
)

# Task 4: Générer rapport de monitoring
def generate_monitoring_report(**context):
    """Génère un rapport quotidien de monitoring"""
    import json
    from datetime import datetime
    
    report = {
        'date': datetime.now().isoformat(),
        'monitoring_executed': True,
        'drift_detection_run': True,
        'alerts_checked': True
    }
    
    # Sauvegarder le rapport
    with open('/app/logs/monitoring_daily_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📊 Rapport de monitoring généré")

generate_report = PythonOperator(
    task_id='generate_monitoring_report',
    python_callable=generate_monitoring_report,
    dag=dag
)

# Workflow
run_monitoring >> check_drift >> trigger_retraining >> generate_report