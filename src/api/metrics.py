"""
Métriques Prometheus pour l'API
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# Métriques Prometheus
prediction_counter = Counter(
    'mlops_predictions_total',
    'Nombre total de prédictions',
    ['model_version', 'prediction_class']
)

prediction_duration = Histogram(
    'mlops_prediction_duration_seconds',
    'Durée des prédictions en secondes'
)

model_accuracy = Gauge(
    'mlops_model_accuracy',
    'Accuracy actuelle du modèle'
)

drift_detected = Gauge(
    'mlops_drift_detected',
    'Drift détecté (0=non, 1=oui)',
    ['feature_name']
)

api_requests = Counter(
    'mlops_api_requests_total',
    'Nombre total de requêtes API',
    ['endpoint', 'method', 'status']
)

def track_prediction(prediction_class, duration, model_version='v1.0'):
    """Enregistre une prédiction dans Prometheus"""
    prediction_counter.labels(
        model_version=model_version,
        prediction_class=str(prediction_class)
    ).inc()
    
    prediction_duration.observe(duration)

def update_model_metrics(accuracy):
    """Met à jour les métriques du modèle"""
    model_accuracy.set(accuracy)

def set_drift_status(feature_name, is_drifted):
    """Met à jour le statut de drift"""
    drift_detected.labels(feature_name=feature_name).set(1 if is_drifted else 0)

def track_api_request(endpoint, method, status):
    """Enregistre une requête API"""
    api_requests.labels(
        endpoint=endpoint,
        method=method,
        status=str(status)
    ).inc()

def get_metrics():
    """Retourne les métriques Prometheus"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )