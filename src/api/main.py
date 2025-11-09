from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
import joblib
import numpy as np
import os
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

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

api_requests = Counter(
    'mlops_api_requests_total',
    'Nombre total de requêtes API',
    ['endpoint', 'method', 'status']
)

# Chargement du modèle au démarrage avec lifespan
MODEL_PATH = "./models/model.joblib"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
        # Initialiser l'accuracy (exemple)
        model_accuracy.set(0.85)
    else:
        print(f"⚠️ Modèle non trouvé à {MODEL_PATH}")
    yield
    # Shutdown (cleanup si nécessaire)
    model = None

app = FastAPI(
    title="Road Accident Prediction API",
    description="API pour prédire la gravité des accidents de la route",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware pour tracker les requêtes
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    # Tracker la requête
    api_requests.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    return response

# Modèle Pydantic pour la validation des données d'entrée
class PredictionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "feature1": 0.5,
                    "feature2": 1.2,
                    "feature3": 0.8
                }
            }
        }
    )
    
    features: Dict[str, Any]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.get("/")
async def root():
    return {
        "message": "Road Accident Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/metrics")
async def metrics():
    """Endpoint Prometheus pour les métriques"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Veuillez entraîner un modèle d'abord."
        )
    
    # Mesurer le temps de prédiction
    start_time = time.time()
    
    try:
        # Conversion des features en array
        features_array = np.array([list(input_data.features.values())])
        
        # Prédiction
        prediction = model.predict(features_array)[0]
        
        # Probabilités (si le modèle le supporte)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            probability = float(probabilities[prediction])
        else:
            probability = 1.0
        
        # Mesurer la durée
        duration = time.time() - start_time
        
        # Tracker dans Prometheus
        prediction_counter.labels(
            model_version='v1.0',
            prediction_class=str(prediction)
        ).inc()
        
        prediction_duration.observe(duration)
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=probability
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")

@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH
    }