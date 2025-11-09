import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

client = TestClient(app)

def test_root():
    """Test de l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "running"

def test_health_check():
    """Test du health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_predict_without_model():
    """Test de prédiction (peut échouer si modèle non chargé)"""
    test_input = {
        "features": {
            "feature1": 0.5,
            "feature2": 1.2,
            "feature3": 0.8
        }
    }
    response = client.post("/predict", json=test_input)
    # Accepte soit 200 (si modèle chargé) soit 503 (si pas de modèle)
    assert response.status_code in [200, 503]

def test_model_info():
    """Test de l'endpoint model-info"""
    response = client.get("/model-info")
    # Accepte soit 200 (si modèle chargé) soit 503 (si pas de modèle)
    assert response.status_code in [200, 503]

def test_predict_invalid_input():
    """Test avec des données invalides"""
    test_input = {"invalid": "data"}
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error