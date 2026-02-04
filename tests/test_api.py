# import pytest
# from fastapi.testclient import TestClient
# import sys
# import os

# # Ajouter le dossier src au path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.api.main import app

# client = TestClient(app)

# def test_root():
#     """Test de l'endpoint racine"""
#     response = client.get("/")
#     assert response.status_code == 200
#     data = response.json()
#     assert "message" in data
#     assert data["status"] == "running"

# def test_health_check():
#     """Test du health check"""
#     response = client.get("/health")
#     assert response.status_code == 200
#     data = response.json()
#     assert "status" in data
#     assert data["status"] == "healthy"

# def test_predict_without_model():
#     """Test de prédiction (peut échouer si modèle non chargé)"""
#     test_input = {
#         "features": {
#             "feature1": 0.5,
#             "feature2": 1.2,
#             "feature3": 0.8
#         }
#     }
#     response = client.post("/predict", json=test_input)
#     # Accepte soit 200 (si modèle chargé) soit 503 (si pas de modèle)
#     assert response.status_code in [200, 503]

# def test_model_info():
#     """Test de l'endpoint model-info"""
#     response = client.get("/model-info")
#     # Accepte soit 200 (si modèle chargé) soit 503 (si pas de modèle)
#     assert response.status_code in [200, 503]

# def test_predict_invalid_input():
#     """Test avec des données invalides"""
#     test_input = {"invalid": "data"}
#     response = client.post("/predict", json=test_input)
#     assert response.status_code == 422  # Validation error

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

client = TestClient(app)

def test_root():
    """Test de l'endpoint racine avec validation complète"""
    response = client.get("/")
    
    # Validation du status code précis
    assert response.status_code == 200
    
    # Validation de la structure de réponse
    data = response.json()
    assert "message" in data
    assert "status" in data
    
    # Validation du contenu
    assert data["status"] == "running"
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0

def test_health_check():
    """Test du health check avec validation détaillée"""
    response = client.get("/health")
    
    # Status code précis
    assert response.status_code == 200
    
    # Structure de réponse
    data = response.json()
    assert "status" in data
    
    # Contenu vérifié
    assert data["status"] == "healthy"
    assert isinstance(data["status"], str)

def test_predict_with_model():
    """Test de prédiction avec modèle chargé"""
    test_input = {
        "features": {
            "feature1": 0.5,
            "feature2": 1.2,
            "feature3": 0.8
        }
    }
    response = client.post("/predict", json=test_input)
    
    # Si le modèle est chargé
    if response.status_code == 200:
        data = response.json()
        
        # Validation de la structure
        assert "prediction" in data or "result" in data
        
        # Validation du type de prédiction
        prediction_key = "prediction" if "prediction" in data else "result"
        assert data[prediction_key] is not None
        
        # La prédiction doit être un nombre ou une liste de nombres
        if isinstance(data[prediction_key], list):
            assert len(data[prediction_key]) > 0
        else:
            assert isinstance(data[prediction_key], (int, float))
    
    # Si pas de modèle, doit retourner 503 avec message clair
    elif response.status_code == 503:
        data = response.json()
        assert "detail" in data
        assert "model" in data["detail"].lower() or "not" in data["detail"].lower()
    
    else:
        pytest.fail(f"Status code inattendu: {response.status_code}")

def test_predict_validation_input():
    """Test que la validation des inputs fonctionne"""
    # Test avec des données invalides
    test_cases = [
        {"invalid": "data"},  # Structure incorrecte
        {},  # Données vides
        {"features": {}},  # Features vides
    ]
    
    for invalid_input in test_cases:
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422, f"Devrait rejeter: {invalid_input}"
        
        # Vérifier que le message d'erreur est présent
        data = response.json()
        assert "detail" in data

def test_predict_with_valid_structure():
    """Test avec une structure valide mais vérification du comportement"""
    test_input = {
        "features": {
            "feature1": 0.5,
            "feature2": 1.2,
            "feature3": 0.8
        }
    }
    response = client.post("/predict", json=test_input)
    
    # Doit retourner soit 200 (succès) soit 503 (pas de modèle)
    # Mais PAS d'autres codes
    assert response.status_code in [200, 503], \
        f"Code inattendu: {response.status_code}"
    
    # Si 200, vérifier la structure de réponse
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict), "La réponse doit être un dictionnaire"

def test_model_info():
    """Test de l'endpoint model-info avec validation"""
    response = client.get("/model-info")
    
    if response.status_code == 200:
        data = response.json()
        
        # Vérifier les informations du modèle
        expected_keys = ["model_type", "version", "features"] 
        # Au moins une de ces clés devrait être présente
        assert any(key in data for key in expected_keys), \
            "Aucune information de modèle trouvée"
    
    elif response.status_code == 503:
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)
    
    else:
        pytest.fail(f"Status code inattendu: {response.status_code}")

def test_predict_boundary_values():
    """Test avec des valeurs limites"""
    boundary_cases = [
        {"features": {"feature1": 0.0, "feature2": 0.0, "feature3": 0.0}},  # Zéros
        {"features": {"feature1": 1.0, "feature2": 1.0, "feature3": 1.0}},  # Uns
        {"features": {"feature1": -1.0, "feature2": -1.0, "feature3": -1.0}},  # Négatifs
        {"features": {"feature1": 100.0, "feature2": 100.0, "feature3": 100.0}},  # Grands
    ]
    
    for test_case in boundary_cases:
        response = client.post("/predict", json=test_case)
        # Doit soit prédire (200) soit dire que le modèle n'est pas là (503)
        # Ne devrait PAS crash (500)
        assert response.status_code in [200, 503], \
            f"Échec pour {test_case}: code {response.status_code}"

def test_endpoints_return_json():
    """Vérifie que tous les endpoints retournent du JSON valide"""
    endpoints = [
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/model-info"),
    ]
    
    for method, endpoint in endpoints:
        if method == "GET":
            response = client.get(endpoint)
        
        # Devrait toujours retourner du JSON valide
        assert response.headers["content-type"] == "application/json"
        
        # Devrait être parsable
        try:
            data = response.json()
            assert isinstance(data, dict)
        except Exception as e:
            pytest.fail(f"Impossible de parser JSON pour {endpoint}: {e}")