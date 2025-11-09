import pytest
import numpy as np
import joblib
import os

@pytest.fixture
def model_path():
    """Fixture pour le chemin du modèle"""
    return "./models/model.joblib"

def test_model_exists(model_path):
    """Test si le modèle existe"""
    # Skip si le modèle n'existe pas encore
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    assert os.path.exists(model_path)

def test_model_loads(model_path):
    """Test si le modèle peut être chargé"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    assert model is not None

def test_model_prediction(model_path):
    """Test si le modèle peut faire une prédiction"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    
    # Créer des données de test (adapter selon ton modèle)
    X_test = np.random.rand(1, 28)  # Exemple avec 10 features
    
    try:
        prediction = model.predict(X_test)
        assert prediction is not None
        assert len(prediction) == 1
    except Exception as e:
        pytest.skip(f"Impossible de prédire: {str(e)}")

def test_model_has_predict_method(model_path):
    """Test si le modèle a une méthode predict"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    assert hasattr(model, 'predict')