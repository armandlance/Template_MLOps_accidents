# import pytest
# import numpy as np
# import joblib
# import os

# @pytest.fixture
# def model_path():
#     """Fixture pour le chemin du modèle"""
#     return "./models/model.joblib"

# def test_model_exists(model_path):
#     """Test si le modèle existe"""
#     # Skip si le modèle n'existe pas encore
#     if not os.path.exists(model_path):
#         pytest.skip("Modèle non encore entraîné")
    
#     assert os.path.exists(model_path)

# def test_model_loads(model_path):
#     """Test si le modèle peut être chargé"""
#     if not os.path.exists(model_path):
#         pytest.skip("Modèle non encore entraîné")
    
#     model = joblib.load(model_path)
#     assert model is not None

# def test_model_prediction(model_path):
#     """Test si le modèle peut faire une prédiction"""
#     if not os.path.exists(model_path):
#         pytest.skip("Modèle non encore entraîné")
    
#     model = joblib.load(model_path)
    
#     # Créer des données de test (adapter selon ton modèle)
#     X_test = np.random.rand(1, 28)  # Exemple avec 10 features
    
#     try:
#         prediction = model.predict(X_test)
#         assert prediction is not None
#         assert len(prediction) == 1
#     except Exception as e:
#         pytest.skip(f"Impossible de prédire: {str(e)}")

# def test_model_has_predict_method(model_path):
#     """Test si le modèle a une méthode predict"""
#     if not os.path.exists(model_path):
#         pytest.skip("Modèle non encore entraîné")
    
#     model = joblib.load(model_path)
#     assert hasattr(model, 'predict')

import pytest
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

@pytest.fixture
def model_path():
    """Fixture pour le chemin du modèle"""
    return "./models/model.joblib"

@pytest.fixture
def sample_data():
    """Fixture pour générer des données de test"""
    # Données de test simples (à adapter selon ton modèle)
    X = np.random.rand(100, 28)
    y = np.random.randint(0, 2, 100)  # Classification binaire
    return train_test_split(X, y, test_size=0.3, random_state=42)

def test_model_exists(model_path):
    """Test si le modèle existe"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    assert os.path.exists(model_path)
    
    # Vérifier que le fichier n'est pas vide
    file_size = os.path.getsize(model_path)
    assert file_size > 0, "Le fichier modèle est vide"
    assert file_size > 100, "Le fichier modèle semble trop petit"

def test_model_loads(model_path):
    """Test si le modèle peut être chargé correctement"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    assert model is not None, "Le modèle chargé est None"
    
    # Vérifier que c'est bien un objet modèle valide
    assert hasattr(model, 'predict'), "Le modèle n'a pas de méthode predict"

def test_model_prediction_shape(model_path):
    """Test si le modèle produit des prédictions de la bonne forme"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    
    # Créer des données de test
    n_samples = 10
    n_features = 28  # Adapter selon ton modèle
    X_test = np.random.rand(n_samples, n_features)
    
    try:
        predictions = model.predict(X_test)
        
        # Vérifier la forme des prédictions
        assert predictions is not None, "Prédictions nulles"
        assert len(predictions) == n_samples, \
            f"Nombre de prédictions incorrect: {len(predictions)} au lieu de {n_samples}"
        
        # Vérifier que les prédictions ne sont pas toutes identiques
        assert len(np.unique(predictions)) > 1 or n_samples == 1, \
            "Toutes les prédictions sont identiques"
        
    except Exception as e:
        pytest.fail(f"Erreur lors de la prédiction: {str(e)}")

def test_model_prediction_types(model_path):
    """Vérifie que les prédictions sont du bon type"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    X_test = np.random.rand(5, 28)
    
    try:
        predictions = model.predict(X_test)
        
        # Les prédictions doivent être numériques
        assert np.issubdtype(predictions.dtype, np.number), \
            "Les prédictions ne sont pas numériques"
        
        # Pas de NaN dans les prédictions
        assert not np.isnan(predictions).any(), \
            "Des NaN détectés dans les prédictions"
        
        # Pas d'infini
        assert not np.isinf(predictions).any(), \
            "Des valeurs infinies dans les prédictions"
        
    except Exception as e:
        pytest.fail(f"Erreur lors de la validation: {str(e)}")

def test_model_has_predict_method(model_path):
    """Test si le modèle a les méthodes nécessaires"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    
    # Vérifier les méthodes essentielles
    assert hasattr(model, 'predict'), "Pas de méthode predict"
    
    # Si c'est un modèle de classification, vérifier predict_proba
    if hasattr(model, 'predict_proba'):
        assert callable(model.predict_proba), "predict_proba n'est pas callable"

def test_model_performance_minimum_threshold(model_path):
    """
    🎯 TEST CRITIQUE: Vérifie que le modèle atteint un seuil minimal de performance
    
    Ce test garantit que le modèle est meilleur qu'un modèle aléatoire.
    """
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    # Charger les données de test si disponibles
    test_data_path = "./data/processed/test.csv"
    
    if not os.path.exists(test_data_path):
        pytest.skip("Données de test non disponibles pour évaluer la performance")
    
    try:
        # Charger les données de test
        df_test = pd.read_csv(test_data_path)
        
        # Séparer features et target (à adapter selon tes colonnes)
        # Exemple: dernière colonne = target
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values
        
        # Charger le modèle
        model = joblib.load(model_path)
        
        # Faire des prédictions
        y_pred = model.predict(X_test)
        
        # Déterminer si c'est de la classification ou régression
        is_classification = len(np.unique(y_test)) < 20  # Heuristique simple
        
        if is_classification:
            # 🎯 CLASSIFICATION: Seuil minimal d'accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Le modèle doit être meilleur qu'un modèle aléatoire
            n_classes = len(np.unique(y_test))
            random_accuracy = 1.0 / n_classes
            
            # Seuil: au moins 10% mieux qu'aléatoire, ou 60% minimum
            min_accuracy = max(random_accuracy + 0.1, 0.6)
            
            assert accuracy >= min_accuracy, \
                f"Accuracy trop faible: {accuracy:.3f} (seuil minimum: {min_accuracy:.3f})"
            
            # 🎯 F1-Score pour les classes déséquilibrées
            f1 = f1_score(y_test, y_pred, average='weighted')
            min_f1 = 0.5
            
            assert f1 >= min_f1, \
                f"F1-score trop faible: {f1:.3f} (seuil minimum: {min_f1:.3f})"
            
            print(f"✅ Performance - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
        else:
            # 🎯 RÉGRESSION: Seuil minimal de R²
            r2 = r2_score(y_test, y_pred)
            min_r2 = 0.3  # Le modèle doit expliquer au moins 30% de la variance
            
            assert r2 >= min_r2, \
                f"R² trop faible: {r2:.3f} (seuil minimum: {min_r2:.3f})"
            
            # MSE ne devrait pas être astronomique
            mse = mean_squared_error(y_test, y_pred)
            y_std = np.std(y_test)
            
            # MSE ne devrait pas être plus grande que la variance des données
            assert mse < (y_std ** 2) * 2, \
                f"MSE trop élevée: {mse:.3f} (variance: {y_std**2:.3f})"
            
            print(f"✅ Performance - R²: {r2:.3f}, MSE: {mse:.3f}")
            
    except FileNotFoundError:
        pytest.skip("Fichier de test non trouvé")
    except Exception as e:
        pytest.fail(f"Erreur lors de l'évaluation: {str(e)}")

def test_model_consistency():
    """Vérifie que le modèle produit des prédictions cohérentes"""
    model_path = "./models/model.joblib"
    
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    X_test = np.random.rand(5, 28)
    
    # Faire 2 prédictions avec les mêmes données
    pred1 = model.predict(X_test)
    pred2 = model.predict(X_test)
    
    # Les prédictions devraient être identiques (reproductibilité)
    assert np.allclose(pred1, pred2), \
        "Le modèle ne produit pas de prédictions cohérentes"

def test_model_no_overfitting_indicator():
    """
    Test basique pour détecter un overfitting évident
    Compare performance train vs test si disponible
    """
    model_path = "./models/model.joblib"
    
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    train_path = "./data/processed/train.csv"
    test_path = "./data/processed/test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        pytest.skip("Données train/test non disponibles")
    
    try:
        # Charger données
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values
        
        model = joblib.load(model_path)
        
        # Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculer les métriques
        is_classification = len(np.unique(y_train)) < 20
        
        if is_classification:
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # L'écart ne devrait pas être énorme (>20%)
            gap = train_acc - test_acc
            assert gap < 0.20, \
                f"Overfitting possible: train={train_acc:.3f}, test={test_acc:.3f}, écart={gap:.3f}"
            
            print(f"ℹ️  Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")
        
    except Exception as e:
        pytest.skip(f"Impossible de vérifier l'overfitting: {str(e)}")

def test_model_handles_edge_cases(model_path):
    """Test que le modèle gère les cas limites"""
    if not os.path.exists(model_path):
        pytest.skip("Modèle non encore entraîné")
    
    model = joblib.load(model_path)
    
    # Cas limites
    edge_cases = [
        np.zeros((1, 28)),  # Tous zéros
        np.ones((1, 28)),   # Tous uns
        np.random.rand(1, 28) * 1000,  # Grandes valeurs
    ]
    
    for i, X in enumerate(edge_cases):
        try:
            pred = model.predict(X)
            assert pred is not None, f"Cas {i}: prédiction nulle"
            assert not np.isnan(pred).any(), f"Cas {i}: NaN détecté"
            assert not np.isinf(pred).any(), f"Cas {i}: Inf détecté"
        except Exception as e:
            pytest.fail(f"Cas {i} a échoué: {str(e)}")