import pandas as pd
import numpy as np
import os

def create_mock_data():
    """Crée des fichiers de données de référence et de données récentes (avec un léger drift) pour le test."""
    
    # Créer le répertoire si non existant
    os.makedirs("./data/processed", exist_ok=True)
    
    N_SAMPLES = 1000
    
    # Données de Référence (Entraînement)
    ref_data = pd.DataFrame({
        'feature_A': np.random.normal(loc=10, scale=2, size=N_SAMPLES),
        'feature_B': np.random.normal(loc=50, scale=5, size=N_SAMPLES),
        'feature_C': np.random.randint(0, 5, size=N_SAMPLES),
        'target': np.random.randint(0, 2, size=N_SAMPLES)
    })
    ref_data['feature_C'] = ref_data['feature_C'].astype('category')
    
    # Données Récentes (avec Data Drift volontaire sur feature_A)
    new_data = pd.DataFrame({
        # Drift: moyenne passe de 10 à 12
        'feature_A': np.random.normal(loc=12, scale=2.5, size=N_SAMPLES),
        # Pas de Drift
        'feature_B': np.random.normal(loc=50, scale=5, size=N_SAMPLES),
        'feature_C': np.random.randint(0, 5, size=N_SAMPLES),
        'target': np.random.randint(0, 2, size=N_SAMPLES)
    })
    new_data['feature_C'] = new_data['feature_C'].astype('category')
    
    # Sauvegarde des fichiers
    ref_data.to_csv("./data/processed/train_data.csv", index=False)
    new_data.to_csv("./data/processed/recent_data.csv", index=False)
    
    print("✅ Fichiers de données mockés créés dans ./data/processed/")

if __name__ == "__main__":
    create_mock_data()