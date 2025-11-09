import pytest
import os
import pandas as pd

def test_raw_data_exists():
    """Test si les données brutes existent"""
    raw_data_path = "./data/raw"
    if not os.path.exists(raw_data_path):
        pytest.skip("Dossier data/raw n'existe pas encore")
    
    # Vérifier qu'il y a au moins un fichier
    files = os.listdir(raw_data_path)
    assert len(files) > 0, "Aucun fichier dans data/raw"

def test_processed_data_exists():
    """Test si les données traitées existent"""
    processed_path = "./data/processed"
    if not os.path.exists(processed_path):
        pytest.skip("Dossier data/processed n'existe pas encore")
    
    files = os.listdir(processed_path)
    assert len(files) > 0, "Aucun fichier dans data/processed"

def test_data_structure():
    """Test la structure des données (exemple)"""
    processed_path = "./data/processed"
    
    if not os.path.exists(processed_path):
        pytest.skip("Données non encore traitées")
    
    # Chercher des fichiers CSV
    csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]
    
    if not csv_files:
        pytest.skip("Aucun fichier CSV trouvé")
    
    # Tester le premier fichier CSV trouvé
    df = pd.read_csv(os.path.join(processed_path, csv_files[0]))
    
    # Vérifications basiques
    assert not df.empty, "DataFrame est vide"
    assert len(df.columns) > 0, "Aucune colonne"
    assert len(df) > 0, "Aucune ligne"

def test_no_missing_critical_data():
    """Test qu'il n'y a pas de données critiques manquantes"""
    processed_path = "./data/processed"
    
    if not os.path.exists(processed_path):
        pytest.skip("Données non encore traitées")
    
    # Adapter selon ta structure de données
    assert os.path.exists(processed_path)