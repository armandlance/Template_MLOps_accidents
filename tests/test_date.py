# import pytest
# import os
# import pandas as pd

# def test_raw_data_exists():
#     """Test si les données brutes existent"""
#     raw_data_path = "./data/raw"
#     if not os.path.exists(raw_data_path):
#         pytest.skip("Dossier data/raw n'existe pas encore")
    
#     # Vérifier qu'il y a au moins un fichier
#     files = os.listdir(raw_data_path)
#     assert len(files) > 0, "Aucun fichier dans data/raw"

# def test_processed_data_exists():
#     """Test si les données traitées existent"""
#     processed_path = "./data/processed"
#     if not os.path.exists(processed_path):
#         pytest.skip("Dossier data/processed n'existe pas encore")
    
#     files = os.listdir(processed_path)
#     assert len(files) > 0, "Aucun fichier dans data/processed"

# def test_data_structure():
#     """Test la structure des données (exemple)"""
#     processed_path = "./data/processed"
    
#     if not os.path.exists(processed_path):
#         pytest.skip("Données non encore traitées")
    
#     # Chercher des fichiers CSV
#     csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]
    
#     if not csv_files:
#         pytest.skip("Aucun fichier CSV trouvé")
    
#     # Tester le premier fichier CSV trouvé
#     df = pd.read_csv(os.path.join(processed_path, csv_files[0]))
    
#     # Vérifications basiques
#     assert not df.empty, "DataFrame est vide"
#     assert len(df.columns) > 0, "Aucune colonne"
#     assert len(df) > 0, "Aucune ligne"

# def test_no_missing_critical_data():
#     """Test qu'il n'y a pas de données critiques manquantes"""
#     processed_path = "./data/processed"
    
#     if not os.path.exists(processed_path):
#         pytest.skip("Données non encore traitées")
    
#     # Adapter selon ta structure de données
#     assert os.path.exists(processed_path)
import pytest
import os
import pandas as pd
import numpy as np

def test_raw_data_exists():
    """Test si les données brutes existent"""
    raw_data_path = "./data/raw"
    if not os.path.exists(raw_data_path):
        pytest.skip("Dossier data/raw n'existe pas encore")
    
    # Vérifier qu'il y a au moins un fichier
    files = os.listdir(raw_data_path)
    assert len(files) > 0, "Aucun fichier dans data/raw"
    
    # Vérifier que les fichiers ont une extension connue
    valid_extensions = ['.csv', '.parquet', '.json', '.xlsx']
    file_extensions = [os.path.splitext(f)[1] for f in files]
    assert any(ext in valid_extensions for ext in file_extensions), \
        "Aucun fichier de données valide trouvé"

def test_processed_data_exists():
    """Test si les données traitées existent avec validation"""
    processed_path = "./data/processed"
    if not os.path.exists(processed_path):
        pytest.skip("Dossier data/processed n'existe pas encore")
    
    files = os.listdir(processed_path)
    assert len(files) > 0, "Aucun fichier dans data/processed"
    
    # Vérifier qu'il y a des fichiers de données
    data_files = [f for f in files if f.endswith(('.csv', '.parquet', '.json'))]
    assert len(data_files) > 0, "Aucun fichier de données dans processed"

def test_data_structure():
    """Test la structure des données de manière détaillée"""
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
    
    # Vérifications plus précises
    assert len(df.columns) >= 2, "Il devrait y avoir au moins 2 colonnes"
    assert len(df) >= 10, "Il devrait y avoir au moins 10 lignes de données"

def test_data_types_are_valid():
    """Vérifie que les types de données sont cohérents"""
    processed_path = "./data/processed"
    
    if not os.path.exists(processed_path):
        pytest.skip("Données non encore traitées")
    
    csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]
    if not csv_files:
        pytest.skip("Aucun fichier CSV trouvé")
    
    df = pd.read_csv(os.path.join(processed_path, csv_files[0]))
    
    # Vérifier qu'il n'y a pas que des types 'object' (chaînes)
    # Ce qui pourrait indiquer un problème de parsing
    type_counts = df.dtypes.value_counts()
    assert len(type_counts) > 0, "Aucun type détecté"
    
    # Au moins une colonne numérique devrait exister
    numeric_types = ['int64', 'float64', 'int32', 'float32']
    has_numeric = any(dtype in df.dtypes.values.astype(str) for dtype in numeric_types)
    assert has_numeric, "Aucune colonne numérique trouvée"

def test_no_missing_critical_data():
    """Test qu'il n'y a pas de données critiques manquantes"""
    processed_path = "./data/processed"
    
    if not os.path.exists(processed_path):
        pytest.skip("Données non encore traitées")
    
    csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]
    if not csv_files:
        pytest.skip("Aucun fichier CSV trouvé")
    
    df = pd.read_csv(os.path.join(processed_path, csv_files[0]))
    
    # Vérifier qu'il n'y a pas 100% de valeurs manquantes dans aucune colonne
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df)
        assert missing_pct < 1.0, f"La colonne {col} est entièrement vide"
        
        # Avertissement si plus de 50% de valeurs manquantes
        if missing_pct > 0.5:
            print(f"⚠️  Avertissement: {col} a {missing_pct*100:.1f}% de valeurs manquantes")

def test_data_quality_no_duplicates():
    """Vérifie qu'il n'y a pas trop de lignes dupliquées"""
    processed_path = "./data/processed"
    
    if not os.path.exists(processed_path):
        pytest.skip("Données non encore traitées")
    
    csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]
    if not csv_files:
        pytest.skip("Aucun fichier CSV trouvé")
    
    df = pd.read_csv(os.path.join(processed_path, csv_files[0]))
    
    # Calculer le pourcentage de doublons
    duplicate_pct = df.duplicated().sum() / len(df)
    
    # Pas plus de 10% de doublons
    assert duplicate_pct < 0.1, \
        f"Trop de lignes dupliquées: {duplicate_pct*100:.1f}%"

def test_data_has_reasonable_values():
    """Vérifie que les valeurs numériques sont dans des plages raisonnables"""
    processed_path = "./data/processed"
    
    if not os.path.exists(processed_path):
        pytest.skip("Données non encore traitées")
    
    csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]
    if not csv_files:
        pytest.skip("Aucun fichier CSV trouvé")
    
    df = pd.read_csv(os.path.join(processed_path, csv_files[0]))
    
    # Pour chaque colonne numérique
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Pas de valeurs infinies
        assert not np.isinf(df[col]).any(), \
            f"Valeurs infinies trouvées dans {col}"
        
        # Vérifier qu'il y a de la variance (pas toutes les mêmes valeurs)
        if df[col].notna().sum() > 1:
            assert df[col].std() > 0 or df[col].nunique() > 1, \
                f"Aucune variance dans {col}"

def test_train_test_split_exists():
    """Vérifie que les splits train/test existent si applicable"""
    processed_path = "./data/processed"
    
    if not os.path.exists(processed_path):
        pytest.skip("Données non encore traitées")
    
    files = os.listdir(processed_path)
    
    # Chercher des fichiers train/test
    has_train = any('train' in f.lower() for f in files)
    has_test = any('test' in f.lower() for f in files)
    
    # Si un des deux existe, l'autre devrait aussi exister
    if has_train or has_test:
        assert has_train and has_test, \
            "Split train/test incomplet (seulement un des deux présent)"