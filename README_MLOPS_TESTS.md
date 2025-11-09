# Projet MLOps

Ce document présente l’ensemble des tests permettant de valider le bon fonctionnement de l’environnement, de la pipeline de data, du modèle, du déploiement API, ainsi que du monitoring.

## 🔧 PHASE 1 : Environment Setup

### Test 1.1 : Docker
```powershell
docker --version
docker ps
docker-compose --version
docker-compose ps
```

Si Manquant 
```
docker-compose build --no-cache
docker-compose up -d
```

### Test 1.2 : Git / GitHub
```powershell
git --version
git status
git remote -v
```

### Test 1.3 : Python dans les containers
```powershell
docker-compose exec api python --version
docker-compose exec api pip list | Select-String "fastapi|mlflow|scikit-learn|pandas"
```

### Test 1.4 : Requirements / Poetry
```powershell
Get-Content requirements_mlops.txt
docker-compose exec api pip check
```

## 📊 PHASE 2 : Data Management & Training

### Test 2.1 : DVC
```powershell
dvc version
ls .dvc/
ls *.dvc
dvc status
```

Si status pas bon faire les commanddes suivantes :
```powershell
dvc add data/raw
dvc add data/processed
git add -f data/raw.dvc data/.gitignore
git add -f data/processed.dvc data/.gitignore
dvc status
```


### Test 2.2 : Données
```powershell
ls data/raw/
(Get-ChildItem data/raw/).Count
(Get-ChildItem data/raw/ -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
```

### Test 2.3 : Preprocessing
```powershell
docker-compose exec api python src/data/import_data_raw.py ##Si pas de données
docker-compose exec api python src/data/make_dataset.py ##Mettre en Input ./data/raw et Output ./data/processed
docker-compose exec api ls -la data/processed/
```

### Test 2.4 : Feature Engineering
```powershell
docker-compose exec api python src/features/build_features.py
docker-compose exec api ls -la data/processed/
```

### Test 2.5 : Stockage Local
```powershell
docker volume ls | Select-String "mlops"
docker-compose exec api df -h
docker-compose exec api du -sh data/
```

### Test 2.6 : Training Basique
```powershell
docker-compose exec api python src/models/train_model.py
docker-compose exec api ls -lh models/
```

### Test 2.7 : MLflow Tracking
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/health"
docker-compose exec api python src/models/train_model_mlflow.py
docker-compose exec api mlflow experiments search
docker-compose exec api mlflow runs list --experiment-id 1
```

## 🚀 PHASE 3 : Deployment

### Test 3.1 : API FastAPI
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/"
Invoke-RestMethod -Uri "http://localhost:8000/health"
Invoke-RestMethod -Uri "http://localhost:8000/model-info"
start http://localhost:8000/docs
```

### Test 3.2 : Prédiction
#### Directement avec le terminal
```powershell
$features = Get-Content src/models/test_features.json -Raw | ConvertFrom-Json
$body = @{ features = $features } | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```
#### Sur l'api en allant sur predict -> Try it out -> Copier ey Coller le JSON suivant 
```
{
  "features": {"place": 10,
"catu": 3,
"sexe" : 1,
"secu1" : 0.0,
"year_acc" : 2021,
"victim_age" : 60,
"catv" : 2,
"obsm" : 1,
"motor" : 1,
"catr" : 3,
"circ" : 2,
"surf" : 1,
"situ" : 1,
"vma" : 50,
"jour" : 7,
"mois" : 12,
"lum" : 5,
"dep" : 77,
"com" : 77317,
"agg_" : 2,
"int" : 1,
"atm" : 0,
"col" :6, 
"lat" : 48.60,
"long" : 2.89,
"hour" : 17,
"nb_victim" : 2,
"nb_vehicules" : 1}
}
```
#### Pour lancer plueisurs appels sur le modèle
```
# Script pour générer 50 prédictions                                                                           
 1..50 | ForEach-Object {   
$features = Get-Content src/models/test_features.json -Raw | ConvertFrom-Json
$body = @{ features = $features } | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body | Out-Null
Write-Host "Prédiction $_/50" -ForegroundColor Green
Start-Sleep -Milliseconds 500}
```

## 📊 PHASE 4 : Monitoring

### Test 4.1 : Monitoring Drift
```powershell
docker-compose exec api python src/monitoring/evidently_monitor.py
docker-compose exec api cat logs/drift_alerts.json
```
Si donnée de référence manquante éxécuté : 
```powershell
docker-compose exec api python -c "
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Vrais noms
features = ['place', 'catu', 'sexe', 'secu1', 'year_acc', 'victim_age', 
            'catv', 'obsm', 'motor', 'catr', 'circ', 'surf', 'situ', 
            'vma', 'jour', 'mois', 'lum', 'dep', 'com', 'agg_', 'int', 
            'atm', 'col', 'lat', 'long', 'hour', 'nb_victim', 'nb_vehicules']

# Données de référence
data_ref = {name: np.random.randn(n) for name in features}
pd.DataFrame(data_ref).to_csv('./data/processed/train_data.csv', index=False)

# Nouvelles données (avec drift)
data_new = {name: np.random.randn(n) + 0.2 for name in features}
pd.DataFrame(data_new).to_csv('./data/processed/recent_data.csv', index=False)

print('✅ Fichiers recréés avec vrais noms!')
"

# Relancer monitoring
docker-compose exec api python src/monitoring/monitor.py
docker-compose exec api cat logs/drift_alerts.json
```
### Test 4.2 : Retraining
```powershell
docker-compose exec api python src/models/retrain.py
docker-compose exec api ls -la models/backups/
docker-compose exec api cat logs/retraining_report.json
```

## 🧪 Tests Automatisés (pytest)
```powershell
docker-compose exec api pytest tests/ -v
docker-compose exec api pytest tests/ -v --cov=src
docker-compose exec api pytest tests/test_api.py -v
docker-compose exec api pytest tests/test_model.py -v
docker-compose exec api pytest tests/test_date.py -v
```

## 🔄 Pipeline End-to-End
```powershell
docker-compose exec api python src/data/import_raw_data.py
docker-compose exec api python src/data/make_dataset.py   #Mettre en Input ./data/raw et Output ./data/processed
docker-compose exec api python src/features/build_features.py
docker-compose exec api python src/models/train_model_mlflow.py
docker-compose restart api

$features = Get-Content src/models/test_features.json -Raw | ConvertFrom-Json
$body = @{ features = $features } | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body

docker-compose exec api python src/monitoring/evidently_monitor.py #Faire start chemin_du_rapport 
```

## ✅ LocalHost site
### API : http://localhost:8000/docs
### MLflow : http://localhost:5000
### Grafana : http://localhost:3000
### Prometheus : http://localhost:9090
### Airflow : http://localhorst:8080 

