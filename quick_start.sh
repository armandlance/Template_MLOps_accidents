#!/bin/bash

# Script de démarrage rapide pour le projet MLOps
# Usage: bash quickstart.sh

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "🚀 MLOPS PROJECT - QUICK START"
echo "=========================================="
echo ""

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Vérifier les prérequis
info "Vérification des prérequis..."

if ! command -v docker &> /dev/null; then
    error "Docker n'est pas installé. Veuillez l'installer: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose n'est pas installé."
    exit 1
fi

if ! command -v git &> /dev/null; then
    warn "Git n'est pas installé. Certaines fonctionnalités peuvent ne pas fonctionner."
fi

info "✅ Prérequis OK"
echo ""

# 2. Créer la structure de dossiers
info "Création de la structure de dossiers..."

mkdir -p data/raw data/processed data/interim data/external
mkdir -p models/backups logs mlruns dags tests
mkdir -p src/api src/data src/features src/models src/monitoring src/dashboard
mkdir -p .github/workflows

info "✅ Structure créée"
echo ""

# 3. Créer les fichiers __init__.py
info "Création des fichiers __init__.py..."

touch src/__init__.py
touch src/api/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/monitoring/__init__.py
touch src/dashboard/__init__.py
touch tests/__init__.py

info "✅ Fichiers __init__.py créés"
echo ""

# 4. Initialiser DVC si pas déjà fait
if [ ! -d ".dvc" ]; then
    info "Initialisation de DVC..."
    if command -v dvc &> /dev/null; then
        dvc init
        info "✅ DVC initialisé"
    else
        warn "DVC n'est pas installé. Installation..."
        pip install dvc
        dvc init
        info "✅ DVC installé et initialisé"
    fi
else
    info "✅ DVC déjà initialisé"
fi
echo ""

# 5. Créer le fichier .env depuis .env.example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        info "Création du fichier .env..."
        cp .env.example .env
        info "✅ Fichier .env créé"
    else
        warn ".env.example n'existe pas, .env non créé"
    fi
else
    info "✅ .env existe déjà"
fi
echo ""

# 6. Build des images Docker
info "Construction des images Docker..."
echo "⏳ Cela peut prendre quelques minutes..."

docker-compose build

info "✅ Images Docker construites"
echo ""

# 7. Démarrer les services
info "Démarrage des services..."

docker-compose up -d

info "✅ Services démarrés"
echo ""

# 8. Attendre que les services soient prêts
info "Attente du démarrage complet des services..."
sleep 10

# Vérifier si les services sont en cours d'exécution
if docker-compose ps | grep -q "Up"; then
    info "✅ Services en cours d'exécution"
else
    error "Certains services ne sont pas démarrés correctement"
    docker-compose ps
    exit 1
fi
echo ""

# 9. Afficher les URLs d'accès
echo "=========================================="
echo "✅ INSTALLATION TERMINÉE!"
echo "=========================================="
echo ""
echo "📍 Accès aux services:"
echo ""
echo "   🌐 API FastAPI:     http://localhost:8000"
echo "   📚 API Docs:        http://localhost:8000/docs"
echo "   💾 MLflow UI:       http://localhost:5000"
echo "   🔄 Airflow UI:      http://localhost:8080"
echo "      Login: admin / admin"
echo ""
echo "=========================================="
echo "📝 Prochaines étapes:"
echo "=========================================="
echo ""
echo "1. Vérifier la santé de l'API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "2. Entraîner le modèle:"
echo "   docker-compose exec api python src/models/train_model_mlflow.py"
echo ""
echo "3. Faire une prédiction:"
echo "   curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"features\":{\"feature1\":0.5}}'"
echo ""
echo "4. Voir les logs:"
echo "   docker-compose logs -f"
echo ""
echo "5. Arrêter les services:"
echo "   docker-compose down"
echo ""
echo "=========================================="
echo "📚 Documentation complète: README_MLOPS_COMPLET.md"
echo "=========================================="
echo ""

# Test de santé de l'API
info "Test de santé de l'API..."
sleep 5

if curl -s http://localhost:8000/health > /dev/null; then
    info "✅ L'API répond correctement!"
else
    warn "⚠️ L'API ne répond pas encore. Attendez quelques secondes et réessayez:"
    echo "   curl http://localhost:8000/health"
fi

echo ""
echo "🎉 Tout est prêt! Bon développement!"