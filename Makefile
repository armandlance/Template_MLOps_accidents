.PHONY: help setup build up down logs test clean

help:
	@echo "MLOps Project - Available commands:"
	@echo "  make setup     - Initial setup (create folders, init DVC)"
	@echo "  make build     - Build Docker images"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make logs      - Show logs from all services"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean up generated files"
	@echo "  make train     - Train model with MLflow"
	@echo "  make predict   - Test prediction API"

setup:
	@echo "🔧 Setting up project..."
	mkdir -p data/raw data/processed data/interim data/external
	mkdir -p models logs mlruns dags tests src/api
	touch src/api/__init__.py tests/__init__.py
	dvc init || true
	@echo "✅ Setup complete!"

build:
	@echo "🏗️  Building Docker images..."
	docker-compose build

up:
	@echo "🚀 Starting services..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "   API: http://localhost:8000"
	@echo "   MLflow: http://localhost:5000"
	@echo "   Airflow: http://localhost:8080"

down:
	@echo "🛑 Stopping services..."
	docker-compose down

logs:
	docker-compose logs -f

test:
	@echo "🧪 Running tests..."
	docker-compose exec api pytest tests/ -v

train:
	@echo "🎯 Training model..."
	docker-compose exec api python src/models/train_model_mlflow.py

predict:
	@echo "🔮 Testing prediction..."
	curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"features": {"feature1": 0.5, "feature2": 1.2, "feature3": 0.8}}'

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✅ Cleanup complete!"