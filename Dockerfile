FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements
COPY requirements_mlops.txt .

# Installation des packages Python (ignore setup.py errors)
RUN pip install --no-cache-dir -r requirements_mlops.txt || true && \
    pip install --no-cache-dir mlflow dvc fastapi uvicorn pytest scikit-learn pandas numpy joblib apache-airflow pydantic requests python-dotenv SQLAlchemy evidently prometheus-client

# Copie du code source
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p data/raw data/processed logs models

# Exposition du port pour l'API
EXPOSE 8000

# Commande par défaut
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]