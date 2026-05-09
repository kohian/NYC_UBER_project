# -------- Base image --------
FROM python:3.11-slim AS base

RUN useradd -m appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir --no-deps .

# -------- Training image --------
FROM base AS training

COPY requirements_training.txt .

RUN pip install --no-cache-dir -r requirements_training.txt

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "nyc_forecasting.training.train_xgboost"]

# -------- Inference image --------
FROM base AS inference

COPY requirements_inference.txt .

RUN pip install --no-cache-dir -r requirements_inference.txt

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "nyc_forecasting.inference.xgboost_inference"]


# LOCAL RUNNING
# docker run --rm -v "$env:APPDATA\gcloud:/home/appuser/.config/gcloud:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/home/appuser/.config/gcloud/application_default_credentials.json nyc_uber_demand