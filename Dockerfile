# -------- Base image --------
FROM python:3.11-slim AS base

RUN useradd -m appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir --no-deps .

# # -------- Production stage --------
# FROM base AS prod

RUN chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "nyc_forecasting.train_lstm"]

# # -------- Test stage --------
# FROM base AS test
# COPY requirements_dev.txt .
# RUN pip install --no-cache-dir -r requirements_dev.txt
# COPY tests/ ./tests/

# RUN chown -R appuser:appuser /app

# USER appuser

# CMD ["pytest"]

# LOCAL RUNNING
# docker run --rm -v "$env:APPDATA\gcloud:/home/appuser/.config/gcloud:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/home/appuser/.config/gcloud/application_default_credentials.json nyc_uber_demand