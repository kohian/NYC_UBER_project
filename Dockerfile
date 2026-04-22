# -------- Base image --------
FROM python:3.11-slim AS base

RUN useradd -m appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir --no-deps .

# -------- Test stage --------
FROM base AS test
COPY requirements_dev.txt .
RUN pip install --no-cache-dir -r requirements_dev.txt
COPY tests/ ./tests/
COPY model_artifacts/ ./model_artifacts

RUN chown -R appuser:appuser /app

USER appuser

CMD ["pytest"]

# -------- Production stage --------
FROM base AS prod

RUN chown -R appuser:appuser /app

USER appuser

CMD ["uvicorn", "iris_inference_api.main:app", "--host", "0.0.0.0", "--port", "8080"]