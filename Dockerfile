ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS deps
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_lg

FROM python:${PYTHON_VERSION}-slim AS production
ARG SERVICE_NAME
ENV SERVICE_NAME=${SERVICE_NAME} \
    PYTHONPATH=/app \
    # System users have HOME=/nonexistent; redirect all model/config caches to /tmp
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl libpq5 && rm -rf /var/lib/apt/lists/*
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY shared/ ./shared/
COPY services/${SERVICE_NAME}/app/ ./app/
USER appuser
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --no-access-log
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
