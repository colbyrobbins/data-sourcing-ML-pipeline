# One image for the whole project. The first argument picks what to run:
#   serve (default)  start the API on $PORT (Cloud Run sets PORT=8080)
#   ingest           download prices from TCGCSV and build a snapshot
#   train            train + track the model, then build the search index
#   test             run the test suite
#   mlflow-ui        browse tracked runs on port 5000
# See README.md ("Docker") for the volume mounts each command expects.

FROM python:3.12-slim

ENV GIT_PYTHON_REFRESH=quiet \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PORT=8080

WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the API doesn't download it on cold start.
COPY config.yaml .
RUN python -c "import yaml; from sentence_transformers import SentenceTransformer; \
SentenceTransformer(yaml.safe_load(open('config.yaml'))['search']['embedding_model'])"

# Pass with --build-arg GIT_COMMIT=$(git rev-parse HEAD) so MLflow runs record the code
# version. Declared after the installs so a new commit doesn't invalidate their cache.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Code, plus the trained model and search index if they exist (see .dockerignore).
COPY . .

# Train model and build search index if not already present in the build context
RUN if [ ! -f "artifacts/model.joblib" ]; then \
      python -m pipeline.train && python -m pipeline.search build; \
    fi

RUN chmod +x docker-entrypoint.sh \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8080
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["serve"]
