#!/bin/sh
# Entry points for the Docker image. Usage: docker run IMAGE <command> [args...]
set -e

cmd="${1:-serve}"
[ $# -gt 0 ] && shift

case "$cmd" in
  serve)
    if [ ! -f "${MODEL_DIR:-artifacts}/model.joblib" ]; then
      echo "No trained model in ${MODEL_DIR:-artifacts}/. Run 'train' first (see README)." >&2
      exit 1
    fi
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}" "$@"
    ;;
  ingest)
    exec python -m pipeline.ingest "$@"
    ;;
  train)
    python -m pipeline.train "$@"
    exec python -m pipeline.search "$@" build
    ;;
  test)
    exec python -m pytest -q "$@"
    ;;
  mlflow-ui)
    exec mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////app/mlruns/mlflow.db "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
