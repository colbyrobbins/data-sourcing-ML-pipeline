#!/usr/bin/env bash
# Deploy Pokémon Card Fair-Value API to Google Cloud Run.
# Usage: ./scripts/deploy_cloud_run.sh <PROJECT_ID> [SERVICE_NAME] [REGION]
set -e

PROJECT_ID="${1}"
SERVICE_NAME="${2:-pokemon-fair-value-api}"
REGION="${3:-us-central1}"

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: $0 <PROJECT_ID> [SERVICE_NAME] [REGION]"
    echo "Example: $0 my-pokemon-project pokemon-fair-value-api us-central1"
    exit 1
fi

IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "==> Setting gcloud project to: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

if [ ! -f "artifacts/model.joblib" ] || [ ! -f "artifacts/model_metadata.json" ]; then
    echo "==> Training Pokémon model and building search index..."
    python3 -m pipeline.train
    python3 -m pipeline.search build
fi

echo "==> Enabling required GCP services (Cloud Build, Cloud Run)..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

echo "==> Building and pushing container image with Cloud Build: $IMAGE_TAG"
gcloud builds submit --tag "$IMAGE_TAG"

echo "==> Deploying service to Google Cloud Run in $REGION..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_TAG" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --port 8080

echo "==> Fetching service URL..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)')

echo "=========================================================="
echo " Deployment successful!"
echo " Live Cloud Run API URL: $SERVICE_URL"
echo " Health check:           $SERVICE_URL/health"
echo " Interactive Docs:       $SERVICE_URL/docs"
echo "=========================================================="
