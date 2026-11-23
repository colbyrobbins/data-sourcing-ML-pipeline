param (
    [Parameter(Mandatory=$true)][string]$ProjectId,
    [string]$ServiceName = "pokemon-fair-value-api",
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"

$ImageTag = "gcr.io/$ProjectId/${ServiceName}:latest"

Write-Host "==> Setting gcloud project to: $ProjectId"
gcloud config set project $ProjectId

if (-not (Test-Path "artifacts/model.joblib") -or -not (Test-Path "artifacts/model_metadata.json")) {
    Write-Host "==> Training Pokémon model and building search index..."
    python -m pipeline.train
    python -m pipeline.search build
}

Write-Host "==> Enabling required GCP services (Cloud Build, Cloud Run)..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

Write-Host "==> Building and pushing container image with Cloud Build: $ImageTag"
gcloud builds submit --tag $ImageTag

Write-Host "==> Deploying service to Google Cloud Run in $Region..."
gcloud run deploy $ServiceName `
    --image $ImageTag `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --port 8080

$ServiceUrl = gcloud run services describe $ServiceName --platform managed --region $Region --format 'value(status.url)'

Write-Host "=========================================================="
Write-Host " Deployment successful!"
Write-Host " Live Cloud Run API URL: $ServiceUrl"
Write-Host " Health check:           $ServiceUrl/health"
Write-Host " Interactive Docs:       $ServiceUrl/docs"
Write-Host "=========================================================="
