param (
    [Parameter(Mandatory=$true)][string]$ProjectId,
    [Parameter(Mandatory=$true)][string]$BucketName,
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Setting gcloud project to: $ProjectId"
gcloud config set project $ProjectId

Write-Host "==> Creating GCS bucket: gs://$BucketName in region $Region"
gcloud storage buckets create "gs://$BucketName" --location=$Region --uniform-bucket-level-access

Write-Host "==> Granting public read access so training downloads work without credentials..."
gcloud storage buckets add-iam-policy-binding "gs://$BucketName" `
    --member="allUsers" `
    --role="roles/storage.objectViewer"

Write-Host "==> Ingesting and uploading snapshot to gs://$BucketName/snapshots/..."
python -m pipeline.ingest --upload

Write-Host "==> Done! GCS Bucket is ready and snapshot is uploaded."
