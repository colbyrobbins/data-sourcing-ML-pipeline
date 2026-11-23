#!/usr/bin/env bash
# Setup Google Cloud Storage bucket for Pokémon Card Fair-Value snapshots.
# Usage: ./scripts/setup_gcs.sh <PROJECT_ID> <BUCKET_NAME> [REGION]
set -e

PROJECT_ID="${1}"
BUCKET_NAME="${2}"
REGION="${3:-us-central1}"

if [ -z "$PROJECT_ID" ] || [ -z "$BUCKET_NAME" ]; then
    echo "Usage: $0 <PROJECT_ID> <BUCKET_NAME> [REGION]"
    echo "Example: $0 my-pokemon-project pokemon-fair-value-data us-central1"
    exit 1
fi

echo "==> Setting gcloud project to: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "==> Creating GCS bucket: gs://$BUCKET_NAME in region $REGION"
gcloud storage buckets create "gs://$BUCKET_NAME" --location="$REGION" --uniform-bucket-level-access || true

echo "==> Granting public read access so training downloads work without credentials..."
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
    --member="allUsers" \
    --role="roles/storage.objectViewer"

echo "==> Ensuring required Python dependencies are present..."
python3 -c "import pandas, google.cloud.storage, pyarrow, yaml, requests" 2>/dev/null || pip install pandas pyarrow pyyaml requests google-cloud-storage

echo "==> Updating config.yaml with GCS bucket..."
python -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['storage']['backend'] = 'gcs'
cfg['storage']['gcs_bucket'] = '$BUCKET_NAME'
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('Updated config.yaml: storage.backend=gcs, storage.gcs_bucket=$BUCKET_NAME')
"

echo "==> Ingesting and uploading snapshot to gs://$BUCKET_NAME/snapshots/..."
python -m pipeline.ingest --upload

echo "==> Done! GCS Bucket is ready and snapshot is uploaded."
