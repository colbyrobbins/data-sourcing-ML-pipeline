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
# Edit the two lines in place so the comments in config.yaml survive.
BUCKET_NAME="$BUCKET_NAME" python3 - <<'PY'
import os, re
from pathlib import Path
path = Path("config.yaml")
text = path.read_text()
text = re.sub(r"(?m)^(  backend:)\s*[^#\s]*", r"\1 gcs", text, count=1)
text = re.sub(r'(?m)^(  gcs_bucket:)\s*[^#\s]*', lambda m: f'{m.group(1)} "{os.environ["BUCKET_NAME"]}"', text, count=1)
path.write_text(text)
print(f"Updated config.yaml: storage.backend=gcs, storage.gcs_bucket={os.environ['BUCKET_NAME']}")
PY

echo "==> Ingesting and uploading snapshot to gs://$BUCKET_NAME/snapshots/..."
python -m pipeline.ingest --upload

echo "==> Done! The bucket is ready and today's snapshot is uploaded."
echo "    To train on it, copy the date and SHA-256 printed above into data.snapshot_date"
echo "    and data.snapshot_sha256 in config.yaml."
