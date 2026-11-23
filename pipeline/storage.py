"""Google Cloud Storage helpers for uploading and downloading data snapshots."""

from pathlib import Path


def _client():
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage

    try:
        return storage.Client()
    except DefaultCredentialsError:
        # No credentials on this machine: fall back to anonymous access, which
        # works for reading from a bucket that allows public reads.
        return storage.Client.create_anonymous_client()


def gcs_uri(bucket: str, blob_name: str) -> str:
    return f"gs://{bucket}/{blob_name}"


def upload_file(local_path: Path, bucket: str, blob_name: str) -> str:
    """Upload a local file to gs://bucket/blob_name and return the URI."""
    blob = _client().bucket(bucket).blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return gcs_uri(bucket, blob_name)


def download_file(bucket: str, blob_name: str, local_path: Path) -> Path:
    """Download gs://bucket/blob_name to local_path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob = _client().bucket(bucket).blob(blob_name)
    blob.download_to_filename(str(local_path))
    return local_path
