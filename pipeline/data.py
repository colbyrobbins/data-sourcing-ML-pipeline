"""Load the cleaned card snapshot from cloud storage (or a local copy) and split it."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import storage
from .config import snapshot_filename
from .ingest import sha256_of


def load_snapshot(cfg: dict) -> pd.DataFrame:
    """Return the snapshot named in the config, downloading it from GCS if needed.

    A local copy in data/processed is reused when its checksum matches, so repeated
    training runs don't re-download. With storage.backend = "local" the file must
    already exist (run `python -m pipeline.ingest` first).
    """
    name = snapshot_filename(cfg)
    local_path = Path(cfg["data"]["processed_dir"]) / name
    expected = cfg["data"].get("snapshot_sha256") or None
    backend = cfg["storage"]["backend"]

    have_valid_copy = local_path.exists() and (expected is None or sha256_of(local_path) == expected)
    if backend == "gcs" and not have_valid_copy:
        bucket = cfg["storage"]["gcs_bucket"]
        if not bucket:
            raise SystemExit(
                "storage.backend is 'gcs' but storage.gcs_bucket is empty. Set the bucket in "
                "config.yaml, or set storage.backend to 'local' to use data/processed."
            )
        blob = f"{cfg['storage']['gcs_prefix']}/{name}"
        print(f"Downloading {storage.gcs_uri(bucket, blob)} ...")
        storage.download_file(bucket, blob, local_path)
    elif backend not in ("gcs", "local"):
        raise ValueError(f"Unknown storage.backend: {backend!r}")

    if not local_path.exists():
        raise FileNotFoundError(
            f"{local_path} not found. Run `python -m pipeline.ingest` to create a snapshot."
        )
    if expected is not None:
        actual = sha256_of(local_path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {local_path}: expected {expected}, got {actual}")

    return pd.read_parquet(local_path)


def split_data(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/validation/test, keeping all printings of a product in one split."""
    s = cfg["split"]
    groups = df[s["group_column"]]

    outer = GroupShuffleSplit(n_splits=1, test_size=s["test_size"], random_state=s["random_state"])
    trainval_idx, test_idx = next(outer.split(df, groups=groups))
    trainval = df.iloc[trainval_idx]

    # val_size is a share of the whole dataset, so rescale it for the remaining rows.
    val_share = s["val_size"] / (1 - s["test_size"])
    inner = GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=s["random_state"])
    train_idx, val_idx = next(inner.split(trainval, groups=trainval[s["group_column"]]))

    return (
        trainval.iloc[train_idx].reset_index(drop=True),
        trainval.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )
