"""Data sourcing: download the dataset from Kaggle and load it into a DataFrame."""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_dataset(dest: Path = config.RAW_CSV, force: bool = False) -> Path:
    """Download the pinned Kaggle dataset zip, verify its SHA-256 checksum, and save to ``dest``.

    The dataset is public, so Kaggle's download endpoint works without an API key.
    If the download fails, download the CSV manually from Kaggle and place it at ``dest``.
    """
    if dest.exists() and not force:
        if _sha256(dest.read_bytes()) == config.RAW_CSV_SHA256:
            return dest
        print(f"Existing {dest} failed SHA-256 verification; re-downloading ...")

    print(f"Downloading {config.KAGGLE_DATASET} (v{config.KAGGLE_DATASET_VERSION}) ...")
    with urllib.request.urlopen(config.KAGGLE_DOWNLOAD_URL, timeout=60) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(config.CSV_NAME_IN_ZIP))
        csv_bytes = zf.read(member)

    actual = _sha256(csv_bytes)
    if actual != config.RAW_CSV_SHA256:
        raise ValueError(
            f"Checksum mismatch for {config.CSV_NAME_IN_ZIP}: "
            f"expected {config.RAW_CSV_SHA256}, got {actual}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(csv_bytes)
    print(f"Saved and verified {dest}")
    return dest


def load_data(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and apply basic cleaning."""
    # The CSV starts with a UTF-8 BOM, which would otherwise end up in the first column name.
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    expected = set(config.NUMERIC_FEATURES) | set(config.ORDINAL_FEATURES) | set(
        config.NOMINAL_FEATURES
    ) | {config.TARGET}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the binary target (1 = depressed)."""
    y = (df[config.TARGET] == config.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[config.TARGET])
    return X, y


if __name__ == "__main__":
    download_dataset()
