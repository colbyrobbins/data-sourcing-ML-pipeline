"""Data sourcing: download the dataset from Kaggle and load it into a DataFrame."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def download_dataset(dest: Path = config.RAW_CSV, force: bool = False) -> Path:
    """Download the Kaggle dataset zip and extract the CSV to ``dest``.

    The dataset is public, so Kaggle's download endpoint works without an API key.
    If the download fails, download the CSV manually from Kaggle and place it at ``dest``.
    """
    if dest.exists() and not force:
        return dest

    print(f"Downloading {config.KAGGLE_DATASET} ...")
    with urllib.request.urlopen(config.KAGGLE_DOWNLOAD_URL, timeout=60) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(config.CSV_NAME_IN_ZIP))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(member))

    print(f"Saved to {dest}")
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

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the binary target (1 = depressed)."""
    y = (df[config.TARGET] == config.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[config.TARGET])
    return X, y
