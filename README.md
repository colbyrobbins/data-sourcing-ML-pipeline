# data-sourcing-ML-pipeline

Reproducible ML pipeline starting with the Kaggle [Student Depression Detection Dataset](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset).

## Setup

Create/activate the virtual environment and install the package in editable mode:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Reproducible Dataset Ingestion

Raw dataset files are gitignored under `data/raw/` and pinned in [`data/manifest.json`](data/manifest.json) by Kaggle dataset version (`versions/1`), file byte size, and SHA-256 digest.

Fetch and verify the dataset into `data/raw/`:

```powershell
.\.venv\Scripts\python.exe -m ml_pipeline.data.ingest
```

Or using the installed script entrypoint:

```powershell
.\.venv\Scripts\fetch-dataset.exe
```

Useful flags:
- `--verify-only`: Check local `data/raw/` files against `data/manifest.json` without downloading.
- `--force`: Re-download from KaggleHub and overwrite local files.

This public dataset downloads without authentication via `kagglehub`. If you later configure private datasets or hit rate limits, set `KAGGLE_USERNAME` / `KAGGLE_KEY` or place `~/.kaggle/kaggle.json`.

### Dataset Contents (`versions/1`)

- `data/raw/Student_Depression_Dataset.csv` (4,464 rows, 13 columns: `Gender`, `Age`, `Family_background`, `Relationship_status`, `Marital_status`, `Living_style`, `Current_semester`, `CGPA`, `friend_circle`, `Health_condition`, `Cost_without_study`, `Home_division`, `Depression`)
- `data/raw/Student_Depression_Dataset.xlsx`

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
