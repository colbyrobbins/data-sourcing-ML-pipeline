# data-sourcing-ML-pipeline

A simple, end-to-end ML pipeline that predicts student depression from the Kaggle
[Student Depression Detection Dataset](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset)
(4,464 students from Bangladesh, 12 features, binary `Depression` target, ~27% positive).

## Quick start

```bash
pip install -r requirements.txt
python -m pipeline
```

The last line of output is the headline result, the accuracy on the held-out test set:

```
Test accuracy: 0.9888
```

It is also the first line of `artifacts/report.md`.

The dataset is public, so it is downloaded automatically from Kaggle on the first run
(no API key needed) into `data/raw/`. If the download fails, download the CSV manually
from Kaggle and place it at `data/raw/Student_Depression_Dataset.csv`.

Options:

```bash
python -m pipeline --exclude Health_condition   # drop one or more features
python -m pipeline --force-download             # re-download the data
python -m pipeline --out artifacts/my_run       # choose output folder
```

Run the tests (they use a small synthetic dataset, so no download is needed):

```bash
python -m pytest
```

## Pipeline steps

| Step | Module | What it does |
|---|---|---|
| 1. Source | `pipeline/data.py` | Downloads the Kaggle zip, extracts the CSV, loads it (handles the file's UTF-8 BOM), trims strings, drops duplicates, validates columns |
| 2. Split | `pipeline/__main__.py` | Stratified 80/20 train/test split |
| 3. Preprocess | `pipeline/features.py` | Numeric: median impute + standard scale. Ordered categories (`Family_background`, `Health_condition`): ordinal encode. Other categories: one-hot encode |
| 4. Select | `pipeline/model.py` | 5-fold stratified CV on the training set for Logistic Regression, Random Forest and Gradient Boosting (class-weighted where supported); picks the best by ROC-AUC |
| 5. Evaluate | `pipeline/evaluate.py` | Trains the best model on the full training set, scores it once on the held-out test set, computes permutation feature importance |
| 6. Save | `pipeline/__main__.py` | Writes `model.joblib`, `metrics.json`, `model_comparison.csv` and `report.md` to `artifacts/` |

Column groups, paths and constants live in `pipeline/config.py`.

## Results

| Run | Best model | Test ROC-AUC | Test F1 | Test accuracy |
|---|---|---|---|---|
| All features | Random Forest | 0.999 | 0.979 | 0.989 |
| Without `Health_condition` | Random Forest | 0.726 | 0.492 | 0.727 |

### Caveat: `Health_condition` nearly determines the label

In this dataset every student with `Health_condition = Critical` is labelled depressed and
every student with `Good` is not (`Abnormal` is 55% depressed, `Normal` 5%). The model
leans almost entirely on that one column, which is why the full-feature scores are near
perfect. If health condition is assessed together with (or after) the depression
diagnosis, it leaks the answer, and the honest estimate is the run without it
(ROC-AUC ~0.73). Without it, `Relationship_status`, `friend_circle` and `CGPA` are the
most useful signals.

## Project layout

```
pipeline/
  config.py      paths, column groups, constants
  data.py        download + load + clean
  features.py    preprocessing (ColumnTransformer)
  model.py       candidate models, CV comparison, training
  evaluate.py    test metrics, feature importance, report
  __main__.py    CLI that runs every step
tests/
  test_pipeline.py
```

`data/` and `artifacts/` are git-ignored; they are recreated by running the pipeline.
