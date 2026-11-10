"""Smoke tests that run the pipeline pieces on a small synthetic dataset (no download needed)."""

import numpy as np
import pandas as pd
import pytest

from pipeline import config
from pipeline.data import load_data, split_features_target
from pipeline.evaluate import evaluate
from pipeline.features import build_preprocessor
from pipeline.model import compare_models, train_model


@pytest.fixture
def sample_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    health = rng.choice(["Critical", "Abnormal", "Normal", "Good"], n)
    df = pd.DataFrame(
        {
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 28, n),
            "Family_background": rng.choice(config.ORDINAL_FEATURES["Family_background"], n),
            "Relationship_status": rng.choice(["Single", "In a Relationship", "Complicated"], n),
            "Marital_status": rng.choice(["Unmarried", "Married", "Separated"], n),
            "Living_style": rng.choice(["Bachelor", "Family"], n),
            "Current_semester": rng.integers(1, 12, n),
            "CGPA": rng.uniform(2.5, 4.0, n).round(2),
            "friend_circle": rng.choice(["Yes", "No"], n),
            "Health_condition": health,
            "Cost_without_study": rng.integers(0, 30000, n),
            "Home_division": rng.choice(["Dhaka", "Khulna", "Sylhet"], n),
            "Depression": np.where(np.isin(health, ["Critical", "Abnormal"]), "Yes", "No"),
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")  # mimic the BOM in the real file
    return path


def test_load_data_strips_bom(sample_csv):
    df = load_data(sample_csv)
    assert "Gender" in df.columns
    assert len(df) == 200


def test_split_features_target(sample_csv):
    X, y = split_features_target(load_data(sample_csv))
    assert config.TARGET not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_preprocessor_excludes_columns(sample_csv):
    X, _ = split_features_target(load_data(sample_csv))
    full = build_preprocessor().fit_transform(X)
    reduced = build_preprocessor(exclude=("Health_condition",)).fit_transform(X)
    assert reduced.shape[1] == full.shape[1] - 1


def test_end_to_end(sample_csv):
    X, y = split_features_target(load_data(sample_csv))
    comparison = compare_models(X, y)
    assert len(comparison) == 3
    model = train_model(comparison.loc[0, "model"], X, y)
    metrics = evaluate(model, X, y)
    assert metrics["roc_auc"] > 0.9  # target is a deterministic function of Health_condition
