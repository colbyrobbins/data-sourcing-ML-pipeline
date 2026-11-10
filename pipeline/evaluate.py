"""Hold-out evaluation and feature importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import config


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No depression", "Depression"], zero_division=0
        ),
    }


def feature_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
    """Permutation importance on the raw input columns (drop in ROC-AUC when shuffled)."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)


def format_report(
    comparison: pd.DataFrame,
    best_name: str,
    metrics: dict,
    importance: pd.Series,
    exclude: tuple[str, ...],
    n_train: int,
    n_test: int,
) -> str:
    lines = [
        "# Student Depression Detection - Pipeline Report",
        "",
        f"Train rows: {n_train} | Test rows: {n_test}",
        f"Excluded features: {', '.join(exclude) if exclude else 'none'}",
        "",
        f"## Model comparison ({config.CV_FOLDS}-fold CV on training set)",
        "",
        comparison.round(4).to_markdown(index=False),
        "",
        f"## Best model on hold-out test set: {best_name}",
        "",
    ]
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lines.append(f"- {key}: {metrics[key]:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    lines += [
        "",
        "Confusion matrix (rows = actual, cols = predicted; order: No, Yes):",
        "",
        "```",
        str(cm),
        "```",
        "",
        "```",
        metrics["classification_report"],
        "```",
        "",
        "## Permutation feature importance (mean ROC-AUC drop)",
        "",
        importance.round(4).to_frame("importance").to_markdown(),
        "",
    ]
    return "\n".join(lines)
