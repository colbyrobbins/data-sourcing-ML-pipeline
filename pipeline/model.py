"""Model candidates, cross-validated selection, and final training."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor

CV_METRICS = ["roc_auc", "f1", "accuracy"]


def candidate_models() -> dict:
    """Classifiers to compare. Class weights offset the ~27% positive rate."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


def build_pipeline(estimator, exclude: tuple[str, ...] = ()) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(exclude)), ("model", estimator)])


def compare_models(
    X: pd.DataFrame, y: pd.Series, exclude: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Cross-validate every candidate on the training data; return mean scores sorted by ROC-AUC."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, estimator in candidate_models().items():
        scores = cross_validate(
            build_pipeline(estimator, exclude), X, y, cv=cv, scoring=CV_METRICS, n_jobs=-1
        )
        row = {"model": name}
        for metric in CV_METRICS:
            row[f"cv_{metric}"] = scores[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)


def train_model(name: str, X: pd.DataFrame, y: pd.Series, exclude: tuple[str, ...] = ()) -> Pipeline:
    pipe = build_pipeline(candidate_models()[name], exclude)
    pipe.fit(X, y)
    return pipe
