"""Candidate regressors. Each is a full sklearn pipeline: raw card fields in, dollars out."""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import card_features_from_config


def _tree_preprocessor(categorical: list, numeric: list, impute: bool) -> ColumnTransformer:
    num_step = SimpleImputer(strategy="median") if impute else "passthrough"
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", num_step, numeric),
        ],
        verbose_feature_names_out=False,
    )


def _linear_preprocessor(categorical: list, numeric: list) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_model(name: str, params: dict, cfg: dict, random_state: int = 42):
    """Build candidate `name`. The target is modeled as log(1 + price) and converted back."""
    features = card_features_from_config(cfg)
    cat, num = features.categorical, features.output_numeric
    n_cat = len(cat)

    if name == "hist_gbm":
        preprocess = _tree_preprocessor(cat, num, impute=False)  # handles NaN natively
        regressor = HistGradientBoostingRegressor(
            categorical_features=list(range(n_cat)), random_state=random_state, **params
        )
    elif name == "random_forest":
        preprocess = _tree_preprocessor(cat, num, impute=True)
        regressor = RandomForestRegressor(n_jobs=-1, random_state=random_state, **params)
    elif name == "ridge":
        preprocess = _linear_preprocessor(cat, num)
        regressor = Ridge(**params)
    else:
        raise ValueError(f"Unknown model candidate: {name!r}")

    pipeline = Pipeline([("features", features), ("preprocess", preprocess), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
