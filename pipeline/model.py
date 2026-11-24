"""Candidate regressors. Each is a full sklearn pipeline: raw card fields in, dollars out."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import card_features_from_config


class TwoStageHurdleRegressor(BaseEstimator, RegressorMixin):
    """Two-stage hurdle regressor for heavily right-skewed collectible card prices.

    Stage 1: A classifier predicts whether a card is an expensive outlier
             (log price >= threshold, corresponding to e.g. $25+).
    Stage 2: Two separate regressors fit on standard cards and high-value cards.
    Prediction: Soft blending using the predicted probability of being high value:
        y_pred = p_high * pred_high + (1 - p_high) * pred_bulk
    """

    def __init__(
        self,
        threshold_price: float = 25.0,
        categorical_features: list | None = None,
        min_samples_leaf: int = 5,
        clf_max_iter: int = 100,
        clf_max_depth: int = 4,
        reg_max_iter: int = 200,
        random_state: int = 42,
    ):
        self.threshold_price = threshold_price
        self.categorical_features = categorical_features
        self.min_samples_leaf = min_samples_leaf
        self.clf_max_iter = clf_max_iter
        self.clf_max_depth = clf_max_depth
        self.reg_max_iter = reg_max_iter
        self.random_state = random_state

    def fit(self, X, y):
        # y is log(1 + price) because of TransformedTargetRegressor
        y = np.asarray(y)
        log_thresh = np.log1p(self.threshold_price)
        is_high = (y >= log_thresh).astype(int)

        self.reg_bulk_ = HistGradientBoostingRegressor(
            max_iter=self.reg_max_iter,
            min_samples_leaf=self.min_samples_leaf,
            categorical_features=self.categorical_features,
            random_state=self.random_state,
        )
        self.reg_high_ = HistGradientBoostingRegressor(
            max_iter=self.reg_max_iter,
            min_samples_leaf=self.min_samples_leaf,
            categorical_features=self.categorical_features,
            random_state=self.random_state,
        )

        n_classes = len(np.unique(is_high))
        if n_classes < 2 or (is_high == 1).sum() < max(5, self.min_samples_leaf):
            # Fallback when there are too few high-value samples in a split
            self.single_class_ = True
            self.reg_bulk_.fit(X, y)
            self.reg_high_.fit(X, y)
        else:
            self.single_class_ = False
            self.clf_ = HistGradientBoostingClassifier(
                max_iter=self.clf_max_iter,
                max_depth=self.clf_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                categorical_features=self.categorical_features,
                random_state=self.random_state,
            )
            self.clf_.fit(X, is_high)

            mask_bulk = is_high == 0
            mask_high = is_high == 1
            self.reg_bulk_.fit(X[mask_bulk], y[mask_bulk])
            self.reg_high_.fit(X[mask_high], y[mask_high])

        return self

    def predict(self, X):
        if getattr(self, "single_class_", False):
            return self.reg_bulk_.predict(X)

        p_high = self.clf_.predict_proba(X)[:, 1]
        pred_bulk = self.reg_bulk_.predict(X)
        pred_high = self.reg_high_.predict(X)
        return p_high * pred_high + (1.0 - p_high) * pred_bulk


def _tree_preprocessor(categorical: list, numeric: list, impute: bool) -> ColumnTransformer:
    num_step = SimpleImputer(strategy="median", keep_empty_features=True) if impute else "passthrough"
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
    elif name == "hurdle_outlier":
        preprocess = _tree_preprocessor(cat, num, impute=True)
        regressor = TwoStageHurdleRegressor(
            categorical_features=list(range(n_cat)), random_state=random_state, **params
        )
    else:
        raise ValueError(f"Unknown model candidate: {name!r}")

    pipeline = Pipeline([("features", features), ("preprocess", preprocess), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
