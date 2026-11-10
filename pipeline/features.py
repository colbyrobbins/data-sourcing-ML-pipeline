"""Feature preprocessing: scaling numerics and encoding categoricals."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config


def build_preprocessor(exclude: tuple[str, ...] = ()) -> ColumnTransformer:
    """Build a ColumnTransformer for the dataset, optionally leaving some columns out."""
    numeric = [c for c in config.NUMERIC_FEATURES if c not in exclude]
    ordinal = {c: cats for c, cats in config.ORDINAL_FEATURES.items() if c not in exclude}
    nominal = [c for c in config.NOMINAL_FEATURES if c not in exclude]

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if ordinal:
        transformers.append(
            (
                "ord",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                categories=list(ordinal.values()),
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                list(ordinal),
            )
        )
    if nominal:
        transformers.append(
            (
                "nom",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                nominal,
            )
        )

    # Any column not listed (including excluded ones) is dropped.
    return ColumnTransformer(transformers, remainder="drop")
