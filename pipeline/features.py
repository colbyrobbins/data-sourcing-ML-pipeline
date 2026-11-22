"""Feature engineering that turns raw card attributes into model inputs.

`CardFeatures` is the first step of the saved model, so the model accepts the same raw
fields a user (or the API) knows about a card. The API never re-implements features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Raw fields the model expects for each card. Missing optional values may be None.
RAW_INPUT_COLUMNS = [
    "name",            # e.g. "Charizard ex - 199/165"
    "rarity",          # e.g. "Special Illustration Rare"
    "printing",        # Normal / Holofoil / Reverse Holofoil / 1st Edition ...
    "card_type",       # Fire / Water / Supporter / Item ... (optional)
    "stage",           # Basic / Stage 1 / Stage 2 ... (optional)
    "hp",              # number (optional)
    "retreat_cost",    # number (optional)
    "number",          # collector number, e.g. "199/165" (optional)
    "set_published",   # set release date, "YYYY-MM-DD"
    "is_supplemental", # bool: promo / special set
    "has_attack2",     # bool: card has a second attack
]

_NUMBER_PATTERN = r"(\d+)\D*/\D*(\d+)"  # "199/165", "TG05/TG30"


class CardFeatures(BaseEstimator, TransformerMixin):
    """Derive numeric and flag features from raw card attributes."""

    def __init__(self, reference_date: str, name_flags: dict, categorical: list, numeric: list):
        self.reference_date = reference_date
        self.name_flags = name_flags
        self.categorical = categorical
        self.numeric = numeric

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in RAW_INPUT_COLUMNS:
            if col not in X:
                X[col] = None

        ref = pd.Timestamp(self.reference_date)
        published = pd.to_datetime(X["set_published"], errors="coerce")
        X["set_age_years"] = (ref - published).dt.days / 365.25

        numbers = X["number"].astype("string").str.extract(_NUMBER_PATTERN)
        X["card_number"] = pd.to_numeric(numbers[0], errors="coerce")
        X["set_size"] = pd.to_numeric(numbers[1], errors="coerce")
        # Cards numbered past the set size are "secret" rares, usually the chase cards.
        X["is_secret_number"] = (X["card_number"] > X["set_size"]).astype(float)

        for col in ("hp", "retreat_cost"):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in ("has_attack2", "is_supplemental"):
            X[col] = X[col].fillna(False).astype(bool).astype(float)

        names = X["name"].fillna("").astype(str)
        for flag, pattern in self.name_flags.items():
            X[flag] = names.str.contains(pattern, regex=True).astype(float)

        out = pd.DataFrame(index=X.index)
        for col in self.categorical:
            out[col] = X[col].astype("object").where(X[col].notna(), "missing").astype(str)
        for col in self.output_numeric:
            out[col] = X[col].astype(float)
        return out

    @property
    def output_numeric(self) -> list:
        return list(self.numeric) + list(self.name_flags)

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.categorical) + self.output_numeric)


def card_features_from_config(cfg: dict) -> CardFeatures:
    f = cfg["features"]
    return CardFeatures(
        reference_date=f["reference_date"],
        name_flags=dict(f["name_flags"]),
        categorical=list(f["categorical"]),
        numeric=list(f["numeric"]),
    )
