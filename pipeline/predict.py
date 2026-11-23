"""Load the trained model and predict fair values. Meant to be imported by the API.

Usage:
    python -m pipeline.predict '{"name": "Charizard ex - 199/165", "rarity": "Special Illustration Rare",
        "printing": "Holofoil", "number": "199/165", "set_published": "2023-09-22"}' --asking 120
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import ROOT
from .features import RAW_INPUT_COLUMNS

DEFAULT_MODEL_DIR = ROOT / "artifacts"


@lru_cache(maxsize=4)
def load_model(model_dir: str | Path = DEFAULT_MODEL_DIR):
    """Return (model, metadata). Cached so the API loads the model once."""
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    return model, metadata


def predict_fair_value(cards: list[dict], model=None) -> list[float]:
    """Predict a fair market price in dollars for each card dict (see RAW_INPUT_COLUMNS)."""
    if model is None:
        model, _ = load_model()
    frame = pd.DataFrame([{col: card.get(col) for col in RAW_INPUT_COLUMNS} for card in cards])
    return [round(max(float(p), 0.01), 2) for p in model.predict(frame)]


def price_verdict(asking_price: float, fair_value: float, tolerance: float = 0.15) -> str:
    """'deal' if asking is more than `tolerance` below fair value, 'overpriced' if above, else 'fair'."""
    if asking_price < fair_value * (1 - tolerance):
        return "deal"
    if asking_price > fair_value * (1 + tolerance):
        return "overpriced"
    return "fair"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("card", help="Card attributes as a JSON object")
    parser.add_argument("--asking", type=float, help="Asking price to compare against")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args(argv)

    model, _ = load_model(args.model_dir)
    fair = predict_fair_value([json.loads(args.card)], model)[0]
    result = {"fair_value": fair}
    if args.asking is not None:
        result["asking_price"] = args.asking
        result["verdict"] = price_verdict(args.asking, fair)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
