"""API tests with a fake model and fake search, so no training or model download is needed."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from pipeline.features import RAW_INPUT_COLUMNS

METADATA = {
    "model": "hist_gbm",
    "snapshot_date": "2026-11-22",
    "trained_at": "2026-11-22T18:00:00+00:00",
    "test_metrics": {"rmse_log": 0.66, "r2_log": 0.80},
}


class FakeModel:
    """Prices a card at $100 if it's a Special Illustration Rare, else $1."""

    def __init__(self):
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame)
        return np.where(frame["rarity"] == "Special Illustration Rare", 100.0, 1.0)


class FakeSearch:
    def search(self, query, top_k=5):
        return [
            {
                "product_id": 517045, "name": "Charizard ex - 199/165", "set_name": "SV: Scarlet & Violet 151",
                "rarity": "Special Illustration Rare", "number": "199/165", "card_type": "Fire",
                "stage": "Stage 2", "hp": 330.0, "retreat_cost": 2.0, "set_published": "2023-09-22",
                "is_supplemental": False, "has_attack2": True, "printings": {"Holofoil": 279.37},
                "image_url": "https://example.com/c.jpg", "url": "https://example.com/c", "score": 1.14,
            }
        ][:top_k]


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def client(model):
    app = create_app(model_loader=lambda: (model, METADATA), search_loader=lambda: FakeSearch())
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_name"] == "hist_gbm" and body["search_enabled"]


def test_predict_with_verdict(client, model):
    resp = client.post("/predict", json={
        "name": "Charizard ex - 199/165", "rarity": "Special Illustration Rare", "printing": "Holofoil",
        "number": "199/165", "set_published": "2023-09-22", "hp": 330, "asking_price": 60,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["fair_value"] == 100.0 and body["verdict"] == "deal"
    assert body["snapshot_date"] == "2026-11-22"
    # The model receives exactly the raw columns it was trained on.
    assert list(model.frames[-1].columns) == RAW_INPUT_COLUMNS


def test_predict_minimal_card_without_asking_price(client):
    body = client.post("/predict", json={"name": "Bulbasaur", "rarity": "Common"}).json()
    assert body["fair_value"] == 1.0 and body["verdict"] is None


@pytest.mark.parametrize("bad", [
    {"rarity": "Common"},                                              # missing name
    {"name": "X", "rarity": "Common", "set_published": "Sept 2023"},   # bad date format
    {"name": "X", "rarity": "Common", "asking_price": -5},             # negative price
])
def test_predict_rejects_bad_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_search_result_round_trips_into_predict(client):
    body = client.get("/search", params={"q": "charizard ex 151", "k": 3}).json()
    top = body["results"][0]
    assert top["name"] == "Charizard ex - 199/165" and top["printings"] == {"Holofoil": 279.37}
    assert top["card"]["printing"] == "Holofoil"
    pred = client.post("/predict", json={**top["card"], "asking_price": 400}).json()
    assert pred["fair_value"] == 100.0 and pred["verdict"] == "overpriced"


def test_search_validates_query(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert client.get("/search", params={"q": "pikachu", "k": 100}).status_code == 422


def test_search_unavailable_returns_503(model):
    def broken():
        raise FileNotFoundError("no index")

    app = create_app(model_loader=lambda: (model, METADATA), search_loader=broken)
    with TestClient(app) as c:
        assert c.get("/health").json()["search_enabled"] is False
        assert c.get("/search", params={"q": "pikachu"}).status_code == 503
        assert c.post("/predict", json={"name": "Pikachu", "rarity": "Common"}).status_code == 200
