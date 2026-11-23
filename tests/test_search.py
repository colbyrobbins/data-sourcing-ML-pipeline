"""Semantic search tests. A hashing bag-of-words embedder stands in for the real model."""

import hashlib
import re

import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.features import RAW_INPUT_COLUMNS
from pipeline.predict import predict_fair_value
from pipeline.search import CardSearch, build_index, card_documents, model_input


class FakeEmbedder:
    """Deterministic, offline: each word hashes to a dimension; vectors are L2-normalized."""

    dim = 256

    def encode(self, texts, batch_size=256):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def _cards() -> pd.DataFrame:
    base = {"card_type": None, "stage": "Basic", "hp": 60.0, "retreat_cost": 1.0, "set_name": "Test Set",
            "set_published": "2023-09-22", "is_supplemental": False, "has_attack2": False,
            "card_text": None, "image_url": "https://example.com/x.jpg", "url": "https://example.com/x"}
    rows = [
        {**base, "product_id": 1, "name": "Charizard ex - 199/165", "rarity": "Special Illustration Rare",
         "number": "199/165", "card_type": "Fire", "attacks": "[RR] Burning Darkness (180) flames scorch",
         "printing": "Holofoil", "market_price": 279.37},
        {**base, "product_id": 2, "name": "Pikachu - 025/165", "rarity": "Common", "number": "025/165",
         "card_type": "Lightning", "attacks": "[L] Thunder Shock (20) electric paralysis",
         "printing": "Normal", "market_price": 0.25},
        {**base, "product_id": 2, "name": "Pikachu - 025/165", "rarity": "Common", "number": "025/165",
         "card_type": "Lightning", "attacks": "[L] Thunder Shock (20) electric paralysis",
         "printing": "Reverse Holofoil", "market_price": 0.6},
        {**base, "product_id": 3, "name": "Professor's Research", "rarity": "Uncommon", "number": "189/198",
         "card_type": "Supporter", "stage": None, "hp": np.nan, "retreat_cost": np.nan, "attacks": None,
         "card_text": "Discard your hand and draw 7 cards.", "printing": "Normal", "market_price": 0.15},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path):
    config = load_config()
    config["search"]["index_dir"] = str(tmp_path / "index")
    config["data"]["snapshot_date"] = "2026-01-01"
    return config


def test_card_documents_one_per_product():
    ids, docs, metas = card_documents(_cards())
    assert ids == ["1", "2", "3"]
    pika = metas[1]
    assert pika["printings"] == '{"Normal": 0.25, "Reverse Holofoil": 0.6}'
    assert "Thunder Shock" in docs[1] and "60 HP" in docs[1]
    # Nulls are dropped (Chroma rejects them); text-only cards still get a document.
    assert "hp" not in metas[2] and "draw 7 cards" in docs[2]
    assert all(v is not None for m in metas for v in m.values())


def test_build_and_search(cfg):
    n = build_index(cfg, _cards(), embedder=FakeEmbedder())
    assert n == 3
    search = CardSearch(cfg["search"]["index_dir"], cfg["search"]["collection"], FakeEmbedder())

    top = search.search("electric thunder pikachu", top_k=2)
    assert top[0]["name"] == "Pikachu - 025/165"
    assert top[0]["printings"] == {"Normal": 0.25, "Reverse Holofoil": 0.6}
    assert top[0]["score"] > top[1]["score"]

    assert search.search("discard hand draw cards", top_k=1)[0]["name"] == "Professor's Research"


class BlindEmbedder:
    """Same vector for every text: no semantic signal, so only keyword matching can rank."""

    def encode(self, texts, batch_size=256):
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)


def test_keyword_match_finds_exact_cards(cfg):
    build_index(cfg, _cards(), embedder=BlindEmbedder())
    # Pool of 1: the embedding side alone would return an arbitrary card.
    search = CardSearch(cfg["search"]["index_dir"], cfg["search"]["collection"], BlindEmbedder(),
                        candidate_pool=1, keyword_weight=0.5)
    top = search.search("professor's research", top_k=1)[0]
    assert top["name"] == "Professor's Research"
    assert top["keyword_match"] == 1.0
    # Rare words outweigh common ones: "pikachu" beats the shared "165".
    assert search.search("pikachu 165", top_k=1)[0]["name"] == "Pikachu - 025/165"


def test_rebuild_replaces_old_index(cfg):
    build_index(cfg, _cards(), embedder=FakeEmbedder())
    build_index(cfg, _cards().query("product_id != 3"), embedder=FakeEmbedder())
    search = CardSearch(cfg["search"]["index_dir"], cfg["search"]["collection"], FakeEmbedder())
    assert search.collection.count() == 2


def test_search_result_feeds_the_price_model(cfg):
    build_index(cfg, _cards(), embedder=FakeEmbedder())
    search = CardSearch(cfg["search"]["index_dir"], cfg["search"]["collection"], FakeEmbedder())
    card = search.search("charizard flames", top_k=1)[0]
    printing = next(iter(card["printings"]))
    features = model_input(card, printing)
    assert set(features) == set(RAW_INPUT_COLUMNS)
    assert features["printing"] == "Holofoil" and features["number"] == "199/165"

    class EchoModel:  # stands in for the trained model: checks the frame shape it receives
        def predict(self, frame):
            assert list(frame.columns) == RAW_INPUT_COLUMNS
            return np.array([42.0])

    assert predict_fair_value([features], EchoModel()) == [42.0]
