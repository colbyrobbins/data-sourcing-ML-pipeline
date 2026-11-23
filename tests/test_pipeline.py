"""Offline tests: synthetic TCGCSV files in, trained + tracked model out. No network needed."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline import data as data_mod
from pipeline import storage
from pipeline.config import DEFAULT_CONFIG, load_config, snapshot_filename
from pipeline.data import load_snapshot, split_data
from pipeline.features import RAW_INPUT_COLUMNS, card_features_from_config
from pipeline.ingest import build_table, clean, sha256_of
from pipeline.predict import load_model, predict_fair_value, price_verdict

RARITY_PRICE = {"Common": 0.2, "Uncommon": 0.4, "Rare": 2.0, "Ultra Rare": 20.0, "Special Illustration Rare": 80.0}


def _write_raw(raw_dir, n_groups=4, cards_per_group=60, seed=0):
    """Write groups/products/prices JSON shaped like TCGCSV's responses."""
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    pid = 1000
    for g in range(n_groups):
        gid = 100 + g
        groups.append(
            {
                "groupId": gid,
                "name": f"Set {g}",
                "abbreviation": f"S{g}",
                "isSupplemental": g == 0,
                "publishedOn": f"{2015 + 2 * g}-03-01T00:00:00",
            }
        )
        products, prices = [], []
        # One sealed product without rarity: must be dropped by clean().
        products.append({"productId": pid, "name": f"Set {g} Booster Pack", "extendedData": []})
        prices.append({"productId": pid, "subTypeName": "Normal", "marketPrice": 5.0,
                       "lowPrice": 4.0, "midPrice": 5.0, "highPrice": 9.0})
        pid += 1
        for i in range(cards_per_group):
            rarity = rng.choice(list(RARITY_PRICE))
            name = f"{rng.choice(['Pikachu', 'Charizard ex', 'Bulbasaur', 'Eevee V'])} - {i + 1:03d}/{cards_per_group - 10}"
            products.append(
                {
                    "productId": pid,
                    "name": name,
                    "imageUrl": f"https://example.com/{pid}.jpg",
                    "url": f"https://example.com/{pid}",
                    "extendedData": [
                        {"name": "Rarity", "value": rarity},
                        {"name": "Number", "value": f"{i + 1:03d}/{cards_per_group - 10}"},
                        {"name": "Card Type", "value": str(rng.choice(["Fire", "Water", "Grass"]))},
                        {"name": "Stage", "value": "Basic"},
                        {"name": "HP", "value": str(int(rng.integers(40, 300)))},
                        {"name": "RetreatCost", "value": str(int(rng.integers(0, 4)))},
                        {"name": "Attack 1", "value": "Tackle"},
                    ],
                }
            )
            base = RARITY_PRICE[rarity] * (1 + g * 0.3)
            for printing in ("Normal", "Reverse Holofoil"):
                price = round(float(base * rng.lognormal(0, 0.2)), 2)
                prices.append({"productId": pid, "subTypeName": printing, "marketPrice": price,
                               "lowPrice": price * 0.8, "midPrice": price, "highPrice": price * 2})
            pid += 1
        # A price row with no market price: must be dropped.
        prices.append({"productId": pid - 1, "subTypeName": "Holofoil", "marketPrice": None,
                       "lowPrice": None, "midPrice": None, "highPrice": None})
        (raw_dir / f"products_{gid}.json").write_text(json.dumps(products))
        (raw_dir / f"prices_{gid}.json").write_text(json.dumps(prices))
    (raw_dir / "groups.json").write_text(json.dumps(groups))
    return raw_dir


@pytest.fixture
def cfg(tmp_path):
    """Config pointing every path at tmp_path, with small models for speed."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data"].update(snapshot_date="2026-01-01", snapshot_sha256="",
                       raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed"))
    raw["storage"].update(backend="local", gcs_bucket="")
    raw["tracking"].update(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                           artifact_location=str(tmp_path / "mlruns"))
    raw["output"]["model_dir"] = str(tmp_path / "artifacts")
    raw["model"]["candidates"] = {"hist_gbm": {"max_iter": 50}, "ridge": {"alpha": 1.0}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path), path


@pytest.fixture
def snapshot(cfg, tmp_path):
    """Build a cleaned snapshot parquet from synthetic raw files."""
    config, _ = cfg
    raw_dir = _write_raw(tmp_path / "raw" / "2026-01-01")
    cards = clean(build_table(raw_dir), config)
    out = tmp_path / "processed" / snapshot_filename(config)
    out.parent.mkdir(parents=True)
    cards.to_parquet(out, index=False)
    return out


# --- ingestion ---------------------------------------------------------------


def test_clean_keeps_priced_single_cards(cfg, tmp_path):
    config, _ = cfg
    table = build_table(_write_raw(tmp_path / "raw"))
    cards = clean(table, config)
    assert len(table) == 4 * (1 + 60 * 2 + 1)
    assert len(cards) == 4 * 60 * 2  # sealed product and unpriced row removed
    assert cards["rarity"].notna().all()
    assert cards["market_price"].min() > 0
    assert cards["hp"].dtype.kind == "f"
    assert cards["set_published"].str.match(r"\d{4}-\d{2}-\d{2}$").all()


def test_snapshot_is_deterministic(cfg, tmp_path):
    config, _ = cfg
    raw_dir = _write_raw(tmp_path / "raw")
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    clean(build_table(raw_dir), config).to_parquet(a, index=False)
    clean(build_table(raw_dir), config).to_parquet(b, index=False)
    assert sha256_of(a) == sha256_of(b)


# --- loading -----------------------------------------------------------------


def test_load_snapshot_checks_sha(cfg, snapshot):
    config, _ = cfg
    config["data"]["snapshot_sha256"] = sha256_of(snapshot)
    assert len(load_snapshot(config)) == 4 * 60 * 2
    config["data"]["snapshot_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_snapshot(config)


def test_load_snapshot_downloads_from_gcs(cfg, snapshot, tmp_path, monkeypatch):
    config, _ = cfg
    remote = tmp_path / "remote.parquet"
    snapshot.rename(remote)  # local copy gone -> must download
    config["storage"].update(backend="gcs", gcs_bucket="my-bucket")
    config["data"]["snapshot_sha256"] = sha256_of(remote)
    calls = []

    def fake_download(bucket, blob, local_path):
        calls.append((bucket, blob))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(remote.read_bytes())
        return local_path

    monkeypatch.setattr(storage, "download_file", fake_download)
    assert len(load_snapshot(config)) == 4 * 60 * 2
    assert calls == [("my-bucket", f"snapshots/{snapshot_filename(config)}")]
    load_snapshot(config)  # valid cached copy: no second download
    assert len(calls) == 1


def test_gcs_backend_without_bucket_fails_clearly(cfg):
    config, _ = cfg
    config["storage"]["backend"] = "gcs"
    with pytest.raises(SystemExit, match="gcs_bucket"):
        data_mod.load_snapshot(config)


# --- features and split ----------------------------------------------------------


def test_features_from_raw_fields(cfg):
    config, _ = cfg
    feats = card_features_from_config(config)
    raw = pd.DataFrame(
        [
            {"name": "Charizard ex - 199/165", "rarity": "Special Illustration Rare", "printing": "Holofoil",
             "number": "199/165", "set_published": "2025-01-01", "hp": "330", "has_attack2": True},
            {"name": "Pikachu", "rarity": "Common"},  # most fields missing
        ]
    )
    out = feats.transform(raw)
    assert list(out.columns) == list(feats.get_feature_names_out())
    row = out.iloc[0]
    assert row["is_secret_number"] == 1.0 and row["card_number"] == 199 and row["set_size"] == 165
    assert row["is_charizard"] == 1.0 and row["is_ex"] == 1.0 and row["is_pikachu"] == 0.0
    assert row["set_age_years"] == pytest.approx(1.0, abs=0.01)  # reference date 2026-01-01
    assert out.iloc[1]["card_type"] == "missing" and np.isnan(out.iloc[1]["hp"])


def test_split_keeps_products_together(cfg, snapshot):
    config, _ = cfg
    df = pd.read_parquet(snapshot)
    train, val, test = split_data(df, config)
    assert len(train) + len(val) + len(test) == len(df)
    ids = [set(s["product_id"]) for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert 0.08 < len(test) / len(df) < 0.25 and 0.08 < len(val) / len(df) < 0.25


# --- training and prediction -------------------------------------------------------


def test_train_end_to_end(cfg, snapshot):
    import mlflow

    from pipeline import train

    config, path = cfg
    metadata = train.main(["--config", str(path)])

    assert metadata["model"] in {"hist_gbm", "ridge"}
    assert metadata["input_columns"] == RAW_INPUT_COLUMNS
    assert metadata["test_metrics"]["r2_log"] > 0.5  # price is driven by rarity

    mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[config["tracking"]["experiment_name"]])
    assert len(runs) == 3  # parent + 2 candidates
    parent = runs[runs["tags.mlflow.runName"].str.startswith("train")].iloc[0]
    assert "metrics.test_rmse_log" in runs and not np.isnan(parent["metrics.test_rmse_log"])
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(parent.run_id, "evaluation")}
    assert {"evaluation/pred_vs_actual.png", "evaluation/error_by_rarity.png"} <= artifacts

    model, meta = load_model(config["output"]["model_dir"])
    cheap, pricey = predict_fair_value(
        [{"name": "Bulbasaur", "rarity": "Common", "set_published": "2019-03-01"},
         {"name": "Charizard ex", "rarity": "Special Illustration Rare", "set_published": "2019-03-01"}],
        model,
    )
    assert 0 < cheap < pricey
    assert meta["mlflow_run_id"] == parent.run_id


def test_price_verdict():
    assert price_verdict(50, 100) == "deal"
    assert price_verdict(100, 100) == "fair"
    assert price_verdict(150, 100) == "overpriced"
