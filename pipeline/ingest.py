"""Source the data: download TCGplayer prices from TCGCSV, clean them, and publish a snapshot.

Usage:
    python -m pipeline.ingest                 # fetch today's prices, clean, save locally
    python -m pipeline.ingest --upload        # ...and upload the snapshot to GCS
    python -m pipeline.ingest --skip-fetch    # rebuild the snapshot from raw files already on disk

The snapshot is named after the date it was taken. After making a new one, put that
date and the printed SHA-256 into config.yaml so training uses exactly that file.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import storage
from .config import load_config, snapshot_filename

USER_AGENT = "aipi510-pokemon-fair-value/0.1 (student project)"
ATTACK_FIELDS = ("Attack 1", "Attack 2", "Attack 3", "Attack 4")


# ---------------------------------------------------------------------------
# 1. Fetch raw JSON
# ---------------------------------------------------------------------------


def _get_json(session: requests.Session, url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch {url}") from exc
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_raw(cfg: dict, raw_dir: Path) -> Path:
    """Download every set's products and prices into raw_dir. Files already present are skipped."""
    base = f"{cfg['data']['source_url']}/{cfg['data']['category_id']}"
    delay = cfg["data"]["request_delay_seconds"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    groups = _get_json(session, f"{base}/groups")["results"]
    (raw_dir / "groups.json").write_text(json.dumps(groups))
    print(f"Fetching {len(groups)} sets from TCGCSV ...")

    for i, group in enumerate(groups, start=1):
        gid = group["groupId"]
        for kind in ("products", "prices"):
            path = raw_dir / f"{kind}_{gid}.json"
            if path.exists():
                continue
            path.write_text(json.dumps(_get_json(session, f"{base}/{gid}/{kind}")["results"]))
            time.sleep(delay)
        if i % 25 == 0 or i == len(groups):
            print(f"  {i}/{len(groups)} sets")
    return raw_dir


# ---------------------------------------------------------------------------
# 2. Flatten and clean
# ---------------------------------------------------------------------------


def build_table(raw_dir: Path) -> pd.DataFrame:
    """Join products with prices into one row per (product, printing)."""
    groups = json.loads((raw_dir / "groups.json").read_text())
    rows = []
    for group in groups:
        gid = group["groupId"]
        products_path = raw_dir / f"products_{gid}.json"
        prices_path = raw_dir / f"prices_{gid}.json"
        if not (products_path.exists() and prices_path.exists()):
            continue
        products = {p["productId"]: p for p in json.loads(products_path.read_text())}
        for price in json.loads(prices_path.read_text()):
            product = products.get(price["productId"])
            if product is None:
                continue
            ext = {e["name"]: e["value"] for e in product.get("extendedData", [])}
            rows.append(
                {
                    "product_id": product["productId"],
                    "name": product["name"],
                    "group_id": gid,
                    "set_name": group["name"],
                    "set_abbr": group.get("abbreviation"),
                    "set_published": group.get("publishedOn"),
                    "is_supplemental": group.get("isSupplemental", False),
                    "printing": price["subTypeName"],
                    "market_price": price["marketPrice"],
                    "low_price": price["lowPrice"],
                    "mid_price": price["midPrice"],
                    "high_price": price["highPrice"],
                    "rarity": ext.get("Rarity"),
                    "number": ext.get("Number"),
                    "card_type": ext.get("Card Type"),
                    "stage": ext.get("Stage"),
                    "hp": ext.get("HP"),
                    "retreat_cost": ext.get("RetreatCost"),
                    "weakness": ext.get("Weakness"),
                    "resistance": ext.get("Resistance"),
                    "has_attack2": "Attack 2" in ext,
                    # Free text, used by the semantic search index (not by the price model).
                    "attacks": " | ".join(ext[a] for a in ATTACK_FIELDS if ext.get(a)) or None,
                    "card_text": ext.get("CardText"),
                    "image_url": product.get("imageUrl"),
                    "url": product.get("url"),
                }
            )
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep priced, collectible single cards and normalize their fields."""
    data_cfg = cfg["data"]
    out = df.copy()

    # Sealed products (packs, boxes) have no rarity; keep single cards only.
    out = out[out["rarity"].notna()]
    out = out[~out["rarity"].isin(data_cfg["drop_rarities"])]
    out = out[out["market_price"].notna() & (out["market_price"] >= data_cfg["min_price"])]

    for col in ("name", "rarity", "printing", "card_type", "stage", "number", "set_name"):
        out[col] = out[col].astype("string").str.strip()
    for col in ("attacks", "card_text"):
        out[col] = _plain_text(out[col])

    # Always float so missing values (trainers have no HP) are representable.
    out["hp"] = pd.to_numeric(out["hp"], errors="coerce").astype(float)
    out["retreat_cost"] = pd.to_numeric(out["retreat_cost"], errors="coerce").astype(float)
    out["set_published"] = pd.to_datetime(
        out["set_published"], utc=True, format="mixed", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    out["is_supplemental"] = out["is_supplemental"].astype(bool)
    out["has_attack2"] = out["has_attack2"].astype(bool)

    out = out.drop_duplicates(subset=["product_id", "printing"]).sort_values(
        ["group_id", "product_id", "printing"]
    )
    return out.reset_index(drop=True)


def _plain_text(s: pd.Series) -> pd.Series:
    """Strip HTML tags and collapse whitespace in TCGplayer's card text."""
    return (
        s.astype("string")
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--date", help="Snapshot date label (default: today, UTC)")
    parser.add_argument("--skip-fetch", action="store_true", help="Use raw files already on disk")
    parser.add_argument("--upload", action="store_true", help="Upload the snapshot to GCS")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cfg["data"]["snapshot_date"] = date
    raw_dir = Path(cfg["data"]["raw_dir"]) / date

    if not args.skip_fetch:
        fetch_raw(cfg, raw_dir)
    table = build_table(raw_dir)
    cards = clean(table, cfg)
    print(f"Raw rows: {len(table)} -> cleaned card rows: {len(cards)}")

    out_path = Path(cfg["data"]["processed_dir"]) / snapshot_filename(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cards.to_parquet(out_path, index=False)
    digest = sha256_of(out_path)
    print(f"Saved {out_path}\n  sha256: {digest}")

    if args.upload:
        bucket = cfg["storage"]["gcs_bucket"]
        if not bucket:
            raise SystemExit("Set storage.gcs_bucket in config.yaml before using --upload.")
        uri = storage.upload_file(out_path, bucket, f"{cfg['storage']['gcs_prefix']}/{out_path.name}")
        print(f"Uploaded to {uri}")

    print(
        "\nTo train on this snapshot, set in config.yaml:\n"
        f'  data.snapshot_date: "{date}"\n'
        f'  data.snapshot_sha256: "{digest}"'
    )
    return out_path


if __name__ == "__main__":
    main()
