"""Load config.yaml and resolve paths relative to the project root."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. Relative paths inside it are resolved against the project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in ("raw_dir", "processed_dir"):
        cfg["data"][key] = str(resolve(cfg["data"][key]))
    cfg["output"]["model_dir"] = str(resolve(cfg["output"]["model_dir"]))
    if "search" in cfg:
        cfg["search"]["index_dir"] = str(resolve(cfg["search"]["index_dir"]))

    tracking = cfg["tracking"]
    uri = tracking["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri[len(prefix):].startswith("/"):
        # Relative SQLite path -> absolute, so runs land in one place from any cwd.
        tracking["tracking_uri"] = prefix + str(resolve(uri[len(prefix):]))
    artifacts = tracking.get("artifact_location") or "mlruns"
    if "://" not in artifacts:
        tracking["artifact_location"] = resolve(artifacts).as_uri()

    if not cfg["features"].get("reference_date"):
        cfg["features"]["reference_date"] = cfg["data"]["snapshot_date"]
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def snapshot_filename(cfg: dict) -> str:
    return f"cards_{cfg['data']['snapshot_date']}.parquet"
