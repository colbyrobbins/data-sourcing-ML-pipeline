"""Train the fair-value model and track the run in MLflow.

Usage:
    python -m pipeline.train                  # uses config.yaml
    python -m pipeline.train --config other.yaml

Steps: load snapshot (GCS or local) -> train/val/test split -> fit every candidate on
train and compare on validation -> refit the best on train + val -> score once on test
-> log params, metrics, plots and the model to MLflow -> save the model for serving.
"""

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn

from .config import ROOT, load_config, snapshot_filename
from .data import load_snapshot, split_data
from .evaluate import plot_error_by_rarity, plot_pred_vs_actual, regression_metrics
from .features import RAW_INPUT_COLUMNS
from .model import build_model


def _git_commit() -> str | None:
    if os.environ.get("GIT_COMMIT"):  # set at Docker build time (the image has no .git)
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _setup_tracking(tracking: dict) -> None:
    uri = tracking["tracking_uri"]
    if uri.startswith("sqlite:///"):
        Path(uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    name = tracking["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=tracking["artifact_location"])
    mlflow.set_experiment(name)


def _xy(df: pd.DataFrame, target: str):
    return df[RAW_INPUT_COLUMNS], df[target]


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target = cfg["model"]["target"]
    seed = cfg["split"]["random_state"]
    select_by = cfg["model"]["selection_metric"]

    df = load_snapshot(cfg)
    train, val, test = split_data(df, cfg)
    print(f"Rows: {len(df):,} | train {len(train):,} | val {len(val):,} | test {len(test):,}")

    _setup_tracking(cfg["tracking"])

    with mlflow.start_run(run_name=f"train-{cfg['data']['snapshot_date']}") as run:
        mlflow.set_tags({"git_commit": _git_commit() or "unknown", "stage": "train"})
        mlflow.log_params(
            {
                "snapshot_date": cfg["data"]["snapshot_date"],
                "snapshot_file": snapshot_filename(cfg),
                "snapshot_sha256": cfg["data"].get("snapshot_sha256") or "unchecked",
                "storage_backend": cfg["storage"]["backend"],
                "reference_date": cfg["features"]["reference_date"],
                "test_size": cfg["split"]["test_size"],
                "val_size": cfg["split"]["val_size"],
                "random_state": seed,
                "n_rows": len(df),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "selection_metric": select_by,
            }
        )
        if args.config:
            mlflow.log_artifact(args.config)
        else:
            mlflow.log_artifact(str(ROOT / "config.yaml"))

        # 1. Compare candidates on the validation split.
        X_train, y_train = _xy(train, target)
        X_val, y_val = _xy(val, target)
        results = {}
        for name, params in cfg["model"]["candidates"].items():
            with mlflow.start_run(run_name=name, nested=True):
                model = build_model(name, params or {}, cfg, random_state=seed)
                model.fit(X_train, y_train)
                metrics = regression_metrics(y_val, model.predict(X_val))
                mlflow.log_params({"model": name, **(params or {})})
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                results[name] = metrics
                print(f"  {name:14s} val rmse_log={metrics['rmse_log']:.3f}  r2_log={metrics['r2_log']:.3f}")

        best = min(results, key=lambda n: results[n][select_by])
        best_params = cfg["model"]["candidates"][best] or {}
        print(f"Best on validation: {best}")

        # 2. Refit the best model on train + val and score it once on test.
        trainval = pd.concat([train, val], ignore_index=True)
        X_tv, y_tv = _xy(trainval, target)
        X_test, y_test = _xy(test, target)
        final = build_model(best, best_params, cfg, random_state=seed)
        final.fit(X_tv, y_tv)
        test_pred = final.predict(X_test)
        test_metrics = regression_metrics(y_test, test_pred)

        mlflow.log_params({"best_model": best, **{f"best_{k}": v for k, v in best_params.items()}})
        mlflow.log_metrics({f"val_{k}": v for k, v in results[best].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # 3. Plots and comparison table.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_pred_vs_actual(y_test, test_pred, tmp / "pred_vs_actual.png", f"Test set: {best}")
            plot_error_by_rarity(test, test_pred, tmp / "error_by_rarity.png")
            pd.DataFrame(results).T.rename_axis("model").to_csv(tmp / "candidate_comparison.csv")
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        # 4. Model: log to MLflow and save a copy for the API.
        input_example = X_test.head(3).astype(object).where(X_test.head(3).notna(), None)
        mlflow.sklearn.log_model(
            final,
            name="model",
            input_example=input_example,
            # The model contains our own CardFeatures step, so use pickle (same trust
            # model as the joblib copy) and ship the pipeline package with it.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            code_paths=[str(ROOT / "pipeline")],
        )

        metadata = {
            "model": best,
            "params": best_params,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "snapshot_date": cfg["data"]["snapshot_date"],
            "snapshot_sha256": cfg["data"].get("snapshot_sha256") or None,
            "reference_date": cfg["features"]["reference_date"],
            "input_columns": RAW_INPUT_COLUMNS,
            "val_metrics": results[best],
            "test_metrics": test_metrics,
            "mlflow_run_id": run.info.run_id,
            "git_commit": _git_commit(),
            "sklearn_version": sklearn.__version__,
        }
        model_dir = Path(cfg["output"]["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, model_dir / "model.joblib")
        (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(model_dir / "model_metadata.json"))

    print("\nTest set:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved {model_dir / 'model.joblib'} (MLflow run {run.info.run_id})")
    return metadata


if __name__ == "__main__":
    main()
