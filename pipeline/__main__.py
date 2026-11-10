"""Run the full pipeline: download -> load -> split -> compare models -> train -> evaluate -> save.

Usage:
    python -m pipeline                              # all features
    python -m pipeline --exclude Health_condition   # leave a feature out
"""

import argparse
import json

import joblib
from sklearn.model_selection import train_test_split

from . import config
from .data import download_dataset, load_data, split_features_target
from .evaluate import evaluate, feature_importance, format_report
from .model import compare_models, train_model


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--exclude", nargs="*", default=[], help="Feature columns to leave out")
    parser.add_argument("--force-download", action="store_true", help="Re-download the dataset")
    parser.add_argument("--out", default=str(config.ARTIFACTS_DIR), help="Output directory")
    args = parser.parse_args(argv)
    exclude = tuple(args.exclude)
    out_dir = config.Path(args.out)

    # 1. Source data
    csv_path = download_dataset(force=args.force_download)
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows, positive rate {(df[config.TARGET] == config.POSITIVE_LABEL).mean():.1%}")

    # 2. Split (stratified, so both sets keep the class balance)
    X, y = split_features_target(df)
    X = X.drop(columns=list(exclude))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    # 3. Compare candidates with CV on the training set only
    comparison = compare_models(X_train, y_train, exclude)
    print("\nCross-validation results:")
    print(comparison[["model", "cv_roc_auc", "cv_f1", "cv_accuracy"]].round(4).to_string(index=False))
    best_name = comparison.loc[0, "model"]

    # 4. Train best model on full training set, evaluate once on the hold-out test set
    model = train_model(best_name, X_train, y_train, exclude)
    metrics = evaluate(model, X_test, y_test)
    importance = feature_importance(model, X_test, y_test)
    print(f"\nBest model: {best_name}")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  test {key}: {metrics[key]:.4f}")

    # 5. Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    summary = {
        "best_model": best_name,
        "excluded_features": list(exclude),
        **{k: v for k, v in metrics.items() if k != "classification_report"},
        "feature_importance": importance.round(6).to_dict(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(
        format_report(comparison, best_name, metrics, importance, exclude, len(X_train), len(X_test))
    )
    print(f"\nSaved model and reports to {out_dir}")
    return summary


if __name__ == "__main__":
    main()
