"""Regression metrics and diagnostic plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Chart styling: one series color, recessive gray grid and reference lines.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def regression_metrics(y_true, y_pred) -> dict:
    """Metrics on prices in dollars. rmse_log/r2_log measure error on log(1 + price)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    abs_err = np.abs(y_pred - y_true)
    return {
        "rmse_log": float(np.sqrt(np.mean((lp - lt) ** 2))),
        "r2_log": float(1 - np.sum((lp - lt) ** 2) / np.sum((lt - lt.mean()) ** 2)),
        "mae": float(abs_err.mean()),
        "median_abs_error": float(np.median(abs_err)),
        "median_abs_pct_error": float(np.median(abs_err / y_true)),
    }


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _money(v, _pos=None):
    return f"${v:,.0f}" if v >= 1 else f"${v:.2f}"


def plot_pred_vs_actual(y_true, y_pred, path: Path, title: str) -> Path:
    y_true, y_pred = np.asarray(y_true), np.clip(np.asarray(y_pred), 0.01, None)
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=SURFACE)
    _style(ax)
    ax.scatter(y_true, y_pred, s=6, color=SERIES, alpha=0.25, linewidths=0)
    lo, hi = 0.01, max(y_true.max(), y_pred.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=1.2, label="Perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(_money)
    ax.yaxis.set_major_formatter(_money)
    ax.set_xlabel("Actual market price", color=TEXT_SECONDARY)
    ax.set_ylabel("Predicted fair value", color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_by_rarity(df: pd.DataFrame, y_pred, path: Path, top_n: int = 12) -> Path:
    """Median absolute % error for the most common rarities in the evaluated split."""
    d = df[["rarity", "market_price"]].copy()
    d["pct_err"] = np.abs(np.asarray(y_pred) - d["market_price"]) / d["market_price"]
    top = d["rarity"].value_counts().head(top_n).index
    summary = (
        d[d["rarity"].isin(top)]
        .groupby("rarity")
        .agg(median_pct_err=("pct_err", "median"), n=("pct_err", "size"))
        .sort_values("median_pct_err")
    )

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(summary) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    labels = [f"{r}  (n={n:,})" for r, n in zip(summary.index, summary["n"])]
    ax.barh(labels, summary["median_pct_err"] * 100, color=SERIES, height=0.6)
    ax.set_xlabel("Median absolute error (% of actual price)", color=TEXT_SECONDARY)
    ax.set_title("Prediction error by rarity (test set)", color=TEXT, loc="left", fontsize=12)
    ax.tick_params(axis="y", colors=TEXT, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
