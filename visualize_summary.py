#!/usr/bin/env python3
"""Generate compact summary plots from sweep outputs.

This mirrors the core logic in the larger plotting scripts:
- cross-sweep best validation wMAPE by model
- RF feature importance from the all-features sweep
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


BASE = Path(__file__).resolve().parent
SWEEPS = BASE / "big_sweep"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)


def latest_sweep(prefix: str) -> Path:
    matches = sorted(SWEEPS.glob(f"{prefix}_*"))
    if not matches:
        raise FileNotFoundError(f"No sweep directories matching '{prefix}_*' in {SWEEPS}")
    return matches[-1]


def read_sweep_results(prefix: str) -> pd.DataFrame:
    path = latest_sweep(prefix) / "sweep_results.csv"
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"].copy()
    return df


def plot_cross_sweep_best_wmape() -> None:
    sweeps = [
        ("Dimensions", read_sweep_results("sweep_dimensions")),
        ("Qualities", read_sweep_results("sweep_qualities")),
        ("All Features", read_sweep_results("sweep_all_features")),
    ]
    models = [("rfr", "Random Forest"), ("knn", "KNN"), ("elasticnet", "ElasticNet")]
    colors = {"rfr": "#1f77b4", "knn": "#2ca02c", "elasticnet": "#ff7f0e"}

    x = np.arange(len(sweeps))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, (model_key, label) in enumerate(models):
        vals = []
        for _, df in sweeps:
            subset = df[df["model"] == model_key]
            vals.append(subset["wmapE_val"].min() if not subset.empty else np.nan)
        bars = ax.bar(
            x + idx * width,
            vals,
            width=width,
            label=label,
            color=colors[model_key],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, vals):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title("Best Validation wMAPE by Model Across Feature Sweeps")
    ax.set_ylabel("wMAPE (lower is better)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([name for name, _ in sweeps])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "summary_cross_sweep_wmape.png")
    plt.close(fig)


def plot_rf_feature_importance() -> None:
    imp_path = latest_sweep("sweep_all_features") / "feature_importance.csv"
    imp = pd.read_csv(imp_path).sort_values("importance", ascending=False).head(12)
    imp = imp.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(imp["feature"], imp["importance"], color="#4c78a8")
    ax.set_title("Random Forest Feature Importance (All Features Sweep)")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "summary_rf_feature_importance.png")
    plt.close(fig)


def main() -> None:
    plot_cross_sweep_best_wmape()
    plot_rf_feature_importance()
    print(f"Saved plots to {PLOTS}")


if __name__ == "__main__":
    main()
