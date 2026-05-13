# compare_diamond_models.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "big_sweep/sweep_local_20260309_184135/_sweep_checkpoint.csv"
OUTPUT_DIR = "plot_new"
SAVE_DPI = 150
FIGSIZE = (10, 6)

METRICS_TO_PLOT = [
    "r2_test", "rmse_test", "mae_test", "wmapE_test",
    "r2_val",  "rmse_val",  "mae_val",  "wmapE_val"
]

# Lower is better for these
LOWER_IS_BETTER = {"rmse_test", "mae_test", "wmapE_test", "rmse_val", "mae_val", "wmapE_val"}

# Higher is better
HIGHER_IS_BETTER = {"r2_test", "r2_val", "r2_train"}

# ────────────────────────────────────────────────
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading data...")
    df = pd.read_csv(CSV_PATH)

    # Clean column names just in case
    df.columns = df.columns.str.strip().str.lower()

    print(f"Loaded {len(df)} models")

    # 1. Overall comparison bar plots (test set)
    print("Creating test set comparison plots...")
    plot_metric_bars(df, "r2_test",    title="R² (test set) – higher is better",    higher_better=True)
    plot_metric_bars(df, "rmse_test",  title="RMSE (test set) – lower is better",   higher_better=False)
    plot_metric_bars(df, "mae_test",   title="MAE (test set) – lower is better",    higher_better=False)
    plot_metric_bars(df, "wmapE_test", title="wMAPE (test set) – lower is better",  higher_better=False)

    # 2. Best models summary
    print("Finding best models...")
    best = find_best_models(df)
    print("\nBest models by key metrics:")
    print(best.to_string(index=False))

    # 3. Scatter: performance vs complexity
    if "n_features" in df.columns and "param_n_estimators" in df.columns:
        plot_performance_vs_complexity(df)

    # 4. Train vs Test gap (overfitting check)
    plot_train_vs_test_gap(df)

    # 5. Pareto-like plot: R² vs wMAPE
    plot_r2_vs_wmape(df)

    print(f"\nAll plots saved in ./{OUTPUT_DIR}/")

# ────────────────────────────────────────────────
def plot_metric_bars(df, metric, title=None, higher_better=True, top_n=20):
    if metric not in df.columns:
        print(f"Column {metric} not found — skipping")
        return

    # Sort
    ascending = not higher_better
    df_sorted = df.sort_values(metric, ascending=ascending).head(top_n).copy()

    # Add short label
    df_sorted["short_label"] = df_sorted["run_name"] + "_" + df_sorted["combo_id"].astype(str)

    plt.figure(figsize=FIGSIZE)
    ax = sns.barplot(
        data=df_sorted,
        x="short_label",
        y=metric,
        hue="feature_set",
        dodge=False,
        palette="Set2"
    )

    # Highlight best
    best_val = df_sorted[metric].iloc[0]
    ax.axhline(best_val, color="darkred", linestyle="--", alpha=0.7, linewidth=1.2)
    ax.text(0.5, best_val, f"best = {best_val:.4f}", color="darkred", va="bottom", ha="center")

    plt.title(title or metric.replace("_", " ").title())
    plt.ylabel(metric)
    plt.xlabel("Model / Combination")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()

    clean_name = metric.replace("_", "-")
    plt.savefig(f"{OUTPUT_DIR}/{clean_name}-comparison.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

# ────────────────────────────────────────────────
def find_best_models(df):
    best_rows = []

    for metric in ["r2_test", "wmapE_test", "rmse_test", "mae_test"]:
        if metric not in df.columns:
            continue
        ascending = metric in LOWER_IS_BETTER
        best = df.loc[df[metric].idxmin() if ascending else df[metric].idxmax()].copy()
        best["rank_by"] = metric
        best_rows.append(best)

    # Overall best by wMAPE_test (most common business KPI for price prediction)
    if "wmapE_test" in df.columns:
        best_wmape = df.loc[df["wmapE_test"].idxmin()].copy()
        best_wmape["rank_by"] = "best_wmape_test"
        best_rows.append(best_wmape)

    return pd.DataFrame(best_rows)

# ────────────────────────────────────────────────
def plot_performance_vs_complexity(df):
    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        data=df,
        x="n_features",
        y="param_n_estimators",
        size="wmapE_test",
        hue="r2_test",
        palette="viridis_r",
        sizes=(30, 300),
        alpha=0.7,
        edgecolor="none"
    )

    plt.title("Model Complexity vs Performance\n(point size = wMAPE test, color = R² test)")
    plt.xlabel("Number of features")
    plt.ylabel("n_estimators")

    # Annotate a few top models
    top5 = df.nsmallest(5, "wmapE_test")
    for i, row in top5.iterrows():
        plt.text(
            row["n_features"] + 0.1,
            row["param_n_estimators"],
            f"{row['combo_id']}\nwMAPE={row['wmapE_test']:.3f}",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/complexity-vs-performance.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

# ────────────────────────────────────────────────
def plot_train_vs_test_gap(df):
    if "r2_train" not in df.columns or "r2_test" not in df.columns:
        return

    df["r2_gap"] = df["r2_train"] - df["r2_test"]

    plt.figure(figsize=FIGSIZE)
    sns.scatterplot(
        data=df,
        x="wmapE_test",
        y="r2_gap",
        hue="feature_set",
        size="n_features",
        sizes=(40, 220),
        alpha=0.8
    )

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.title("Overfitting Check: R² train – test gap vs wMAPE test")
    plt.xlabel("wMAPE test (lower = better)")
    plt.ylabel("R² gap (train - test) – >0 means overfitting")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/overfitting-r2-gap.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

# ────────────────────────────────────────────────
def plot_r2_vs_wmape(df):
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="wmapE_test",
        y="r2_test",
        hue="feature_set",
        size="n_features",
        sizes=(50, 300),
        alpha=0.75,
        edgecolor="none"
    )

    # Annotate best few models
    best = df.nsmallest(6, "wmapE_test")
    for i, row in best.iterrows():
        plt.text(
            row["wmapE_test"] * 1.02,
            row["r2_test"],
            f"{row['combo_id']}\n{row['wmapE_test']:.3f}",
            fontsize=9,
            va="center"
        )

    plt.title("Test set: R² vs wMAPE (lower wMAPE + higher R² = better)")
    plt.xlabel("wMAPE test")
    plt.ylabel("R² test")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pareto-r2-vs-wmape.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

# ────────────────────────────────────────────────
if __name__ == "__main__":
    main()