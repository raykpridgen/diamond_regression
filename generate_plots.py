#!/usr/bin/env python3
"""Generate all report figures from sweep CSV data."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
})

BASE = Path(__file__).parent
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

dim_csv = BASE / "big_sweep/sweep_dimensions_20260309_044717/sweep_results.csv"
qual_csv = BASE / "big_sweep/sweep_qualities_20260309_051409/sweep_results.csv"
all_csv = BASE / "big_sweep/sweep_all_features_20260309_131859/sweep_results.csv"

dim = pd.read_csv(dim_csv)
qual = pd.read_csv(qual_csv)
allf = pd.read_csv(all_csv)

for df in (dim, qual, allf):
    df.loc[:, "status"] = df["status"].fillna("ok")

dim_ok = dim[dim["status"] == "ok"].copy()
qual_ok = qual[qual["status"] == "ok"].copy()
allf_ok = allf[allf["status"] == "ok"].copy()


# =========================================================================
# 1. Cross-sweep best wMAPE comparison (grouped bar chart)
# =========================================================================
def plot_cross_sweep_wmape():
    sweeps = ["Dimensions\n(x, y, z)", "Qualities\n(carat, cut, ...)", "All Features\n(9 total)"]
    models = ["rfr", "knn", "elasticnet"]
    model_labels = ["Random Forest", "KNN", "ElasticNet"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    best = {}
    for name, df_ok in [("dim", dim_ok), ("qual", qual_ok), ("all", allf_ok)]:
        for m in models:
            sub = df_ok[df_ok["model"] == m]
            best[(name, m)] = sub["wmapE_val"].min() if not sub.empty else np.nan

    x = np.arange(len(sweeps))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (m, ml, c) in enumerate(zip(models, model_labels, colors)):
        vals = [best[("dim", m)], best[("qual", m)], best[("all", m)]]
        bars = ax.bar(x + i * width, vals, width, label=ml, color=c, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Best Validation wMAPE (lower is better)")
    ax.set_title("Best wMAPE by Model Type Across Feature Sweeps")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sweeps)
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "cross_sweep_wmape_comparison.png")
    plt.close()

plot_cross_sweep_wmape()
print("  [1/10] cross_sweep_wmape_comparison.png")


# =========================================================================
# 2. Cross-sweep best R² comparison
# =========================================================================
def plot_cross_sweep_r2():
    sweeps = ["Dimensions\n(x, y, z)", "Qualities\n(carat, cut, ...)", "All Features\n(9 total)"]
    models = ["rfr", "knn", "elasticnet"]
    model_labels = ["Random Forest", "KNN", "ElasticNet"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    best = {}
    for name, df_ok in [("dim", dim_ok), ("qual", qual_ok), ("all", allf_ok)]:
        for m in models:
            sub = df_ok[df_ok["model"] == m]
            best[(name, m)] = sub["r2_val"].max() if not sub.empty else np.nan

    x = np.arange(len(sweeps))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (m, ml, c) in enumerate(zip(models, model_labels, colors)):
        vals = [best[("dim", m)], best[("qual", m)], best[("all", m)]]
        bars = ax.bar(x + i * width, vals, width, label=ml, color=c, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Best Validation R² (higher is better)")
    ax.set_title("Best R² by Model Type Across Feature Sweeps")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sweeps)
    ax.legend(loc="lower right")
    ax.set_ylim(0.65, 1.02)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "cross_sweep_r2_comparison.png")
    plt.close()

plot_cross_sweep_r2()
print("  [2/10] cross_sweep_r2_comparison.png")


# =========================================================================
# 3. wMAPE distribution box plots across sweeps, per model
# =========================================================================
def plot_wmape_boxplots():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    sweep_data = [
        ("Dimensions (x, y, z)", dim_ok),
        ("Qualities (carat, cut, ...)", qual_ok),
        ("All Features (9)", allf_ok),
    ]
    colors = {"rfr": "#2196F3", "knn": "#4CAF50", "elasticnet": "#FF9800"}
    model_labels = {"rfr": "RF", "knn": "KNN", "elasticnet": "ElasticNet"}

    for ax, (title, df_ok) in zip(axes, sweep_data):
        data, labels, cols = [], [], []
        for m in ["rfr", "knn", "elasticnet"]:
            sub = df_ok[df_ok["model"] == m]["wmapE_val"].dropna()
            if not sub.empty:
                data.append(sub.values)
                labels.append(model_labels[m])
                cols.append(colors[m])

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(markersize=3, alpha=0.4))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel("Validation wMAPE")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Validation wMAPE Distribution by Model and Feature Set", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "wmape_boxplots_by_sweep.png")
    plt.close()

plot_wmape_boxplots()
print("  [3/10] wmape_boxplots_by_sweep.png")


# =========================================================================
# 4. RF hyperparameter heatmaps (n_estimators x max_depth) for each sweep
# =========================================================================
def plot_rf_heatmaps():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sweep_data = [
        ("Dimensions", dim_ok),
        ("Qualities", qual_ok),
        ("All Features", allf_ok),
    ]

    for ax, (title, df_ok) in zip(axes, sweep_data):
        rfr = df_ok[(df_ok["model"] == "rfr") & (df_ok["feature_set"] == "all")].copy()
        if rfr.empty or "param_n_estimators" not in rfr.columns:
            ax.set_title(f"{title}: no RF data")
            continue

        rfr["param_max_depth"] = rfr["param_max_depth"].astype(str)
        pivot = (rfr.groupby(["param_n_estimators", "param_max_depth"])["wmapE_val"]
                 .min().unstack())

        depth_order = []
        for c in sorted(pivot.columns, key=lambda x: float(x) if x not in ("None", "nan") else 1e9):
            depth_order.append(c)
        pivot = pivot[depth_order]

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="wMAPE", fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c if c != "None" else "None\n(unlimited)" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(v)) for v in pivot.index], fontsize=9)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("n_estimators")
        ax.set_title(f"RF wMAPE: {title}")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7,
                            color="white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black")

    plt.tight_layout()
    plt.savefig(PLOTS / "rf_heatmaps_all_sweeps.png")
    plt.close()

plot_rf_heatmaps()
print("  [4/10] rf_heatmaps_all_sweeps.png")


# =========================================================================
# 5. Feature ablation importance (all_features sweep)
# =========================================================================
def plot_ablation():
    abl = pd.read_csv(BASE / "big_sweep/sweep_all_features_20260309_131859/ablation_importance.csv")
    abl = abl.sort_values("wmapE_delta", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c" if d > 0.005 else "#f39c12" if d > 0.001 else "#2ecc71"
              for d in abl["wmapE_delta"]]
    ax1.barh(abl["feature"], abl["wmapE_delta"], color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("wMAPE increase when feature removed")
    ax1.set_title("Feature Importance via Ablation (wMAPE)")
    ax1.axvline(0, color="black", linewidth=0.5)
    for i, (_, row) in enumerate(abl.iterrows()):
        ax1.text(row["wmapE_delta"] + 0.001, i, f"{row['wmapE_delta']:.4f}",
                 va="center", fontsize=8)
    ax1.grid(axis="x", alpha=0.3)

    abl_r2 = abl.sort_values("r2_delta", ascending=True)
    colors2 = ["#e74c3c" if d > 0.01 else "#f39c12" if d > 0.001 else "#2ecc71"
               for d in abl_r2["r2_delta"]]
    ax2.barh(abl_r2["feature"], abl_r2["r2_delta"], color=colors2, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("R² decrease when feature removed")
    ax2.set_title("Feature Importance via Ablation (R²)")
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "ablation_importance.png")
    plt.close()

plot_ablation()
print("  [5/10] ablation_importance.png")


# =========================================================================
# 6. RF feature importance comparison (qualities vs all_features)
# =========================================================================
def plot_feature_importance_comparison():
    qual_imp = pd.read_csv(BASE / "big_sweep/sweep_qualities_20260309_051409/feature_importance.csv")
    all_imp = pd.read_csv(BASE / "big_sweep/sweep_all_features_20260309_131859/feature_importance.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    top_q = qual_imp.head(10).sort_values("importance", ascending=True)
    ax1.barh(top_q["feature"], top_q["importance"], color="#2196F3", edgecolor="white")
    ax1.set_title("Qualities Sweep: Top 10 RF Importances")
    ax1.set_xlabel("Importance")
    ax1.grid(axis="x", alpha=0.3)

    top_a = all_imp.head(10).sort_values("importance", ascending=True)
    ax2.barh(top_a["feature"], top_a["importance"], color="#4CAF50", edgecolor="white")
    ax2.set_title("All Features Sweep: Top 10 RF Importances")
    ax2.set_xlabel("Importance")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "feature_importance_comparison.png")
    plt.close()

plot_feature_importance_comparison()
print("  [6/10] feature_importance_comparison.png")


# =========================================================================
# 7. KNN: n_neighbors vs wMAPE across sweeps
# =========================================================================
def plot_knn_neighbors():
    fig, ax = plt.subplots(figsize=(10, 6))
    sweep_data = [
        ("Dimensions", dim_ok, "#2196F3", "o"),
        ("Qualities", qual_ok, "#4CAF50", "s"),
        ("All Features", allf_ok, "#FF9800", "^"),
    ]

    for label, df_ok, color, marker in sweep_data:
        knn = df_ok[(df_ok["model"] == "knn") & (df_ok["feature_set"] == "all")].copy()
        if knn.empty:
            continue
        grouped = knn.groupby("param_n_neighbors")["wmapE_val"].min().sort_index()
        ax.plot(grouped.index, grouped.values, marker=marker, label=label,
                color=color, linewidth=2, markersize=8)

    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("Best Validation wMAPE")
    ax.set_title("KNN: Effect of n_neighbors on wMAPE (all features within each sweep)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "knn_neighbors_vs_wmape.png")
    plt.close()

plot_knn_neighbors()
print("  [7/10] knn_neighbors_vs_wmape.png")


# =========================================================================
# 8. ElasticNet: alpha vs wMAPE with l1_ratio hue
# =========================================================================
def plot_elasticnet_alpha():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sweep_data = [
        ("Dimensions", dim_ok),
        ("Qualities", qual_ok),
        ("All Features", allf_ok),
    ]
    cmap = plt.cm.viridis

    for ax, (title, df_ok) in zip(axes, sweep_data):
        en = df_ok[(df_ok["model"] == "elasticnet") & (df_ok["feature_set"] == "all")].copy()
        if en.empty:
            ax.set_title(f"{title}: no EN data")
            continue

        l1_vals = sorted(en["param_l1_ratio"].unique())
        norm = plt.Normalize(min(l1_vals), max(l1_vals))

        for l1 in l1_vals:
            sub = en[en["param_l1_ratio"] == l1].sort_values("param_alpha")
            ax.plot(sub["param_alpha"], sub["wmapE_val"],
                    marker="o", label=f"l1={l1:.2f}", color=cmap(norm(l1)), linewidth=1.5)

        ax.set_xlabel("alpha")
        ax.set_ylabel("wMAPE (val)")
        ax.set_title(f"ElasticNet: {title}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "elasticnet_alpha_sweep.png")
    plt.close()

plot_elasticnet_alpha()
print("  [8/10] elasticnet_alpha_sweep.png")


# =========================================================================
# 9. Train vs Val vs Test metric comparison for best models
# =========================================================================
def plot_train_val_test():
    # Read best RF from each sweep's best_per_model.csv
    sweep_dirs = [
        ("Dimensions\n(RF)", BASE / "big_sweep/sweep_dimensions_20260309_044717"),
        ("Qualities\n(RF)", BASE / "big_sweep/sweep_qualities_20260309_051409"),
        ("All Features\n(RF)", BASE / "big_sweep/sweep_all_features_20260309_131859"),
    ]
    best_data = {}
    for label, sweep_dir in sweep_dirs:
        bpm = pd.read_csv(sweep_dir / "best_per_model.csv")
        rfr = bpm[bpm["model"] == "rfr"].iloc[0]
        best_data[label] = {
            "train": {"r2": rfr["r2_train"], "wmapE": rfr["wmapE_train"]},
            "val":   {"r2": rfr["r2_val"], "wmapE": rfr["wmapE_val"]},
            "test":  {"r2": rfr["r2_test"], "wmapE": rfr["wmapE_test"]},
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(best_data))
    width = 0.25
    split_colors = {"train": "#66BB6A", "val": "#42A5F5", "test": "#EF5350"}

    for i, split in enumerate(["train", "val", "test"]):
        vals = [best_data[k][split]["wmapE"] for k in best_data]
        bars = ax1.bar(x + i * width, vals, width, label=split.capitalize(),
                       color=split_colors[split], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_ylabel("wMAPE")
    ax1.set_title("Best RF: Train/Val/Test wMAPE")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(best_data.keys())
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for i, split in enumerate(["train", "val", "test"]):
        vals = [best_data[k][split]["r2"] for k in best_data]
        bars = ax2.bar(x + i * width, vals, width, label=split.capitalize(),
                       color=split_colors[split], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_ylabel("R²")
    ax2.set_title("Best RF: Train/Val/Test R²")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(best_data.keys())
    ax2.legend(loc="lower left")
    ax2.set_ylim(0.8, 1.02)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "train_val_test_comparison.png")
    plt.close()

plot_train_val_test()
print("  [9/10] train_val_test_comparison.png")


# =========================================================================
# 10. RF: min_samples_leaf sensitivity (all_features sweep, full feature set)
# =========================================================================
def plot_rf_min_samples_leaf():
    fig, ax = plt.subplots(figsize=(10, 6))
    sweep_data = [
        ("Dimensions", dim_ok, "#2196F3"),
        ("Qualities", qual_ok, "#4CAF50"),
        ("All Features", allf_ok, "#FF9800"),
    ]

    for label, df_ok, color in sweep_data:
        rfr = df_ok[(df_ok["model"] == "rfr") & (df_ok["feature_set"] == "all")].copy()
        if rfr.empty:
            continue
        grouped = rfr.groupby("param_min_samples_leaf")["wmapE_val"].min().sort_index()
        ax.plot(grouped.index, grouped.values, marker="o", label=label,
                color=color, linewidth=2, markersize=8)

    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("Best Validation wMAPE")
    ax.set_title("Random Forest: Effect of min_samples_leaf on wMAPE")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_min_samples_leaf.png")
    plt.close()

plot_rf_min_samples_leaf()
print("  [10/10] rf_min_samples_leaf.png")

print(f"\nAll plots saved to {PLOTS}/")
