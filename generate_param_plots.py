#!/usr/bin/env python3
"""Generate detailed parameter-vs-wMAPE plots for heuristic guidance."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

dim = pd.read_csv(BASE / "big_sweep/sweep_dimensions_20260309_044717/sweep_results.csv")
qual = pd.read_csv(BASE / "big_sweep/sweep_qualities_20260309_051409/sweep_results.csv")
allf = pd.read_csv(BASE / "big_sweep/sweep_all_features_20260309_131859/sweep_results.csv")

SWEEP_SETS = [
    ("Dimensions (x,y,z)", dim),
    ("Qualities (carat,cut,...)", qual),
    ("All Features (9)", allf),
]
SWEEP_COLORS = ["#2196F3", "#4CAF50", "#FF9800"]

def get_rfr_full(df):
    return df[(df["model"] == "rfr") & (df["status"] == "ok") & (df["feature_set"] == "all")].copy()

def depth_label(v):
    if pd.isna(v):
        return "None"
    return str(int(v))


# =========================================================================
# 1. n_estimators vs wMAPE — one line per min_samples_leaf, panels per sweep
#    (marginalised over max_depth by taking the min)
# =========================================================================
def plot_nestimators_by_leaf():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    leaf_colors = {1: "#E53935", 2: "#1E88E5", 4: "#43A047", 8: "#FB8C00", 16: "#8E24AA"}

    for ax, (title, df), _ in zip(axes, SWEEP_SETS, SWEEP_COLORS):
        rfr = get_rfr_full(df)
        for leaf in sorted(rfr["param_min_samples_leaf"].unique()):
            sub = rfr[rfr["param_min_samples_leaf"] == leaf]
            grouped = sub.groupby("param_n_estimators")["wmapE_val"].min().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o", linewidth=2, markersize=7,
                    label=f"leaf={int(leaf)}", color=leaf_colors.get(int(leaf), "gray"))
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("Best Val wMAPE (across max_depth)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted(rfr["param_n_estimators"].unique()))

    fig.suptitle("RF: n_estimators vs wMAPE, grouped by min_samples_leaf\n(best across max_depth)", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS / "param_nestimators_by_leaf.png")
    plt.close()

plot_nestimators_by_leaf()
print("  [1/8] param_nestimators_by_leaf.png")


# =========================================================================
# 2. max_depth vs wMAPE — one line per min_samples_leaf, panels per sweep
#    (marginalised over n_estimators by taking the min)
# =========================================================================
def plot_maxdepth_by_leaf():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    leaf_colors = {1: "#E53935", 2: "#1E88E5", 4: "#43A047", 8: "#FB8C00", 16: "#8E24AA"}

    for ax, (title, df), _ in zip(axes, SWEEP_SETS, SWEEP_COLORS):
        rfr = get_rfr_full(df)
        rfr["depth_str"] = rfr["param_max_depth"].apply(depth_label)
        depth_order = ["5", "10", "20", "30", "None"]
        depth_x = {d: i for i, d in enumerate(depth_order)}

        for leaf in sorted(rfr["param_min_samples_leaf"].unique()):
            sub = rfr[rfr["param_min_samples_leaf"] == leaf]
            grouped = sub.groupby("depth_str")["wmapE_val"].min()
            xs = [depth_x[d] for d in depth_order if d in grouped.index]
            ys = [grouped[d] for d in depth_order if d in grouped.index]
            ax.plot(xs, ys, marker="s", linewidth=2, markersize=7,
                    label=f"leaf={int(leaf)}", color=leaf_colors.get(int(leaf), "gray"))

        ax.set_xlabel("max_depth")
        ax.set_ylabel("Best Val wMAPE (across n_estimators)")
        ax.set_title(title)
        ax.set_xticks(range(len(depth_order)))
        ax.set_xticklabels(depth_order)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("RF: max_depth vs wMAPE, grouped by min_samples_leaf\n(best across n_estimators)", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS / "param_maxdepth_by_leaf.png")
    plt.close()

plot_maxdepth_by_leaf()
print("  [2/8] param_maxdepth_by_leaf.png")


# =========================================================================
# 3. n_estimators vs wMAPE — one line per max_depth, panels per sweep
#    (marginalised over min_samples_leaf by taking the min)
# =========================================================================
def plot_nestimators_by_depth():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    depth_colors = {"5": "#E53935", "10": "#1E88E5", "20": "#43A047", "30": "#FB8C00", "None": "#8E24AA"}

    for ax, (title, df), _ in zip(axes, SWEEP_SETS, SWEEP_COLORS):
        rfr = get_rfr_full(df)
        rfr["depth_str"] = rfr["param_max_depth"].apply(depth_label)

        for d_str in ["5", "10", "20", "30", "None"]:
            sub = rfr[rfr["depth_str"] == d_str]
            if sub.empty:
                continue
            grouped = sub.groupby("param_n_estimators")["wmapE_val"].min().sort_index()
            label = f"depth={d_str}" if d_str != "None" else "depth=None (unlimited)"
            ax.plot(grouped.index, grouped.values, marker="o", linewidth=2, markersize=7,
                    label=label, color=depth_colors[d_str])

        ax.set_xlabel("n_estimators")
        ax.set_ylabel("Best Val wMAPE (across leaf)")
        ax.set_title(title)
        ax.set_xticks(sorted(rfr["param_n_estimators"].unique()))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("RF: n_estimators vs wMAPE, grouped by max_depth\n(best across min_samples_leaf)", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS / "param_nestimators_by_depth.png")
    plt.close()

plot_nestimators_by_depth()
print("  [3/8] param_nestimators_by_depth.png")


# =========================================================================
# 4. 3-way RF heatmap: n_estimators x min_samples_leaf, one panel per
#    max_depth, for the all-features sweep (the richest one)
# =========================================================================
def plot_rf_3way_heatmap():
    rfr = get_rfr_full(allf)
    rfr["depth_str"] = rfr["param_max_depth"].apply(depth_label)
    depths = ["5", "10", "20", "30", "None"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    vmin = rfr["wmapE_val"].min()
    vmax = min(rfr["wmapE_val"].max(), 0.18)

    for ax, d in zip(axes, depths):
        sub = rfr[rfr["depth_str"] == d]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index="param_n_estimators", columns="param_min_samples_leaf",
                                values="wmapE_val", aggfunc="min")
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot[sorted(pivot.columns)]

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(v)) for v in pivot.index], fontsize=9)
        ax.set_xlabel("min_samples_leaf")
        if d == "5":
            ax.set_ylabel("n_estimators")
        label = d if d != "None" else "unlimited"
        ax.set_title(f"depth={label}", fontsize=11)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if val > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=axes, label="Val wMAPE", shrink=0.8, pad=0.02)
    fig.suptitle("RF All Features: n_estimators x min_samples_leaf heatmaps by max_depth\n(lower/greener = better)", fontsize=13, y=1.05)
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_3way_heatmap_allfeatures.png")
    plt.close()

plot_rf_3way_heatmap()
print("  [4/8] rf_3way_heatmap_allfeatures.png")


# =========================================================================
# 5. Scatter: every RF combo colored by min_samples_leaf
#    x = n_estimators (jittered), y = wMAPE, marker size = max_depth
# =========================================================================
def plot_rf_scatter_all():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    leaf_colors = {1: "#E53935", 2: "#1E88E5", 4: "#43A047", 8: "#FB8C00", 16: "#8E24AA"}
    depth_sizes = {5: 30, 10: 60, 20: 100, 30: 140, "None": 180}

    for ax, (title, df), _ in zip(axes, SWEEP_SETS, SWEEP_COLORS):
        rfr = get_rfr_full(df)
        rfr["depth_str"] = rfr["param_max_depth"].apply(depth_label)

        np.random.seed(42)
        jitter = np.random.uniform(-15, 15, len(rfr))
        for _, row in rfr.iterrows():
            leaf = int(row["param_min_samples_leaf"])
            d_str = depth_label(row["param_max_depth"])
            sz = depth_sizes.get(int(row["param_max_depth"]) if not pd.isna(row["param_max_depth"]) else "None", 80)
            c = leaf_colors.get(leaf, "gray")
            ax.scatter(row["param_n_estimators"] + jitter[int(row.name) % len(jitter)],
                       row["wmapE_val"], s=sz, c=c, alpha=0.6, edgecolors="white", linewidth=0.3)

        ax.set_xlabel("n_estimators (jittered)")
        ax.set_ylabel("Val wMAPE")
        ax.set_title(title)
        ax.grid(alpha=0.3)

        leaf_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=leaf_colors[l],
                               markersize=8, label=f"leaf={l}") for l in sorted(leaf_colors)]
        depth_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                                markersize=int(np.sqrt(s)), label=f"depth={d}")
                         for d, s in sorted(depth_sizes.items(), key=lambda x: str(x[0]))]
        ax.legend(handles=leaf_handles + depth_handles, fontsize=7, ncol=2, loc="upper right")

    fig.suptitle("RF: Every combo — n_estimators vs wMAPE\n(color=min_samples_leaf, size=max_depth)", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_scatter_all_combos.png")
    plt.close()

plot_rf_scatter_all()
print("  [5/8] rf_scatter_all_combos.png")


# =========================================================================
# 6. Parallel coordinates style: all RF params → wMAPE for all-features sweep
#    Shows how the top-10 best combos differ from the bottom-10
# =========================================================================
def plot_rf_top_vs_bottom():
    rfr = get_rfr_full(allf).copy()
    rfr["depth_num"] = rfr["param_max_depth"].fillna(999)

    top10 = rfr.nsmallest(10, "wmapE_val")
    bot10 = rfr.nlargest(10, "wmapE_val")

    params = ["param_n_estimators", "param_min_samples_leaf", "depth_num"]
    param_labels = ["n_estimators", "min_samples_leaf", "max_depth"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, param, plabel in zip(axes, params, param_labels):
        vals_top = top10.groupby(param)["wmapE_val"].mean()
        vals_bot = bot10.groupby(param)["wmapE_val"].mean()

        if param == "depth_num":
            top_counts = top10[param].value_counts().sort_index()
            bot_counts = bot10[param].value_counts().sort_index()
            all_vals = sorted(set(top_counts.index) | set(bot_counts.index))
            xlabels = [str(int(v)) if v != 999 else "None" for v in all_vals]

            top_cts = [top_counts.get(v, 0) for v in all_vals]
            bot_cts = [bot_counts.get(v, 0) for v in all_vals]

            x = np.arange(len(all_vals))
            w = 0.35
            ax.bar(x - w/2, top_cts, w, label="Top-10 (best)", color="#43A047", alpha=0.8)
            ax.bar(x + w/2, bot_cts, w, label="Bottom-10 (worst)", color="#E53935", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            ax.set_ylabel("Count in group")
        else:
            top_counts = top10[param].value_counts().sort_index()
            bot_counts = bot10[param].value_counts().sort_index()
            all_vals = sorted(set(top_counts.index) | set(bot_counts.index))

            top_cts = [top_counts.get(v, 0) for v in all_vals]
            bot_cts = [bot_counts.get(v, 0) for v in all_vals]

            x = np.arange(len(all_vals))
            w = 0.35
            ax.bar(x - w/2, top_cts, w, label="Top-10 (best)", color="#43A047", alpha=0.8)
            ax.bar(x + w/2, bot_cts, w, label="Bottom-10 (worst)", color="#E53935", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in all_vals])
            ax.set_ylabel("Count in group")

        ax.set_xlabel(plabel)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("RF All Features: Parameter distribution in Top-10 vs Bottom-10 combos by wMAPE", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_top_vs_bottom_params.png")
    plt.close()

plot_rf_top_vs_bottom()
print("  [6/8] rf_top_vs_bottom_params.png")


# =========================================================================
# 7. min_samples_leaf x max_depth heatmap (marginalised over n_estimators)
#    one panel per sweep
# =========================================================================
def plot_leaf_vs_depth_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (title, df), _ in zip(axes, SWEEP_SETS, SWEEP_COLORS):
        rfr = get_rfr_full(df)
        rfr["depth_str"] = rfr["param_max_depth"].apply(depth_label)
        depth_order = ["5", "10", "20", "30", "None"]

        pivot = rfr.pivot_table(index="param_min_samples_leaf", columns="depth_str",
                                values="wmapE_val", aggfunc="min")
        pivot = pivot[[d for d in depth_order if d in pivot.columns]]
        pivot = pivot.sort_index()

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
        plt.colorbar(im, ax=ax, label="wMAPE", fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(pivot.columns)))
        col_labels = [c if c != "None" else "None\n(unlim.)" for c in pivot.columns]
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(v)) for v in pivot.index], fontsize=9)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("min_samples_leaf")
        ax.set_title(title)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    mid = (pivot.values[~np.isnan(pivot.values)].min() + pivot.values[~np.isnan(pivot.values)].max()) / 2
                    color = "white" if val > mid else "black"
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8, color=color)

    fig.suptitle("RF: min_samples_leaf vs max_depth heatmap (best wMAPE across n_estimators)", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_leaf_vs_depth_heatmap.png")
    plt.close()

plot_leaf_vs_depth_heatmap()
print("  [7/8] rf_leaf_vs_depth_heatmap.png")


# =========================================================================
# 8. Marginal effect summary: for each param, plot the average wMAPE
#    improvement when moving from worst to best value (across all sweeps)
# =========================================================================
def plot_marginal_sensitivity():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    param_configs = [
        ("param_n_estimators", "n_estimators", "RF"),
        ("param_max_depth", "max_depth", "RF"),
        ("param_min_samples_leaf", "min_samples_leaf", "RF"),
        ("param_n_neighbors", "n_neighbors", "KNN"),
        ("param_alpha", "alpha", "ElasticNet"),
        ("param_l1_ratio", "l1_ratio", "ElasticNet"),
    ]

    for ax, (param_col, param_name, model_type) in zip(axes.flat, param_configs):
        model_key = {"RF": "rfr", "KNN": "knn", "ElasticNet": "elasticnet"}[model_type]

        for (title, df), color in zip(SWEEP_SETS, SWEEP_COLORS):
            sub = df[(df["model"] == model_key) & (df["status"] == "ok") & (df["feature_set"] == "all")]
            if sub.empty or param_col not in sub.columns:
                continue
            sub = sub.dropna(subset=[param_col, "wmapE_val"])
            if sub.empty:
                continue

            if param_col == "param_max_depth":
                sub = sub.copy()
                sub["plot_val"] = sub[param_col].fillna(999)
                grouped_mean = sub.groupby("plot_val")["wmapE_val"].mean().sort_index()
                grouped_min = sub.groupby("plot_val")["wmapE_val"].min().sort_index()

                xlabels = [str(int(v)) if v != 999 else "None" for v in grouped_mean.index]
                x = range(len(xlabels))
                ax.plot(x, grouped_mean.values, marker="o", linewidth=2, color=color, alpha=0.7, label=f"{title} (mean)")
                ax.plot(x, grouped_min.values, marker="^", linewidth=1.5, linestyle="--", color=color, alpha=0.9, label=f"{title} (best)")
                ax.set_xticks(x)
                ax.set_xticklabels(xlabels, fontsize=9)
            else:
                grouped_mean = sub.groupby(param_col)["wmapE_val"].mean().sort_index()
                grouped_min = sub.groupby(param_col)["wmapE_val"].min().sort_index()
                ax.plot(grouped_mean.index, grouped_mean.values, marker="o", linewidth=2, color=color, alpha=0.7, label=f"{title} (mean)")
                ax.plot(grouped_min.index, grouped_min.values, marker="^", linewidth=1.5, linestyle="--", color=color, alpha=0.9, label=f"{title} (best)")

        ax.set_xlabel(param_name)
        ax.set_ylabel("Val wMAPE")
        ax.set_title(f"{model_type}: {param_name}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("Marginal Parameter Sensitivity: Mean & Best wMAPE per parameter value\n(holding other params free)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "param_marginal_sensitivity.png")
    plt.close()

plot_marginal_sensitivity()
print("  [8/8] param_marginal_sensitivity.png")


print(f"\nAll parameter plots saved to {PLOTS}/")
