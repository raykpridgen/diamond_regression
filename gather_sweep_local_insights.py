#!/usr/bin/env python3
"""
Gather insights from the most recent sweep_local_* run (or a given sweep directory).

Reads either sweep_results.csv (completed run) or _sweep_checkpoint.csv (partial run),
optionally parses sweep.log for run context, and writes:
  - sweep_local_insights.txt  : human-readable summary + comparison with dim/qual/all sweeps
  - plot_local/               : local-only and comparison visuals (wMAPE, R² vs prior sweeps)
  - report_generated.tex      : key numbers and table for report.tex

Same structure as generate_plots.py / generate_param_plots.py:
  BASE = Path(__file__).parent, find sweep dir, load CSV, analyze, write report + plots.

Usage:
  python gather_sweep_local_insights.py
  python gather_sweep_local_insights.py --sweep_dir sweep_local_20260309_184135
  python gather_sweep_local_insights.py --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SWEEP_BASE = BASE / "big_sweep"  # All sweeps (dim, qual, all_features, local) live here
PLOT_OUTDIR = BASE / "plot_local"
REPORT_GENERATED_TEX = BASE / "report_generated.tex"
SPLIT_LABELS = ("train", "val", "test")
METRIC_KEYS = ("r2", "rmse", "mae", "wmapE")
REFERENCE_SWEEP_PATTERNS = [
    ("Dimensions", "sweep_dimensions_*"),
    ("Qualities", "sweep_qualities_*"),
    ("All features", "sweep_all_features_*"),
]


def find_latest_sweep_local() -> Optional[Path]:
    """Return the most recent sweep_local_* directory under SWEEP_BASE (big_sweep), or None."""
    candidates = sorted(SWEEP_BASE.glob("sweep_local_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_reference_sweeps() -> List[Tuple[str, Path]]:
    """Return [(label, sweep_dir), ...] for Dimensions, Qualities, All features (latest of each)."""
    out: List[Tuple[str, Path]] = []
    for label, pattern in REFERENCE_SWEEP_PATTERNS:
        candidates = sorted(SWEEP_BASE.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for d in candidates:
            if (d / "sweep_results.csv").exists():
                out.append((label, d))
                break
    return out


def get_sweep_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """From a sweep results DataFrame, return best val wMAPE/R² overall and per model."""
    ok = df[df["status"].fillna("ok") == "ok"]
    if ok.empty:
        return {"best_wmape_val": np.nan, "best_r2_val": np.nan, "per_model": {}}
    per_model: Dict[str, Dict[str, float]] = {}
    for model_name, grp in ok.groupby("model"):
        per_model[model_name] = {
            "wmape_val": float(grp["wmapE_val"].min()),
            "r2_val": float(grp["r2_val"].max()),
        }
    return {
        "best_wmape_val": float(ok["wmapE_val"].min()),
        "best_r2_val": float(ok["r2_val"].max()),
        "per_model": per_model,
    }


def get_sweep_csv(sweep_dir: Path) -> Optional[Path]:
    """Return path to sweep_results.csv or _sweep_checkpoint.csv, preferring results."""
    for name in ("sweep_results.csv", "_sweep_checkpoint.csv"):
        p = sweep_dir / name
        if p.exists():
            return p
    return None


def load_sweep_data(csv_path: Path) -> pd.DataFrame:
    """Load and normalize sweep CSV (status fillna like generate_plots)."""
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df.copy()
        df["status"] = df["status"].fillna("ok")
    return df


def parse_best_params(row: pd.Series) -> Dict[str, Any]:
    """Extract param_* columns from a results row into a dict."""
    _INT_PARAMS = {"n_estimators", "max_depth", "min_samples_leaf", "n_neighbors"}
    params: Dict[str, Any] = {}
    for col in row.index:
        if not col.startswith("param_"):
            continue
        v = row[col]
        if pd.isna(v):
            continue
        key = col[len("param_"):]
        if v == "None":
            params[key] = None
        elif isinstance(v, str):
            try:
                params[key] = int(v) if "." not in v else float(v)
            except ValueError:
                params[key] = v
        elif key in _INT_PARAMS:
            params[key] = int(v)
        elif isinstance(v, (np.integer, np.floating)):
            params[key] = int(v) if isinstance(v, np.integer) else float(v)
        else:
            params[key] = v
    return params


def read_sweep_log(sweep_dir: Path) -> List[str]:
    """Read sweep.log lines if present (for run context)."""
    log_path = sweep_dir / "sweep.log"
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8").strip().splitlines()


def build_best_config(best_row: pd.Series) -> Dict[str, Any]:
    """Build a best_config-like dict from a single results row."""
    features = best_row.get("features")
    if isinstance(features, str):
        features = features.split("+")
    return {
        "model": best_row["model"],
        "params": parse_best_params(best_row),
        "features": list(features) if features is not None else [],
        "n_features": int(best_row.get("n_features", 0)),
        "feature_set": best_row.get("feature_set", ""),
        "metrics": {
            f"{mk}_{sl}": float(best_row.get(f"{mk}_{sl}", float("nan")))
            for sl in SPLIT_LABELS for mk in METRIC_KEYS
        },
    }


def fmt(val: Any) -> str:
    if isinstance(val, float) and not np.isnan(val):
        return f"{val:.4f}"
    return str(val)


def write_insights_report(
    sweep_dir: Path,
    csv_path: Path,
    results: pd.DataFrame,
    ok: pd.DataFrame,
    best_config: Dict[str, Any],
    bpm_df: pd.DataFrame,
    log_lines: List[str],
    is_partial: bool,
    out_path: Path,
    ref_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Write sweep_local_insights.txt with summary, best configs, and comparison with prior sweeps."""
    errored = results[results["status"] == "error"]
    skipped = results[results["status"] == "skipped"]

    lines = [
        "=" * 72,
        "  SWEEP LOCAL INSIGHTS  (adjusted run)",
        "=" * 72,
        "",
        f"Sweep dir    : {sweep_dir}",
        f"Data source  : {csv_path.name}" + (" (partial run)" if is_partial else " (full run)"),
        "",
    ]

    if ref_summaries:
        lines.extend([
            "-" * 72,
            "  COMPARISON WITH PRIOR SWEEPS  (best validation metric per run)",
            "-" * 72,
            "",
            "  Sweep            Best val wMAPE   Best val R²   (lower wMAPE / higher R² = better)",
            "  " + "-" * 60,
        ])
        local_wmape = best_config.get("metrics", {}).get("wmapE_val", np.nan)
        local_r2 = best_config.get("metrics", {}).get("r2_val", np.nan)
        for label, summary in ref_summaries.items():
            w = summary.get("best_wmape_val", np.nan)
            r = summary.get("best_r2_val", np.nan)
            lines.append(f"  {label:18s}   {fmt(w):>12}   {fmt(r):>10}")
        lines.append(f"  {'Local (this run)':18s}   {fmt(local_wmape):>12}   {fmt(local_r2):>10}")
        lines.append("")
        best_prior_wmape = min((s.get("best_wmape_val", np.nan) for s in ref_summaries.values() if not np.isnan(s.get("best_wmape_val"))), default=np.nan)
        if not np.isnan(best_prior_wmape) and not np.isnan(local_wmape):
            diff = local_wmape - best_prior_wmape
            lines.append(f"  vs best prior sweep: val wMAPE {diff:+.4f}  " + ("(local better)" if diff < 0 else "(prior better)"))
        lines.append("")

    if log_lines:
        lines.append("-" * 72)
        lines.append("  Run context (from sweep.log)")
        lines.append("-" * 72)
        for ln in log_lines[:12]:  # first few lines
            lines.append(f"  {ln}")
        lines.append("")

    lines += [
        f"Total combos: {len(results):,}",
        f"  Successful : {len(ok):,}",
        f"  Errors     : {len(errored):,}",
        f"  Skipped    : {len(skipped):,}",
        f"Total time  : {results['elapsed_s'].sum():.1f}s",
        "",
        "-" * 72,
        "  BEST OVERALL SO FAR  (by validation wMAPE)",
        "-" * 72,
        f"  Model       : {best_config['model']}",
    ]
    for pk, pv in best_config.get("params", {}).items():
        lines.append(f"  {pk:15s}: {pv}")
    lines.append(
        f"  Features ({best_config.get('n_features', 0):>2d})  : "
        f"{', '.join(best_config.get('features', []))}"
    )
    lines.append("")
    lines.append("                  R²         RMSE        MAE        wMAPE")
    m = best_config.get("metrics", {})
    for sl in SPLIT_LABELS:
        lines.append(
            f"  {sl:7s}     {fmt(m.get(f'r2_{sl}')):>8}    "
            f"{fmt(m.get(f'rmse_{sl}')):>8}    "
            f"{fmt(m.get(f'mae_{sl}')):>8}    "
            f"{fmt(m.get(f'wmapE_{sl}')):>8}"
        )

    lines.extend(["", "-" * 72, "  BEST PER MODEL TYPE", "-" * 72])
    for _, brow in bpm_df.iterrows():
        lines.append(
            f"  {str(brow['model']):12s}  "
            f"R²(val)={fmt(brow.get('r2_val')):>8}  "
            f"wMAPE(val)={fmt(brow.get('wmapE_val')):>8}  "
            f"feature_set={brow.get('feature_set', '')}"
        )

    # Top 10 by val wMAPE
    lines.extend(["", "-" * 72, "  TOP 10 CONFIGS BY VALIDATION wMAPE", "-" * 72])
    top10 = ok.nsmallest(10, "wmapE_val")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        lines.append(
            f"  #{i:2d}  {row['model']:10s}  wMAPE(val)={fmt(row.get('wmapE_val')):>8}  "
            f"R²(val)={fmt(row.get('r2_val')):>8}  "
            f"features={row.get('features', '')[:50]}"
        )

    # Analysis
    if ref_summaries:
        lines.extend([
            "",
            "-" * 72,
            "  ANALYSIS",
            "-" * 72,
            "",
        ])
        best_prior_wmape = min(
            (s.get("best_wmape_val") for s in ref_summaries.values() if not np.isnan(s.get("best_wmape_val", np.nan))),
            default=np.nan,
        )
        best_prior_r2 = max(
            (s.get("best_r2_val") for s in ref_summaries.values() if not np.isnan(s.get("best_r2_val", np.nan))),
            default=np.nan,
        )
        local_w = best_config.get("metrics", {}).get("wmapE_val", np.nan)
        local_r = best_config.get("metrics", {}).get("r2_val", np.nan)
        if not np.isnan(local_w) and not np.isnan(best_prior_wmape):
            lines.append(f"  Best prior sweep val wMAPE: {fmt(best_prior_wmape)}. Local best: {fmt(local_w)}.")
            if local_w < best_prior_wmape:
                lines.append("  -> Local run improves on best prior wMAPE (adjusted grid/features are competitive).")
            else:
                lines.append("  -> Best prior sweep has lower val wMAPE; local run is close but not better so far.")
        if not np.isnan(local_r) and not np.isnan(best_prior_r2):
            lines.append(f"  Best prior sweep val R²: {fmt(best_prior_r2)}. Local best: {fmt(local_r)}.")
        if is_partial:
            lines.append("  Run is partial; completing the sweep may yield a better or worse optimum.")

    lines.extend(["", "=" * 72])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_plots(ok: pd.DataFrame, bpm_df: pd.DataFrame, outdir: Path) -> None:
    """Generate a few insight plots (same style as generate_plots)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
    })
    outdir.mkdir(parents=True, exist_ok=True)

    models = ok["model"].unique().tolist()
    if not models:
        return

    colors = {"rfr": "#2196F3", "knn": "#4CAF50", "elasticnet": "#FF9800"}
    model_colors = [colors.get(m, "#9E9E9E") for m in models]

    # 1. Best wMAPE per model (bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    best_wmape = [bpm_df[bpm_df["model"] == m]["wmapE_val"].min() for m in models]
    bars = ax.bar(range(len(models)), best_wmape, color=model_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Best Validation wMAPE")
    ax.set_title("Sweep local: Best wMAPE per Model")
    for bar, v in zip(bars, best_wmape):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "local_best_wmape_per_model.png")
    plt.close()
    print(f"  [1/2] {outdir.name}/local_best_wmape_per_model.png")

    # 2. wMAPE distribution by model (box)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [ok[ok["model"] == m]["wmapE_val"].dropna().values for m in models]
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], model_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Validation wMAPE")
    ax.set_title("Sweep local: wMAPE Distribution by Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "local_wmape_distribution.png")
    plt.close()
    print(f"  [2/2] {outdir.name}/local_wmape_distribution.png")


def generate_comparison_plots(
    local_bpm_df: pd.DataFrame,
    local_best_wmape: float,
    local_best_r2: float,
    ref_summaries: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    """Generate comparison visuals: local vs Dimensions, Qualities, All features."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
    })
    outdir.mkdir(parents=True, exist_ok=True)

    sweep_labels = list(ref_summaries.keys()) + ["Local\n(partial)"]
    # Overall best: one bar per sweep (wMAPE and R²)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    wmapes = [ref_summaries[l]["best_wmape_val"] for l in ref_summaries.keys()] + [local_best_wmape]
    r2s = [ref_summaries[l]["best_r2_val"] for l in ref_summaries.keys()] + [local_best_r2]
    x = np.arange(len(sweep_labels))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars1 = ax1.bar(x, wmapes, color=colors[: len(sweep_labels)], edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Best validation wMAPE (lower is better)")
    ax1.set_title("Best wMAPE across sweeps")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sweep_labels)
    for bar, v in zip(bars1, wmapes):
        if not np.isnan(v):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, r2s, color=colors[: len(sweep_labels)], edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Best validation R² (higher is better)")
    ax2.set_title("Best R² across sweeps")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sweep_labels)
    for bar, v in zip(bars2, r2s):
        if not np.isnan(v):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0.6, 1.02)
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Sweep comparison: Local (adjusted) run vs Dimensions, Qualities, All features", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "comparison_overall_wmape_r2.png")
    plt.close()
    print(f"  [3/4] {outdir.name}/comparison_overall_wmape_r2.png")

    # Per-model comparison where we have overlapping models
    all_models = set()
    for s in ref_summaries.values():
        all_models.update(s.get("per_model", {}).keys())
    for _, row in local_bpm_df.iterrows():
        all_models.add(row["model"])
    all_models = sorted(all_models)
    if not all_models:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.2
    x = np.arange(len(sweep_labels))
    model_colors = {"rfr": "#2196F3", "knn": "#4CAF50", "elasticnet": "#FF9800"}
    for i, model in enumerate(all_models):
        wmapes_m = []
        for label in ref_summaries.keys():
            pm = ref_summaries[label].get("per_model", {}).get(model, {})
            wmapes_m.append(pm.get("wmape_val", np.nan))
        local_row = local_bpm_df[local_bpm_df["model"] == model]
        local_w = float(local_row["wmapE_val"].min()) if not local_row.empty else np.nan
        wmapes_m.append(local_w)
        offset = (i - len(all_models) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, wmapes_m, width, label=model, color=model_colors.get(model, "#9E9E9E"))
    ax1.set_ylabel("Best validation wMAPE")
    ax1.set_title("Best wMAPE by model across sweeps")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sweep_labels)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for i, model in enumerate(all_models):
        r2s_m = []
        for label in ref_summaries.keys():
            pm = ref_summaries[label].get("per_model", {}).get(model, {})
            r2s_m.append(pm.get("r2_val", np.nan))
        local_row = local_bpm_df[local_bpm_df["model"] == model]
        local_r = float(local_row["r2_val"].max()) if not local_row.empty else np.nan
        r2s_m.append(local_r)
        offset = (i - len(all_models) / 2 + 0.5) * width
        ax2.bar(x + offset, r2s_m, width, label=model, color=model_colors.get(model, "#9E9E9E"))
    ax2.set_ylabel("Best validation R²")
    ax2.set_title("Best R² by model across sweeps")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sweep_labels)
    ax2.legend()
    ax2.set_ylim(0.6, 1.02)
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-model comparison across sweeps", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "comparison_by_model.png")
    plt.close()
    print(f"  [4/4] {outdir.name}/comparison_by_model.png")


def write_report_generated_tex(
    best_config: Dict[str, Any],
    ref_summaries: Dict[str, Dict[str, Any]],
    is_partial: bool,
    out_path: Path,
) -> None:
    """Write report_generated.tex with key numbers and comparison table for report.tex."""
    local_wmape = best_config.get("metrics", {}).get("wmapE_val", np.nan)
    local_r2 = best_config.get("metrics", {}).get("r2_val", np.nan)
    local_test_wmape = best_config.get("metrics", {}).get("wmapE_test", np.nan)
    local_test_r2 = best_config.get("metrics", {}).get("r2_test", np.nan)

    lines = [
        "% Auto-generated by gather_sweep_local_insights.py -- do not edit by hand",
        "",
        "\\newcommand{\\LocalBestWmapeVal}{" + (f"{local_wmape:.4f}" if not np.isnan(local_wmape) else "---") + "}",
        "\\newcommand{\\LocalBestRtwoVal}{" + (f"{local_r2:.4f}" if not np.isnan(local_r2) else "---") + "}",
        "\\newcommand{\\LocalBestWmapeTest}{" + (f"{local_test_wmape:.4f}" if not np.isnan(local_test_wmape) else "---") + "}",
        "\\newcommand{\\LocalBestRtwoTest}{" + (f"{local_test_r2:.4f}" if not np.isnan(local_test_r2) else "---") + "}",
        "\\newcommand{\\LocalRunPartial}{" + ("partial (incomplete)" if is_partial else "full") + "}",
        "\\newcommand{\\LocalBestModel}{" + str(best_config.get("model", "---")) + "}",
        "",
        "% Comparison table rows (prior sweeps)",
    ]
    safe_suffix = {"Dimensions": "Dim", "Qualities": "Qual", "All features": "All"}
    for label, summary in ref_summaries.items():
        w = summary.get("best_wmape_val", np.nan)
        r = summary.get("best_r2_val", np.nan)
        wstr = f"{w:.4f}" if not np.isnan(w) else "---"
        rstr = f"{r:.4f}" if not np.isnan(r) else "---"
        suf = safe_suffix.get(label, label.replace(" ", ""))
        lines.append(f"\\newcommand{{\\PriorWmape{suf}}}{{{wstr}}}")
        lines.append(f"\\newcommand{{\\PriorRtwo{suf}}}{{{rstr}}}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather insights from sweep_local run")
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=None,
        help="Sweep directory (default: latest sweep_local_*)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for insights .txt (default: <sweep_dir>/sweep_local_insights.txt)",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir or find_latest_sweep_local()
    if sweep_dir is None:
        print("No sweep_local_* directory found in big_sweep/. Run with --sweep_dir <path>.")
        return
    if not sweep_dir.is_absolute():
        # Try big_sweep/ first, then BASE
        cand = SWEEP_BASE / sweep_dir
        sweep_dir = cand if cand.exists() else BASE / sweep_dir
    if not sweep_dir.exists():
        print(f"Sweep dir not found: {sweep_dir}")
        return

    csv_path = get_sweep_csv(sweep_dir)
    if csv_path is None:
        print(f"No sweep_results.csv or _sweep_checkpoint.csv in {sweep_dir}")
        return

    is_partial = csv_path.name == "_sweep_checkpoint.csv"
    print(f"Using {csv_path} ({'partial' if is_partial else 'full'} run)")

    results = load_sweep_data(csv_path)
    ok = results[results["status"] == "ok"]
    if ok.empty:
        print("No successful combinations to report.")
        return

    best_idx = ok["wmapE_val"].idxmin()
    best_row = ok.loc[best_idx]
    best_config = build_best_config(best_row)

    param_cols = [c for c in ok.columns if c.startswith("param_")]
    bpm_rows = []
    for model_name, grp in ok.groupby("model"):
        bidx = grp["wmapE_val"].idxmin()
        b = grp.loc[bidx]
        row_dict = {"model": model_name, "feature_set": b.get("feature_set"), "n_features": b.get("n_features")}
        for sl in SPLIT_LABELS:
            for mk in METRIC_KEYS:
                row_dict[f"{mk}_{sl}"] = b.get(f"{mk}_{sl}", float("nan"))
        for pc in param_cols:
            if pd.notna(b.get(pc)):
                row_dict[pc] = b[pc]
        bpm_rows.append(row_dict)
    bpm_df = pd.DataFrame(bpm_rows).sort_values("wmapE_val")

    ref_summaries: Dict[str, Dict[str, Any]] = {}
    for label, ref_dir in find_reference_sweeps():
        ref_csv = ref_dir / "sweep_results.csv"
        ref_df = load_sweep_data(ref_csv)
        ref_summaries[label] = get_sweep_summary(ref_df)
    if ref_summaries:
        print(f"Loaded {len(ref_summaries)} prior sweeps for comparison")

    log_lines = read_sweep_log(sweep_dir)
    out_txt = args.output or sweep_dir / "sweep_local_insights.txt"
    write_insights_report(
        sweep_dir, csv_path, results, ok,
        best_config, bpm_df, log_lines, is_partial, out_txt,
        ref_summaries=ref_summaries or None,
    )
    print(f"Report written to {out_txt}")

    if not args.no_plots:
        generate_plots(ok, bpm_df, PLOT_OUTDIR)
        if ref_summaries:
            local_best_wmape = float(ok["wmapE_val"].min())
            local_best_r2 = float(ok["r2_val"].max())
            generate_comparison_plots(bpm_df, local_best_wmape, local_best_r2, ref_summaries, PLOT_OUTDIR)
        print(f"Plots saved to {PLOT_OUTDIR}/")

    write_report_generated_tex(best_config, ref_summaries, is_partial, REPORT_GENERATED_TEX)
    print(f"Generated {REPORT_GENERATED_TEX.name} for report.tex")


if __name__ == "__main__":
    main()
