#!/usr/bin/env python3
"""
Multi-model hyperparameter & feature-subset sweep.

Evaluates combinatorial grids across multiple regression model types
(linear, ridge, lasso, elasticnet, rfr, knn) with optional feature-subset
sweeps.  Designed for unattended background runs with robust error handling,
incremental checkpointing, and detailed post-sweep reporting.

Wraps regress_train.train_and_evaluate() so all preprocessing, metrics, and
model construction are shared with the single-model training path.

Outputs (in <results_dir>/sweep_<timestamp>/):
  sweep_results.csv         — every combination with full train/val/test metrics
  best_config.json          — single best configuration (by val wMAPE)
  best_per_model.csv        — best hyperparams per model type
  feature_importance.csv    — importance/coefficients from the best model
  ablation_importance.csv   — feature importance via leave-one-out (--sweep_features)
  sweep_summary.txt         — human-readable report
  sweep.log                 — timestamped log for monitoring background runs
  errors.log                — tracebacks for failed combinations (if any)
  models/best/              — saved pipeline + plots for the overall best model
  plots/                    — model comparison and feature importance charts

Dependencies:
  pip install pandas numpy matplotlib scikit-learn rich joblib

Examples:
  # RF-only sweep (backward compatible)
  python rf_param_train.py --csv diamonds.csv --models rfr

  # Full multi-model sweep on diamond prices
  python rf_param_train.py --csv diamonds.csv --target price \
      --models linear ridge lasso elasticnet rfr knn \
      --sweep_features

  # Custom hyperparameter grids
  python rf_param_train.py --csv diamonds.csv --target price \
      --models rfr knn elasticnet \
      --n_estimators 200 400 800 \
      --max_depth 10 20 0 \
      --n_neighbors 3 5 10 20 \
      --alpha 0.01 0.1 1.0 \
      --l1_ratio 0.2 0.5 0.8
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import textwrap
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

from regress_train import (
    MODEL_CHOICES as RT_MODEL_CHOICES,
    build_preprocessor,
    evaluate_regression,
    get_feature_names,
    load_and_prepare,
    make_model,
    now_stamp,
    safe_mkdir,
    save_fig,
    split_data,
    train_and_evaluate,
    wmape,
    write_text,
)

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------
THEME = Theme(
    {
        "title": "bold cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "dim": "dim",
        "key": "bold magenta",
        "path": "cyan",
    }
)
console = Console(theme=THEME)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_ESTIMATORS = [100, 200, 400]
DEFAULT_MAX_DEPTH: List[Optional[int]] = [10, 20, None]
DEFAULT_MIN_SAMPLES_LEAF = [1, 2, 5]
DEFAULT_ALPHA = [0.001, 0.01, 0.1, 1.0, 10.0]
DEFAULT_L1_RATIO = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_N_NEIGHBORS = [3, 5, 10, 15, 20]
DEFAULT_KNN_WEIGHTS = ["uniform", "distance"]

METRIC_KEYS = ("r2", "rmse", "mae", "wmapE")
SPLIT_LABELS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(outdir: Path) -> logging.Logger:
    log = logging.getLogger("sweep")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(outdir / "sweep.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Parameter grid construction
# ---------------------------------------------------------------------------

def build_model_grids(
    models: List[str],
    n_estimators: List[int],
    max_depth: List[Optional[int]],
    min_samples_leaf: List[int],
    alpha: List[float],
    l1_ratio: List[float],
    n_neighbors: List[int],
    knn_weights: List[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return a flat list of (model_name, hyperparams_dict) for every combo."""
    combos: List[Tuple[str, Dict[str, Any]]] = []

    for model in models:
        if model == "linear":
            combos.append(("linear", {}))

        elif model == "ridge":
            for a in alpha:
                combos.append(("ridge", {"alpha": a}))

        elif model == "lasso":
            for a in alpha:
                combos.append(("lasso", {"alpha": a}))

        elif model == "elasticnet":
            for a, lr in itertools.product(alpha, l1_ratio):
                combos.append(("elasticnet", {"alpha": a, "l1_ratio": lr}))

        elif model == "rfr":
            for n, d, lf in itertools.product(n_estimators, max_depth, min_samples_leaf):
                combos.append(("rfr", {
                    "n_estimators": n, "max_depth": d, "min_samples_leaf": lf,
                }))

        elif model == "knn":
            for k, w in itertools.product(n_neighbors, knn_weights):
                combos.append(("knn", {"n_neighbors": k, "weights": w}))

    return combos


# ---------------------------------------------------------------------------
# Feature-subset generation
# ---------------------------------------------------------------------------

def generate_feature_subsets(
    features: List[str],
    min_features: int = 2,
) -> List[Tuple[str, List[str]]]:
    """
    Generate labeled feature subsets:
      - "all"              : the full feature set
      - "drop_<col>"       : leave-one-out variants
      - "keep_<k>of<n>"    : choose-k combos (only when <= 6 features)
    """
    subsets: List[Tuple[str, List[str]]] = [("all", list(features))]

    if len(features) <= 6:
        for k in range(min_features, len(features)):
            for combo in itertools.combinations(features, k):
                subset = list(combo)
                if subset != features:
                    subsets.append((f"keep_{len(subset)}of{len(features)}", subset))
    else:
        for drop in features:
            subset = [f for f in features if f != drop]
            if len(subset) >= min_features:
                subsets.append((f"drop_{drop}", subset))

    return subsets


# ---------------------------------------------------------------------------
# Sweep core
# ---------------------------------------------------------------------------

def run_sweep(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_grids: List[Tuple[str, Dict[str, Any]]],
    feature_subsets: List[Tuple[str, List[str]]],
    seed: int,
    outdir: Path,
    log: logging.Logger,
    checkpoint_every: int = 50,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the full grid sweep over pre-split data.

    Returns (results_df, errors_list).  Each failed combination is caught and
    logged — the sweep continues to completion regardless of individual errors.
    Partial results are checkpointed to disk every *checkpoint_every* combos.
    """
    total = len(model_grids) * len(feature_subsets)
    msg = (
        f"Sweeping {total:,} combinations  "
        f"({len(model_grids)} model configs x {len(feature_subsets)} feature subsets)"
    )
    console.print(f"[key]{msg}[/key]")
    log.info(msg)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    checkpoint_path = outdir / "_sweep_checkpoint.csv"

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("({task.percentage:>5.1f}%)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        ptask = progress.add_task("Sweep", total=total)

        for idx, ((model_name, params), (feat_label, feat_list)) in enumerate(
            itertools.product(model_grids, feature_subsets)
        ):
            combo_id = idx + 1
            t0 = time.perf_counter()

            row: Dict[str, Any] = {
                "combo_id": combo_id,
                "model": model_name,
                "feature_set": feat_label,
                "n_features": len(feat_list),
                "features": "+".join(feat_list),
            }
            for pk, pv in params.items():
                row[f"param_{pk}"] = pv if pv is not None else "None"

            try:
                Xt = X_train[feat_list]
                Xv = X_val[feat_list]
                Xte = X_test[feat_list]

                result = train_and_evaluate(
                    model_name=model_name,
                    X_train=Xt, y_train=y_train,
                    X_val=Xv, y_val=y_val,
                    X_test=Xte, y_test=y_test,
                    outdir=None,
                    random_state=seed,
                    save_model=False,
                    plots=False,
                    quiet=True,
                    **params,
                )

                if result.get("skipped"):
                    row["status"] = "skipped"
                    row["skip_reason"] = result.get("reason", "")
                    log.warning(
                        f"#{combo_id} {model_name}/{feat_label}: "
                        f"skipped — {row['skip_reason']}"
                    )
                else:
                    row["status"] = "ok"
                    for sl in SPLIT_LABELS:
                        for mk in METRIC_KEYS:
                            row[f"{mk}_{sl}"] = result.get(f"{mk}_{sl}", float("nan"))

            except Exception as exc:
                row["status"] = "error"
                row["error_msg"] = str(exc)[:200]
                tb = traceback.format_exc()
                errors.append({
                    "combo_id": combo_id,
                    "model": model_name,
                    "params": {k: str(v) for k, v in params.items()},
                    "feature_set": feat_label,
                    "error": str(exc),
                    "traceback": tb,
                })
                log.error(f"#{combo_id} {model_name}/{feat_label}: {exc}")

            row["elapsed_s"] = round(time.perf_counter() - t0, 3)
            rows.append(row)

            if combo_id % checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                log.info(f"Checkpoint at {combo_id}/{total}")

            progress.advance(ptask)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return pd.DataFrame(rows), errors


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------

def extract_model_feature_importance(
    model_name: str,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    outdir: Path,
    target_name: str,
) -> Optional[pd.DataFrame]:
    """Retrain the best model, save artifacts, and return feature importance."""
    result = train_and_evaluate(
        model_name=model_name,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        outdir=outdir,
        random_state=seed,
        save_model=True,
        plots=True,
        quiet=False,
        target_name=target_name,
        **params,
    )
    if result.get("skipped"):
        return None

    pipe = result["pipeline"]
    model_obj = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    feat_names = get_feature_names(prep) or [f"f_{i}" for i in range(X_train.shape[1])]

    coefs = None
    label = "importance"
    if hasattr(model_obj, "coef_"):
        coefs = np.asarray(model_obj.coef_).ravel()
        label = "coefficient"
    elif hasattr(model_obj, "feature_importances_"):
        coefs = np.asarray(model_obj.feature_importances_).ravel()

    if coefs is None:
        return None

    if len(feat_names) != len(coefs):
        feat_names = [f"f_{i}" for i in range(len(coefs))]

    imp = pd.DataFrame({
        "feature": feat_names,
        label: coefs,
        f"abs_{label}": np.abs(coefs),
    })
    imp = imp.sort_values(f"abs_{label}", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", range(1, len(imp) + 1))
    return imp


def compute_ablation_importance(
    results: pd.DataFrame,
    all_features: List[str],
) -> Optional[pd.DataFrame]:
    """
    Feature importance via leave-one-out ablation: compare full-feature
    performance to the best same-model performance when each feature is dropped.
    A large positive wMAPE delta means the feature is important.
    """
    ok = results[results["status"] == "ok"].copy()
    if ok.empty or "wmapE_val" not in ok.columns:
        return None

    full = ok[ok["feature_set"] == "all"]
    if full.empty:
        return None

    baseline_wmape = full["wmapE_val"].min()
    baseline_r2 = float(full.loc[full["wmapE_val"].idxmin(), "r2_val"])

    rows = []
    for feat in all_features:
        drop_label = f"drop_{feat}"
        dropped = ok[ok["feature_set"] == drop_label]
        if dropped.empty:
            continue

        best_dropped = dropped.loc[dropped["wmapE_val"].idxmin()]
        wmape_without = float(best_dropped["wmapE_val"])
        r2_without = float(best_dropped["r2_val"])

        rows.append({
            "feature": feat,
            "wmapE_val_without": wmape_without,
            "wmapE_val_baseline": baseline_wmape,
            "wmapE_delta": wmape_without - baseline_wmape,
            "r2_val_without": r2_without,
            "r2_val_baseline": baseline_r2,
            "r2_delta": baseline_r2 - r2_without,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("wmapE_delta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _parse_best_params(best_row: pd.Series) -> Dict[str, Any]:
    """Extract hyperparams from a results row's param_* columns."""
    _INT_PARAMS = {"n_estimators", "max_depth", "min_samples_leaf", "n_neighbors"}
    params: Dict[str, Any] = {}
    for col in best_row.index:
        if not col.startswith("param_"):
            continue
        v = best_row[col]
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
        elif isinstance(v, np.integer):
            params[key] = int(v)
        elif isinstance(v, np.floating):
            params[key] = float(v)
        else:
            params[key] = v
    return params


def generate_reports(
    results: pd.DataFrame,
    errors: List[Dict[str, Any]],
    target_col: str,
    all_features: List[str],
    feature_imp: Optional[pd.DataFrame],
    ablation_imp: Optional[pd.DataFrame],
    outdir: Path,
    csv_path: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """Write all report files and return the best config dict."""
    results.to_csv(outdir / "sweep_results.csv", index=False)

    ok = results[results["status"] == "ok"]
    errored = results[results["status"] == "error"]
    skipped = results[results["status"] == "skipped"]

    if ok.empty:
        console.print("[warn]No successful combinations — cannot generate reports.[/warn]")
        log.warning("No successful combinations to report")
        return {}

    # --- Best overall ---
    best_idx = ok["wmapE_val"].idxmin()
    best = ok.loc[best_idx]
    best_params = _parse_best_params(best)

    best_config: Dict[str, Any] = {
        "model": best["model"],
        "params": best_params,
        "features": best["features"].split("+"),
        "n_features": int(best["n_features"]),
        "metrics": {
            f"{mk}_{sl}": float(best.get(f"{mk}_{sl}", float("nan")))
            for sl in SPLIT_LABELS for mk in METRIC_KEYS
        },
    }
    (outdir / "best_config.json").write_text(
        json.dumps(best_config, indent=2), encoding="utf-8",
    )

    # --- Best per model ---
    best_per_model_rows = []
    param_cols = [c for c in ok.columns if c.startswith("param_")]
    for model_name, grp in ok.groupby("model"):
        bidx = grp["wmapE_val"].idxmin()
        b = grp.loc[bidx]
        row_dict: Dict[str, Any] = {"model": model_name}
        for sl in SPLIT_LABELS:
            for mk in METRIC_KEYS:
                col = f"{mk}_{sl}"
                row_dict[col] = b.get(col, float("nan"))
        for pc in param_cols:
            if pd.notna(b.get(pc)):
                row_dict[pc] = b[pc]
        row_dict["feature_set"] = b["feature_set"]
        row_dict["n_features"] = b["n_features"]
        best_per_model_rows.append(row_dict)

    bpm_df = pd.DataFrame(best_per_model_rows).sort_values("wmapE_val")
    bpm_df.to_csv(outdir / "best_per_model.csv", index=False)

    # --- Feature importance CSVs ---
    if feature_imp is not None:
        feature_imp.to_csv(outdir / "feature_importance.csv", index=False)
    if ablation_imp is not None:
        ablation_imp.to_csv(outdir / "ablation_importance.csv", index=False)

    # --- Errors log ---
    if errors:
        err_lines = []
        for e in errors:
            err_lines.append(
                f"--- Combo #{e['combo_id']} ({e['model']}, {e['feature_set']}) ---"
            )
            err_lines.append(f"Params: {e['params']}")
            err_lines.append(e["traceback"])
            err_lines.append("")
        write_text(outdir / "errors.log", "\n".join(err_lines))

    # --- Human-readable summary ---
    _write_sweep_summary(
        outdir, csv_path, target_col, all_features,
        results, ok, errored, skipped,
        best_config, bpm_df, feature_imp, ablation_imp,
    )

    log.info(f"Reports written to {outdir}")
    return best_config


def _write_sweep_summary(
    outdir: Path,
    csv_path: Path,
    target_col: str,
    all_features: List[str],
    results: pd.DataFrame,
    ok: pd.DataFrame,
    errored: pd.DataFrame,
    skipped: pd.DataFrame,
    best_config: Dict[str, Any],
    bpm_df: pd.DataFrame,
    feature_imp: Optional[pd.DataFrame],
    ablation_imp: Optional[pd.DataFrame],
) -> None:
    def fmt(val: Any) -> str:
        if isinstance(val, float) and not np.isnan(val):
            return f"{val:.4f}"
        return str(val)

    lines = [
        "=" * 72,
        "  SWEEP RESULTS SUMMARY",
        "=" * 72,
        "",
        f"CSV            : {csv_path}",
        f"Target         : {target_col}",
        f"Features ({len(all_features):>2d})  : {', '.join(all_features)}",
        "",
        f"Total combos   : {len(results):,}",
        f"  Successful   : {len(ok):,}",
        f"  Errors       : {len(errored):,}",
        f"  Skipped      : {len(skipped):,}",
        f"Total time     : {results['elapsed_s'].sum():.1f}s",
        "",
        "-" * 72,
        "  BEST OVERALL CONFIGURATION  (by validation wMAPE)",
        "-" * 72,
        f"  Model          : {best_config['model']}",
    ]
    for pk, pv in best_config["params"].items():
        lines.append(f"  {pk:15s}: {pv}")
    lines.append(
        f"  Features ({best_config['n_features']:>2d})  : "
        f"{', '.join(best_config['features'])}"
    )
    lines.append("")
    lines.append("                  R²         RMSE        MAE        wMAPE")
    for sl in SPLIT_LABELS:
        m = best_config["metrics"]
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
            f"RMSE(val)={fmt(brow.get('rmse_val')):>8}  "
            f"MAE(val)={fmt(brow.get('mae_val')):>8}"
        )

    if ablation_imp is not None and not ablation_imp.empty:
        lines.extend([
            "", "-" * 72,
            "  FEATURE IMPORTANCE  (ablation / leave-one-out analysis)",
            "-" * 72,
            "  Higher wMAPE delta = removing that feature hurts more = MORE important",
            "",
        ])
        for _, r in ablation_imp.iterrows():
            tag = "IMPORTANT" if r["wmapE_delta"] > 0 else "dispensable"
            lines.append(
                f"  #{int(r['rank']):2d}  {r['feature']:25s}  "
                f"wMAPE delta={r['wmapE_delta']:+.6f}  "
                f"R² delta={r['r2_delta']:+.6f}  ({tag})"
            )

    if feature_imp is not None and not feature_imp.empty:
        label_col = "coefficient" if "coefficient" in feature_imp.columns else "importance"
        lines.extend([
            "", "-" * 72,
            f"  FEATURE {label_col.upper()}S  (from best model)",
            "-" * 72, "",
        ])
        for _, r in feature_imp.head(30).iterrows():
            lines.append(
                f"  #{int(r['rank']):2d}  {r['feature']:35s}  "
                f"{label_col}={r[label_col]:+.6f}"
            )

    wmape_vals = ok["wmapE_val"].dropna()
    if not wmape_vals.empty:
        lines.extend([
            "", "-" * 72,
            "  VALIDATION wMAPE DISTRIBUTION  (all successful combos)",
            "-" * 72,
            f"  min    : {wmape_vals.min():.6f}",
            f"  p10    : {wmape_vals.quantile(0.10):.6f}",
            f"  p25    : {wmape_vals.quantile(0.25):.6f}",
            f"  median : {wmape_vals.median():.6f}",
            f"  p75    : {wmape_vals.quantile(0.75):.6f}",
            f"  p90    : {wmape_vals.quantile(0.90):.6f}",
            f"  max    : {wmape_vals.max():.6f}",
            f"  mean   : {wmape_vals.mean():.6f}",
            f"  std    : {wmape_vals.std():.6f}",
        ])

    r2_vals = ok["r2_val"].dropna()
    if not r2_vals.empty:
        lines.extend([
            "", "-" * 72,
            "  VALIDATION R² DISTRIBUTION  (all successful combos)",
            "-" * 72,
            f"  min    : {r2_vals.min():.6f}",
            f"  p25    : {r2_vals.quantile(0.25):.6f}",
            f"  median : {r2_vals.median():.6f}",
            f"  p75    : {r2_vals.quantile(0.75):.6f}",
            f"  max    : {r2_vals.max():.6f}",
            f"  mean   : {r2_vals.mean():.6f}",
            f"  std    : {r2_vals.std():.6f}",
        ])

    lines.extend([
        "", "-" * 72,
        "  HOW TO REPRODUCE THE BEST MODEL",
        "-" * 72,
        f"  python regress_train.py --csv {csv_path.name} "
        f"--models {best_config['model']} \\",
        f"      --features {' '.join(best_config['features'])} "
        f"--target {target_col}",
    ])
    if best_config["params"]:
        lines.append(f"      # Hyperparams (set in code or extend CLI): {best_config['params']}")
    lines.extend(["", "=" * 72])

    write_text(outdir / "sweep_summary.txt", "\n".join(lines))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_model_comparison(results: pd.DataFrame, outdir: Path) -> None:
    plots_dir = outdir / "plots"
    safe_mkdir(plots_dir)

    ok = results[results["status"] == "ok"]
    if ok.empty:
        return

    best_wmape = ok.groupby("model")["wmapE_val"].min().sort_values()
    plt.figure(figsize=(10, max(4, len(best_wmape) * 0.8)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(best_wmape)))
    bars = plt.barh(best_wmape.index, best_wmape.values, color=colors)
    for bar, val in zip(bars, best_wmape.values):
        plt.text(
            val + best_wmape.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )
    plt.xlabel("Best Validation wMAPE (lower is better)")
    plt.title("Model Comparison — Best wMAPE per Model")
    save_fig(plots_dir / "model_comparison_wmape.png")

    best_r2 = ok.groupby("model")["r2_val"].max().sort_values()
    plt.figure(figsize=(10, max(4, len(best_r2) * 0.8)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(best_r2)))
    bars = plt.barh(best_r2.index, best_r2.values, color=colors)
    for bar, val in zip(bars, best_r2.values):
        plt.text(
            max(0, val - best_r2.max() * 0.08),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
            color="white", fontweight="bold",
        )
    plt.xlabel("Best Validation R² (higher is better)")
    plt.title("Model Comparison — Best R² per Model")
    save_fig(plots_dir / "model_comparison_r2.png")


def plot_ablation_importance(
    ablation_imp: Optional[pd.DataFrame], outdir: Path,
) -> None:
    if ablation_imp is None or ablation_imp.empty:
        return
    plots_dir = outdir / "plots"
    safe_mkdir(plots_dir)

    top = ablation_imp.sort_values("wmapE_delta").tail(25)
    plt.figure(figsize=(12, max(4, len(top) * 0.5)))
    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in top["wmapE_delta"]]
    plt.barh(top["feature"], top["wmapE_delta"], color=colors)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("wMAPE increase when feature removed (higher = more important)")
    plt.title("Feature Importance via Ablation (Leave-One-Out)")
    save_fig(plots_dir / "ablation_feature_importance.png")


def plot_rfr_heatmap(results: pd.DataFrame, outdir: Path) -> None:
    """Plot n_estimators vs max_depth heatmap for RF combos (full feature set)."""
    plots_dir = outdir / "plots"
    safe_mkdir(plots_dir)

    rfr = results[
        (results["model"] == "rfr")
        & (results["status"] == "ok")
        & (results["feature_set"] == "all")
    ]
    if rfr.empty or "param_n_estimators" not in rfr.columns:
        return

    rfr = rfr.copy()
    rfr["param_max_depth"] = rfr["param_max_depth"].astype(str)
    try:
        pivot = (
            rfr.groupby(["param_n_estimators", "param_max_depth"])["wmapE_val"]
            .min()
            .unstack()
        )
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return
        plt.figure(figsize=(10, 7))
        plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(label="wMAPE (val)")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
        plt.xlabel("max_depth")
        plt.ylabel("n_estimators")
        plt.title("Random Forest wMAPE: n_estimators vs max_depth (all features)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                plt.text(
                    j, i, f"{pivot.values[i, j]:.4f}",
                    ha="center", va="center", fontsize=8,
                )
        save_fig(plots_dir / "rfr_heatmap_wmape.png")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Rich console output
# ---------------------------------------------------------------------------

def print_top_results(results: pd.DataFrame, n: int = 15) -> None:
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return

    top = ok.nsmallest(n, "wmapE_val")
    tbl = Table(
        title=f"Top {min(n, len(top))} Configurations (by val wMAPE)",
        title_style="title", expand=True,
    )
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("Model", style="key")
    tbl.add_column("Features", overflow="fold", max_width=20)
    tbl.add_column("Params", overflow="fold", max_width=35)
    tbl.add_column("wMAPE val", justify="right", style="ok")
    tbl.add_column("R² val", justify="right")
    tbl.add_column("RMSE val", justify="right")
    tbl.add_column("MAE val", justify="right")

    p_cols = [c for c in top.columns if c.startswith("param_")]
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        params_str = ", ".join(
            f"{c[6:]}={r[c]}" for c in p_cols if pd.notna(r.get(c))
        )
        tbl.add_row(
            str(rank),
            str(r["model"]),
            str(r["feature_set"]),
            params_str or "—",
            f"{r['wmapE_val']:.4f}",
            f"{r.get('r2_val', float('nan')):.4f}",
            f"{r.get('rmse_val', float('nan')):.4f}",
            f"{r.get('mae_val', float('nan')):.4f}",
        )
    console.print(tbl)


def print_best_per_model(results: pd.DataFrame) -> None:
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return

    tbl = Table(
        title="Best Configuration per Model Type",
        title_style="title", expand=True,
    )
    tbl.add_column("Model", style="key")
    tbl.add_column("R² val", justify="right")
    tbl.add_column("RMSE val", justify="right")
    tbl.add_column("MAE val", justify="right")
    tbl.add_column("wMAPE val", justify="right", style="ok")
    tbl.add_column("R² test", justify="right")
    tbl.add_column("wMAPE test", justify="right")

    for model_name in sorted(ok["model"].unique()):
        grp = ok[ok["model"] == model_name]
        b = grp.loc[grp["wmapE_val"].idxmin()]
        tbl.add_row(
            model_name,
            f"{b.get('r2_val', float('nan')):.4f}",
            f"{b.get('rmse_val', float('nan')):.4f}",
            f"{b.get('mae_val', float('nan')):.4f}",
            f"{b.get('wmapE_val', float('nan')):.4f}",
            f"{b.get('r2_test', float('nan')):.4f}",
            f"{b.get('wmapE_test', float('nan')):.4f}",
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_depth(val: str) -> Optional[int]:
    if val.lower() in ("none", "0"):
        return None
    return int(val)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-model hyperparameter & feature-subset sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python rf_param_train.py --csv diamonds.csv --models rfr
              python rf_param_train.py --csv diamonds.csv --target price \\
                  --models linear ridge lasso elasticnet rfr knn --sweep_features
              python rf_param_train.py --csv diamonds.csv --models rfr knn \\
                  --n_estimators 200 400 800 --n_neighbors 3 5 10 20
        """),
    )

    p.add_argument("--csv", required=True, help="Path to preprocessed CSV")
    p.add_argument("--features", nargs="+", default=None,
                   help="Feature columns (default: all except --target)")
    p.add_argument("--exclude_features", nargs="+", default=None,
                   help="Columns to drop before training (simpler than listing every --features)")
    p.add_argument("--target", default=None,
                   help="Target column (default: last column)")
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--models", nargs="+", required=True, choices=RT_MODEL_CHOICES,
                   help="Model type(s) to sweep")
    p.add_argument("--sweep_features", action="store_true",
                   help="Also sweep feature subsets (leave-one-out / combos)")
    p.add_argument("--min_features", type=int, default=2,
                   help="Minimum features per subset (default: 2)")

    rf = p.add_argument_group("Random Forest grid")
    rf.add_argument("--n_estimators", nargs="+", type=int,
                    default=DEFAULT_N_ESTIMATORS)
    rf.add_argument("--max_depth", nargs="+", type=_parse_depth,
                    default=DEFAULT_MAX_DEPTH,
                    help="Use 0 or 'None' for unlimited")
    rf.add_argument("--min_samples_leaf", nargs="+", type=int,
                    default=DEFAULT_MIN_SAMPLES_LEAF)

    lin = p.add_argument_group("Ridge / Lasso / ElasticNet grid")
    lin.add_argument("--alpha", nargs="+", type=float, default=DEFAULT_ALPHA)
    lin.add_argument("--l1_ratio", nargs="+", type=float, default=DEFAULT_L1_RATIO,
                     help="ElasticNet l1_ratio grid")

    knn = p.add_argument_group("KNN grid")
    knn.add_argument("--n_neighbors", nargs="+", type=int,
                     default=DEFAULT_N_NEIGHBORS)
    knn.add_argument("--knn_weights", nargs="+", default=DEFAULT_KNN_WEIGHTS,
                     choices=["uniform", "distance"])

    p.add_argument("--results_dir", default=".", help="Parent folder for output")
    p.add_argument("--checkpoint_every", type=int, default=50,
                   help="Save partial results every N combos (default: 50)")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv).expanduser().resolve()

    if not csv_path.exists():
        console.print(f"[err]CSV not found:[/err] {csv_path}")
        return 2

    split = tuple(args.split)
    if abs(sum(split) - 1.0) > 1e-6 or any(x <= 0 for x in split):
        console.print("[err]--split must be three positive fractions summing to 1.0[/err]")
        return 2

    run_id = now_stamp()
    outdir = Path(args.results_dir).resolve() / f"sweep_{run_id}"
    safe_mkdir(outdir)

    log = _setup_logger(outdir)
    log.info(f"Sweep started: {run_id}")
    log.info(f"CSV: {csv_path}")
    log.info(f"Models: {args.models}")
    log.info(f"Full args: {vars(args)}")

    # --- Load & split data once ---
    df, target_col = load_and_prepare(
        csv_path, features=args.features, target=args.target,
        exclude=args.exclude_features,
        max_rows=args.max_rows, seed=args.seed,
        shuffle=True,
    )
    all_features = [c for c in df.columns if c != target_col]
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target_col, split)
    del df

    console.print(Panel(
        f"CSV           : [path]{csv_path}[/path]\n"
        f"Target        : [key]{target_col}[/key]\n"
        f"Features ({len(all_features):>2d})  : [dim]{', '.join(all_features)}[/dim]\n"
        f"Models        : [key]{', '.join(args.models)}[/key]\n"
        f"Feat sweep    : {'yes' if args.sweep_features else 'no'}\n"
        f"Split         : train {len(X_train):,}  val {len(X_val):,}  test {len(X_test):,}\n"
        f"Output        : [path]{outdir}[/path]",
        title="rf_param_train — multi-model sweep",
        border_style="cyan",
    ))

    # --- Build grids ---
    model_grids = build_model_grids(
        models=args.models,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        n_neighbors=args.n_neighbors,
        knn_weights=args.knn_weights,
    )
    model_counts = {}
    for m, _ in model_grids:
        model_counts[m] = model_counts.get(m, 0) + 1
    console.print(
        "Model configs : "
        + ", ".join(f"[key]{m}[/key]={n}" for m, n in sorted(model_counts.items()))
    )

    # --- Build feature subsets ---
    if args.sweep_features:
        feature_subsets = generate_feature_subsets(all_features, args.min_features)
        console.print(f"Feature subsets: [key]{len(feature_subsets)}[/key]")
    else:
        feature_subsets = [("all", all_features)]

    t0 = time.perf_counter()

    # --- Run sweep ---
    try:
        results, errors = run_sweep(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            model_grids=model_grids,
            feature_subsets=feature_subsets,
            seed=args.seed,
            outdir=outdir,
            log=log,
            checkpoint_every=args.checkpoint_every,
        )
    except KeyboardInterrupt:
        console.print("\n[warn]Interrupted. Partial results (if any) saved to checkpoint.[/warn]")
        log.warning("Sweep interrupted by user (KeyboardInterrupt)")
        return 130

    # --- Ablation importance ---
    ablation_imp = None
    if args.sweep_features:
        ablation_imp = compute_ablation_importance(results, all_features)

    # --- Retrain best model for full output + feature importance ---
    feature_imp = None
    ok = results[results["status"] == "ok"]
    if not ok.empty:
        best_idx = ok["wmapE_val"].idxmin()
        best_row = ok.loc[best_idx]
        best_model = best_row["model"]
        best_feat_list = best_row["features"].split("+")
        best_params = _parse_best_params(best_row)

        console.print(
            f"\n[title]Retraining best model ({best_model}) "
            f"with full output...[/title]"
        )
        log.info(f"Retraining best: {best_model}, params={best_params}")

        try:
            best_outdir = outdir / "models" / "best"
            feature_imp = extract_model_feature_importance(
                model_name=best_model,
                params=best_params,
                X_train=X_train[best_feat_list],
                y_train=y_train,
                X_val=X_val[best_feat_list],
                y_val=y_val,
                X_test=X_test[best_feat_list],
                y_test=y_test,
                seed=args.seed,
                outdir=best_outdir,
                target_name=target_col,
            )
        except Exception as exc:
            log.error(f"Failed to retrain best model: {exc}\n{traceback.format_exc()}")
            console.print(f"[warn]Could not retrain best model: {exc}[/warn]")

    # --- Generate reports ---
    best_config = generate_reports(
        results=results,
        errors=errors,
        target_col=target_col,
        all_features=all_features,
        feature_imp=feature_imp,
        ablation_imp=ablation_imp,
        outdir=outdir,
        csv_path=csv_path,
        log=log,
    )

    # --- Plots ---
    try:
        plot_model_comparison(results, outdir)
        plot_ablation_importance(ablation_imp, outdir)
        plot_rfr_heatmap(results, outdir)
    except Exception as exc:
        log.error(f"Plot generation failed: {exc}")

    # --- Console summary ---
    print_top_results(results)
    print_best_per_model(results)

    elapsed = time.perf_counter() - t0
    n_ok = len(ok)
    n_err = len(errors)
    console.print(
        f"\n[ok]Done[/ok] in {elapsed:.1f}s — "
        f"{n_ok:,} succeeded, {n_err} errors"
    )
    console.print(f"Results in [path]{outdir}[/path]\n")

    log.info(f"Sweep complete in {elapsed:.1f}s: {n_ok} ok, {n_err} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
