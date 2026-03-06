#!/usr/bin/env python3
"""
Evaluate a saved model from regress_train.py on test (or any) data.

Loads a joblib pipeline, runs inference, prints detailed metrics, and produces
diagnostic analysis: residual distribution, worst predictions, per-category
breakdowns, and prediction-error quantiles.

Dependencies:
  pip install pandas numpy matplotlib scikit-learn rich joblib

Examples:
  # Basic evaluation
  python eval_examine.py --model results/.../models/rfr/model.joblib \
      --csv cab_rides_clean.csv

  # Only evaluate on the last 15% of the CSV (test portion)
  python eval_examine.py --model results/.../models/rfr/model.joblib \
      --csv cab_rides_clean.csv --split_pos 0.85

  # Show the 20 worst predictions
  python eval_examine.py --model results/.../models/rfr/model.joblib \
      --csv cab_rides_clean.csv --worst 20

  # Break down errors by a categorical column
  python eval_examine.py --model results/.../models/rfr/model.joblib \
      --csv cab_rides_clean.csv --group_by cab_type
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from regress_train import (
    derive_time_features,
    evaluate_regression,
    save_fig,
    safe_mkdir,
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
# Loading
# ---------------------------------------------------------------------------

def load_pipeline(model_path: Path) -> Any:
    return joblib.load(model_path)


def prepare_eval_data(
    csv_path: Path,
    target: str,
    split_pos: Optional[float],
    max_rows: Optional[int],
) -> pd.DataFrame:
    """Load CSV, derive time features, optionally take the tail portion."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = derive_time_features(df)

    if split_pos is not None and 0 < split_pos < 1.0:
        start = int(len(df) * split_pos)
        df = df.iloc[start:].reset_index(drop=True)

    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)

    df = df.dropna(subset=[target]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    base = evaluate_regression(y_true, y_pred)
    residuals = y_pred - y_true
    abs_err = np.abs(residuals)
    base.update({
        "median_abs_error": float(np.median(abs_err)),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "max_overpredict": float(np.max(residuals)),
        "max_underpredict": float(np.min(residuals)),
    })
    return base


def worst_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target: str,
    n: int = 10,
) -> pd.DataFrame:
    out = df.copy()
    out["predicted"] = y_pred
    out["abs_error"] = np.abs(y_pred - out[target].values)
    return out.nlargest(n, "abs_error")


def grouped_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target: str,
    group_col: str,
) -> pd.DataFrame:
    work = df.copy()
    work["_pred"] = y_pred
    rows = []
    for name, grp in work.groupby(group_col, dropna=True):
        yt = grp[target].values
        yp = grp["_pred"].values
        m = evaluate_regression(yt, yp)
        m["group"] = str(name)
        m["n"] = len(grp)
        rows.append(m)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("wmapE", ascending=True).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eval_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict, outdir: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=10, alpha=0.3)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.title(f"Eval: R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  wMAPE={metrics['wmapE']:.3f}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    save_fig(outdir / "eval_pred_vs_actual.png")


def plot_eval_residuals(y_true: np.ndarray, y_pred: np.ndarray, outdir: Path) -> None:
    resid = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(resid, bins=60, edgecolor="black", linewidth=0.4)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residual Distribution (pred − actual)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    save_fig(outdir / "eval_residuals_hist.png")


def plot_error_by_actual(y_true: np.ndarray, y_pred: np.ndarray, outdir: Path) -> None:
    abs_err = np.abs(y_pred - y_true)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, abs_err, s=8, alpha=0.3)
    plt.title("Absolute Error vs Actual Value")
    plt.xlabel("Actual")
    plt.ylabel("|Error|")
    save_fig(outdir / "eval_abs_error_vs_actual.png")


# ---------------------------------------------------------------------------
# Rich tables
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict[str, float], n: int) -> None:
    tbl = Table(title="Evaluation Metrics", title_style="title", expand=False)
    tbl.add_column("Metric", style="key")
    tbl.add_column("Value", justify="right")

    tbl.add_row("N (rows)", f"{n:,}")
    tbl.add_row("R²", f"{metrics['r2']:.4f}")
    tbl.add_row("RMSE", f"{metrics['rmse']:.4f}")
    tbl.add_row("MAE", f"{metrics['mae']:.4f}")
    tbl.add_row("wMAPE", f"{metrics['wmapE']:.4f}")
    tbl.add_row("Median |Error|", f"{metrics['median_abs_error']:.4f}")
    tbl.add_row("P90 |Error|", f"{metrics['p90_abs_error']:.4f}")
    tbl.add_row("P95 |Error|", f"{metrics['p95_abs_error']:.4f}")
    tbl.add_row("P99 |Error|", f"{metrics['p99_abs_error']:.4f}")
    tbl.add_row("Mean Residual", f"{metrics['mean_residual']:.4f}")
    tbl.add_row("Std Residual", f"{metrics['std_residual']:.4f}")
    tbl.add_row("Max Over-predict", f"{metrics['max_overpredict']:.4f}")
    tbl.add_row("Max Under-predict", f"{metrics['max_underpredict']:.4f}")

    console.print(tbl)


def print_worst(worst_df: pd.DataFrame, target: str) -> None:
    tbl = Table(title="Worst Predictions (highest |error|)", title_style="warn", expand=True, show_lines=True)
    tbl.add_column("#", justify="right", style="dim")
    display_cols = [c for c in worst_df.columns if c not in ("_pred",)]
    for col in display_cols:
        tbl.add_column(col, overflow="fold")
    for rank, (_, row) in enumerate(worst_df.iterrows(), 1):
        cells = []
        for col in display_cols:
            val = row[col]
            if col == "abs_error":
                cells.append(f"[err]{val:.2f}[/err]")
            elif col == "predicted":
                cells.append(f"{val:.2f}")
            elif col == target:
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        tbl.add_row(str(rank), *cells)
    console.print(tbl)


def print_grouped(grouped_df: pd.DataFrame, group_col: str) -> None:
    tbl = Table(title=f"Metrics by {group_col}", title_style="title", expand=True)
    tbl.add_column(group_col, style="key")
    tbl.add_column("N", justify="right")
    tbl.add_column("R²", justify="right")
    tbl.add_column("RMSE", justify="right")
    tbl.add_column("MAE", justify="right")
    tbl.add_column("wMAPE", justify="right")

    for _, r in grouped_df.iterrows():
        tbl.add_row(
            str(r["group"]),
            f"{int(r['n']):,}",
            f"{r['r2']:.4f}",
            f"{r['rmse']:.4f}",
            f"{r['mae']:.4f}",
            f"{r['wmapE']:.4f}",
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load a saved model and evaluate on test data with detailed diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python eval_examine.py --model ./results/.../models/rfr/model.joblib --csv data.csv
              python eval_examine.py --model model.joblib --csv data.csv --split_pos 0.85 --worst 20
              python eval_examine.py --model model.joblib --csv data.csv --group_by cab_type
        """),
    )
    p.add_argument("--model", required=True, help="Path to saved model.joblib")
    p.add_argument("--csv", required=True, help="Path to CSV (same schema as training data)")
    p.add_argument("--target", default=None,
                   help="Target column (default: last column in CSV)")
    p.add_argument("--split_pos", type=float, default=None,
                   help="Use rows from this fraction onward as eval data (e.g. 0.85 = last 15%%)")
    p.add_argument("--max_rows", type=int, default=None, help="Cap eval rows")
    p.add_argument("--worst", type=int, default=10,
                   help="Show N worst predictions (default: 10)")
    p.add_argument("--group_by", nargs="+", default=None,
                   help="Break down metrics by these categorical column(s)")
    p.add_argument("--outdir", default=None,
                   help="Write plots and report to this directory (default: alongside model file)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()

    if not model_path.exists():
        console.print(f"[err]Model not found:[/err] {model_path}")
        return 2
    if not csv_path.exists():
        console.print(f"[err]CSV not found:[/err] {csv_path}")
        return 2

    if args.target is None:
        header = pd.read_csv(csv_path, nrows=0)
        args.target = header.columns[-1].strip()
        console.print(f"Auto-detected target: [key]{args.target}[/key]")

    console.print(Panel(
        f"Model  : [path]{model_path}[/path]\n"
        f"CSV    : [path]{csv_path}[/path]\n"
        f"Target : [key]{args.target}[/key]"
        + (f"\nSlice  : rows from {args.split_pos:.0%} onward" if args.split_pos else ""),
        title="eval_examine",
        border_style="cyan",
    ))

    pipe = load_pipeline(model_path)
    console.print("[ok]Model loaded.[/ok]")

    df = prepare_eval_data(csv_path, args.target, args.split_pos, args.max_rows)
    console.print(f"Eval rows: [key]{len(df):,}[/key]")

    if len(df) == 0:
        console.print("[err]No rows with non-null target to evaluate.[/err]")
        return 1

    expected_features = pipe.feature_names_in_ if hasattr(pipe, "feature_names_in_") else None
    if expected_features is not None:
        feat_list = list(expected_features)
    else:
        prep = pipe.named_steps.get("prep")
        if prep is not None:
            feat_list = []
            for name, _trans, cols in prep.transformers_:
                if isinstance(cols, list):
                    feat_list.extend(cols)
                elif isinstance(cols, str):
                    feat_list.append(cols)
        else:
            feat_list = [c for c in df.columns if c != args.target]

    missing_feats = [f for f in feat_list if f not in df.columns]
    if missing_feats:
        console.print(f"[warn]Missing feature columns in CSV:[/warn] {', '.join(missing_feats)}")
        for f in missing_feats:
            df[f] = np.nan

    X = df[feat_list]
    y = df[args.target].astype(float).values
    y_pred = pipe.predict(X)

    # --- Metrics ---
    metrics = detailed_metrics(y, y_pred)
    print_metrics(metrics, len(df))

    # --- Output directory ---
    if args.outdir:
        outdir = Path(args.outdir).resolve()
    else:
        outdir = model_path.parent / "eval"
    safe_mkdir(outdir)

    # --- Plots ---
    plot_eval_pred_vs_actual(y, y_pred, metrics, outdir)
    plot_eval_residuals(y, y_pred, outdir)
    plot_error_by_actual(y, y_pred, outdir)
    console.print(f"Plots written to [path]{outdir}[/path]")

    # --- Worst predictions ---
    worst_df = worst_predictions(df, y_pred, args.target, n=args.worst)
    print_worst(worst_df, args.target)

    # --- Grouped metrics ---
    if args.group_by:
        for gc in args.group_by:
            if gc not in df.columns:
                console.print(f"[warn]Column '{gc}' not in data — skipping group_by[/warn]")
                continue
            gm = grouped_metrics(df, y_pred, args.target, gc)
            if not gm.empty:
                print_grouped(gm, gc)
                gm.to_csv(outdir / f"metrics_by_{gc}.csv", index=False)

    # --- Text report ---
    report_lines = [
        f"Model : {model_path}",
        f"CSV   : {csv_path}",
        f"N     : {len(df):,}",
        "",
    ]
    for k, v in metrics.items():
        report_lines.append(f"  {k:25s}: {v:.4f}")
    write_text(outdir / "eval_report.txt", "\n".join(report_lines))

    console.print(f"\n[ok]Evaluation complete.[/ok]  Report in [path]{outdir}[/path]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
