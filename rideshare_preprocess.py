#!/usr/bin/env python3
"""
Rideshare CSV preprocessor — inspect and clean data before modeling.

Two subcommands:
  audit  — read-only scan: missing-value counts, affected rows, hidden nulls
  clean  — produce a cleaned CSV (copy to new file, or --inplace)

Dependencies:
  pip install pandas numpy rich

Examples:
  # Quick audit
  python rideshare_preprocess.py audit --csv rideshare.csv

  # Audit specific columns, list the first 30 affected rows
  python rideshare_preprocess.py audit --csv rideshare.csv --columns price distance --show-rows --limit 30

  # Clean: drop rows missing 'price', median-fill numerics, write a copy
  python rideshare_preprocess.py clean --csv rideshare.csv -o rideshare_clean.csv \
      --drop-missing-target --fill-median --fill-mode

  # Clean in-place (modifies the original file)
  python rideshare_preprocess.py clean --csv rideshare.csv --inplace --drop-missing-target
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

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

HIDDEN_NULL_TOKENS = {"", "NA", "N/A", "n/a", "null", "NULL", "None", "none", "?", "-", "--", "nan", "NaN"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _robust_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_probably_epoch_ms(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return False
    return float(np.nanmedian(s)) > 1e11


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive pickup_dow and pickup_hour from a time_stamp column."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in ("distance", "price", "surge_multiplier"):
        if col in df.columns:
            df[col] = _robust_numeric(df[col])

    if "time_stamp" not in df.columns:
        return df

    if _is_probably_epoch_ms(df["time_stamp"]):
        ts = pd.to_numeric(df["time_stamp"], errors="coerce")
        dt = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(df["time_stamp"], errors="coerce", utc=True)

    df["pickup_dow"] = dt.dt.dayofweek
    df["pickup_hour"] = dt.dt.hour
    return df


def detect_hidden_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Find cells that look like nulls but weren't parsed as NaN."""
    records = []
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].isin(HIDDEN_NULL_TOKENS) | df[col].str.strip().eq("")
        count = int(mask.sum())
        if count:
            records.append({"column": col, "hidden_null_count": count})
    return pd.DataFrame(records)


def missing_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns if columns else df.columns.tolist()
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        null_count = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "total": len(df),
            "missing": null_count,
            "missing_pct": round(null_count / max(len(df), 1) * 100, 2),
            "present": len(df) - null_count,
        })
    return pd.DataFrame(rows).sort_values("missing", ascending=False).reset_index(drop=True)


def rows_with_any_missing(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = [c for c in (columns or df.columns) if c in df.columns]
    mask = df[cols].isna().any(axis=1)
    sub = df.loc[mask].copy()
    sub.insert(0, "orig_row", sub.index)
    return sub


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def cmd_audit(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        console.print(f"[err]File not found:[/err] {csv_path}")
        return 2

    df = load_csv(csv_path)
    columns: Optional[List[str]] = args.columns if args.columns else None

    console.print(Panel(
        f"[path]{csv_path}[/path]\nRows: [key]{len(df):,}[/key]  Columns: [key]{df.shape[1]}[/key]",
        title="CSV Audit",
        border_style="cyan",
    ))

    # --- per-column missing summary ---
    summary = missing_summary(df, columns)
    tbl = Table(title="Missing-Value Summary", title_style="title", expand=True)
    tbl.add_column("Column", style="key")
    tbl.add_column("Dtype", style="dim")
    tbl.add_column("Total", justify="right")
    tbl.add_column("Missing", justify="right")
    tbl.add_column("Missing %", justify="right")
    tbl.add_column("Present", justify="right")

    for _, r in summary.iterrows():
        style = "err" if r["missing_pct"] > 10 else ("warn" if r["missing_pct"] > 0 else "ok")
        tbl.add_row(
            r["column"], r["dtype"],
            f"{r['total']:,}", f"[{style}]{r['missing']:,}[/{style}]",
            f"[{style}]{r['missing_pct']:.2f}%[/{style}]",
            f"{r['present']:,}",
        )
    console.print(tbl)

    # --- rows affected ---
    affected = rows_with_any_missing(df, columns if columns else None)
    total_affected = len(affected)
    console.print(
        f"\nRows with [warn]any[/warn] missing value"
        + (f" (in {', '.join(columns)})" if columns else "")
        + f": [key]{total_affected:,}[/key] / {len(df):,}"
        + f" ([key]{total_affected / max(len(df), 1) * 100:.2f}%[/key])"
    )

    if args.show_rows:
        limit = args.limit or 20
        display = affected.head(limit)
        row_tbl = Table(
            title=f"First {min(limit, total_affected)} rows with missing data",
            title_style="title",
            expand=True,
            show_lines=True,
        )
        row_tbl.add_column("Row #", style="key", justify="right")
        show_cols = [c for c in (columns or df.columns) if c in df.columns]
        for col in show_cols:
            row_tbl.add_column(col)

        for _, r in display.iterrows():
            cells = []
            for col in show_cols:
                val = r[col]
                if pd.isna(val):
                    cells.append("[err]<NULL>[/err]")
                else:
                    cells.append(str(val))
            row_tbl.add_row(str(r["orig_row"]), *cells)

        console.print(row_tbl)
        if total_affected > limit:
            console.print(f"[dim]… {total_affected - limit:,} more rows not shown (use --limit to increase)[/dim]")

    # --- hidden nulls ---
    hidden = detect_hidden_nulls(df)
    if not hidden.empty:
        console.print()
        h_tbl = Table(title="Hidden Nulls (strings that look like missing data)", title_style="warn", expand=True)
        h_tbl.add_column("Column", style="key")
        h_tbl.add_column("Count", justify="right", style="warn")
        for _, r in hidden.iterrows():
            h_tbl.add_row(r["column"], f"{r['hidden_null_count']:,}")
        console.print(h_tbl)
    else:
        console.print("\n[ok]No hidden null tokens detected in string columns.[/ok]")

    # --- duplicate rows ---
    dup_count = int(df.duplicated().sum())
    if dup_count:
        console.print(f"\n[warn]Duplicate rows:[/warn] [key]{dup_count:,}[/key]")
    else:
        console.print(f"\n[ok]No duplicate rows.[/ok]")

    # --- completely empty rows ---
    all_null = int(df.isna().all(axis=1).sum())
    if all_null:
        console.print(f"[err]Completely empty rows:[/err] [key]{all_null:,}[/key]")

    console.print()
    return 0


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
def cmd_clean(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        console.print(f"[err]File not found:[/err] {csv_path}")
        return 2

    if not args.inplace and not args.output:
        console.print("[err]Provide --output <path> or --inplace[/err]")
        return 2

    df = load_csv(csv_path)
    original_len = len(df)
    actions: List[str] = []

    # --- normalize hidden nulls to real NaN ---
    if args.normalize_nulls:
        for col in df.select_dtypes(include="object").columns:
            mask = df[col].isin(HIDDEN_NULL_TOKENS) | df[col].str.strip().eq("")
            converted = int(mask.sum())
            if converted:
                df.loc[mask, col] = np.nan
                actions.append(f"Converted {converted:,} hidden nulls → NaN in '{col}'")

    # --- drop rows where target column is missing ---
    target_col = args.target or "price"
    if args.drop_missing_target and target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            actions.append(f"Dropped {dropped:,} rows missing '{target_col}'")

    # --- drop completely empty rows ---
    if args.drop_empty:
        before = len(df)
        df = df.dropna(how="all").reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            actions.append(f"Dropped {dropped:,} completely empty rows")

    # --- drop specific columns ---
    if args.drop_columns:
        existing = [c for c in args.drop_columns if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            actions.append(f"Dropped columns: {', '.join(existing)}")

    # --- fill numeric columns with median ---
    if args.fill_median:
        for col in df.select_dtypes(include=[np.number]).columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                actions.append(f"Filled {n_missing:,} NaN in '{col}' with median ({median_val:.4g})")

    # --- fill categorical columns with mode ---
    if args.fill_mode:
        for col in df.select_dtypes(include="object").columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                    actions.append(f"Filled {n_missing:,} NaN in '{col}' with mode ('{mode_val.iloc[0]}')")

    # --- drop duplicate rows ---
    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            actions.append(f"Dropped {dropped:,} duplicate rows")

    # --- derive time features from time_stamp ---
    if args.add_time_features:
        if "time_stamp" in df.columns:
            df = add_time_features(df)
            new_cols = [c for c in ("pickup_dow", "pickup_hour") if c in df.columns]
            actions.append(f"Derived time features: {', '.join(new_cols)}")
        else:
            actions.append("Skipped --add-time-features (no 'time_stamp' column found)")

    # --- write output ---
    if args.inplace:
        out_path = csv_path
    else:
        out_path = Path(args.output).expanduser().resolve()

    df.to_csv(out_path, index=False)

    console.print(Panel(
        f"[path]{csv_path}[/path] → [path]{out_path}[/path]",
        title="Clean Complete",
        border_style="ok",
    ))

    if actions:
        tbl = Table(title="Actions Applied", title_style="title", expand=True)
        tbl.add_column("#", justify="right", style="dim")
        tbl.add_column("Action")
        for i, a in enumerate(actions, 1):
            tbl.add_row(str(i), a)
        console.print(tbl)
    else:
        console.print("[warn]No cleaning actions were requested — output is a verbatim copy.[/warn]")

    remaining_missing = int(df.isna().sum().sum())
    console.print(
        f"\nRows: [key]{original_len:,}[/key] → [key]{len(df):,}[/key]"
        f"  (removed [warn]{original_len - len(df):,}[/warn])"
        f"\nRemaining NaN cells: [key]{remaining_missing:,}[/key]"
    )
    console.print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and clean rideshare CSV data before modeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            workflow examples:
              # 1. See what's missing
              python rideshare_preprocess.py audit --csv cab_rides.csv

              # 2. Zoom into specific columns + show row numbers
              python rideshare_preprocess.py audit --csv cab_rides.csv \\
                  --columns price distance surge_multiplier --show-rows --limit 50

              # 3. Clean to a new file (with time-feature derivation)
              python rideshare_preprocess.py clean --csv cab_rides.csv -o cab_rides_clean.csv \\
                  --drop-missing-target --normalize-nulls --fill-median --fill-mode \\
                  --add-time-features

              # 4. Train models on the cleaned file
              python regress_train.py --csv cab_rides_clean.csv --models rfr

              # 5. Sweep RF hyperparameters
              python rf_param_train.py --csv cab_rides_clean.csv --sweep_features

              # 6. Evaluate a saved model
              python eval_examine.py --model results/.../models/rfr/model.joblib \\
                  --csv cab_rides_clean.csv
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- audit --
    audit_p = sub.add_parser("audit", help="Read-only scan for missing / suspect data")
    audit_p.add_argument("--csv", required=True, help="Path to CSV file")
    audit_p.add_argument("--columns", nargs="+", default=None,
                         help="Only inspect these columns (default: all)")
    audit_p.add_argument("--show-rows", action="store_true",
                         help="Print actual rows that contain missing values")
    audit_p.add_argument("--limit", type=int, default=20,
                         help="Max rows to display when --show-rows is set (default: 20)")

    # -- clean --
    clean_p = sub.add_parser("clean", help="Produce a cleaned CSV (copy or in-place)")
    clean_p.add_argument("--csv", required=True, help="Path to input CSV")
    clean_p.add_argument("-o", "--output", default=None,
                         help="Path for cleaned output CSV")
    clean_p.add_argument("--inplace", action="store_true",
                         help="Overwrite the original file instead of writing a copy")
    clean_p.add_argument("--target", default="price",
                         help="Name of the target column (default: price)")
    clean_p.add_argument("--normalize-nulls", action="store_true",
                         help="Convert hidden null tokens ('NA', '', '-', etc.) to real NaN")
    clean_p.add_argument("--drop-missing-target", action="store_true",
                         help="Drop rows where the target column is NaN")
    clean_p.add_argument("--drop-empty", action="store_true",
                         help="Drop rows that are completely empty")
    clean_p.add_argument("--drop-columns", nargs="+", default=None,
                         help="Drop these columns entirely")
    clean_p.add_argument("--fill-median", action="store_true",
                         help="Fill NaN in numeric columns with the column median")
    clean_p.add_argument("--fill-mode", action="store_true",
                         help="Fill NaN in categorical columns with the column mode")
    clean_p.add_argument("--drop-duplicates", action="store_true",
                         help="Remove duplicate rows")
    clean_p.add_argument("--add-time-features", action="store_true",
                         help="Derive pickup_dow and pickup_hour from time_stamp column")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "clean":
        return cmd_clean(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
