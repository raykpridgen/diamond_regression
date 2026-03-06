#!/usr/bin/env python3
"""
Regression training on a preprocessed CSV.

Trains one or more regression models (Linear, Ridge, Lasso, ElasticNet, RFR, KNN),
evaluates per train/val/test split, saves the fitted sklearn pipeline to disk
(joblib) so it can be reloaded by eval_examine.py, and exports predictions,
feature-importance tables, plots, and a model card for each model.

All core functions are importable by rf_param_train.py for hyperparameter
sweeps — the CLI is just a thin wrapper around the library API.

Features and target are auto-detected from the CSV when not specified:
  - --target defaults to the last column
  - --features defaults to all other columns

Dependencies:
  pip install pandas numpy matplotlib scikit-learn rich joblib

Examples:
  # Single model (auto-detect features/target from CSV)
  python regress_train.py --csv data_clean.csv --models rfr

  # Multiple models with custom split and output directory
  python regress_train.py --csv data_clean.csv --models ridge lasso rfr \
      --split 0.7 0.15 0.15 --results_dir ./results

  # Specify which columns to use
  python regress_train.py --csv data_clean.csv --models rfr \
      --features col1 col2 col3 --target y
"""

from __future__ import annotations

import argparse
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

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

MODEL_CHOICES: List[str] = ["ridge", "lasso", "rfr", "linear", "elasticnet", "knn"]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def robust_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_probably_epoch_ms(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return False
    return float(np.nanmedian(s)) > 1e11


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive pickup_dow and pickup_hour from time_stamp, if that column exists."""
    if "time_stamp" not in df.columns:
        return df

    df = df.copy()
    if is_probably_epoch_ms(df["time_stamp"]):
        ts = pd.to_numeric(df["time_stamp"], errors="coerce")
        dt = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(df["time_stamp"], errors="coerce", utc=True)

    df["pickup_dow"] = dt.dt.dayofweek
    df["pickup_hour"] = dt.dt.hour
    return df


def _detect_target(df: pd.DataFrame) -> str:
    """Pick the last column as the default target."""
    return df.columns[-1]


def load_and_prepare(
    csv_path: Path,
    features: Optional[List[str]] = None,
    target: Optional[str] = None,
    exclude: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Load CSV, optionally shuffle/sample, derive time features, restrict columns.

    Returns (prepared_df, target_column_name).  When *features* or *target*
    are ``None`` they are auto-detected from the CSV header.

    *exclude* drops named columns before feature selection — handy for
    removing a handful of problematic columns without listing every keeper.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)

    # Coerce columns that look numeric but were read as strings
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0.5 * len(df):
                df[col] = coerced

    df = derive_time_features(df)

    if target is None:
        target = _detect_target(df)

    if exclude:
        bad = [c for c in exclude if c not in df.columns]
        if bad:
            raise ValueError(f"--exclude_features column(s) not found in CSV: {bad}")
        if target in exclude:
            raise ValueError(f"Cannot exclude the target column '{target}'")
        df = df.drop(columns=exclude)

    if features is not None:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature(s) not found in CSV: {missing}")
        needed = list(features) + [target]
    else:
        needed = [c for c in df.columns if c != target] + [target]

    return df[needed].copy(), target


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target: str,
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split into (X_train, y_train, X_val, y_val, X_test, y_test), dropping NaN targets."""
    tr, va, _te = split
    n = len(df)
    n_train = int(round(tr * n))
    n_val = int(round(va * n))

    X = df.drop(columns=[target])
    y = df[target].astype(float)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val :], y.iloc[n_train + n_val :]

    for Xs, ys, label in [
        (X_train, y_train, "train"),
        (X_val, y_val, "val"),
        (X_test, y_test, "test"),
    ]:
        mask = ys.notna()
        if label == "train":
            X_train, y_train = Xs.loc[mask], ys.loc[mask]
        elif label == "val":
            X_val, y_val = Xs.loc[mask], ys.loc[mask]
        else:
            X_test, y_test = Xs.loc[mask], ys.loc[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Sklearn preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes + scales numerics and one-hot encodes categoricals."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Weighted MAPE: sum(|err|) / sum(|y_true|).  Returns fraction (0.10 = 10%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom < eps:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "wmapE": wmape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def make_model(name: str, random_state: int = 42, **kwargs: Any) -> Any:
    """Instantiate a regression model by short name, with optional overrides."""
    name = name.lower()
    if name == "linear":
        defaults: Dict[str, Any] = {}
        defaults.update(kwargs)
        return LinearRegression(**defaults)
    if name == "ridge":
        defaults = {"alpha": 1.0}
        defaults.update(kwargs)
        return Ridge(random_state=random_state, **defaults)
    if name == "lasso":
        defaults = {"alpha": 0.01, "max_iter": 80_000}
        defaults.update(kwargs)
        return Lasso(random_state=random_state, **defaults)
    if name == "elasticnet":
        defaults = {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 80_000}
        defaults.update(kwargs)
        return ElasticNet(random_state=random_state, **defaults)
    if name == "rfr":
        defaults = {
            "n_estimators": 400,
            "n_jobs": -1,
            "max_depth": None,
            "min_samples_leaf": 2,
        }
        defaults.update(kwargs)
        return RandomForestRegressor(random_state=random_state, **defaults)
    if name == "knn":
        defaults = {"n_neighbors": 5, "n_jobs": -1}
        defaults.update(kwargs)
        return KNeighborsRegressor(**defaults)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Training & evaluation (core importable function)
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outdir: Optional[Path] = None,
    random_state: int = 42,
    save_model: bool = True,
    plots: bool = True,
    quiet: bool = False,
    target_name: str = "target",
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """
    Train a single model, evaluate on all three splits, and optionally write
    outputs to *outdir*.

    Returns a dict with:
      - train/val/test metrics  (keys like r2_train, rmse_val, …)
      - n_train / n_val / n_test
      - model_path  (Path to saved joblib, or None)
      - pipeline     (the fitted sklearn Pipeline object)
    """
    if len(X_train) < 10 or len(X_val) < 5:
        return {"skipped": True, "reason": "insufficient rows"}

    preprocessor = build_preprocessor(X_train)
    model = make_model(model_name, random_state=random_state, **model_kwargs)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    pred_train = pipe.predict(X_train)
    pred_val = pipe.predict(X_val)
    pred_test = pipe.predict(X_test)

    train_m = evaluate_regression(y_train.values, pred_train)
    val_m = evaluate_regression(y_val.values, pred_val)
    test_m = evaluate_regression(y_test.values, pred_test)

    result: Dict[str, Any] = {
        "model_name": model_name,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "pipeline": pipe,
        "model_path": None,
    }
    for split_label, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        for k, v in m.items():
            result[f"{k}_{split_label}"] = v

    if outdir is None:
        return result

    safe_mkdir(outdir)
    plots_dir = outdir / "plots"
    safe_mkdir(plots_dir)

    # Save model
    if save_model:
        model_path = outdir / "model.joblib"
        joblib.dump(pipe, model_path)
        result["model_path"] = model_path

    # Predictions CSVs
    features_used = X_train.columns.tolist()
    for split_label, Xs, ys, preds in [
        ("train", X_train, y_train, pred_train),
        ("val", X_val, y_val, pred_val),
        ("test", X_test, y_test, pred_test),
    ]:
        out = Xs[features_used].copy()
        out[f"actual_{target_name}"] = ys.values
        out[f"predicted_{target_name}"] = np.asarray(preds, dtype=float)
        out.to_csv(outdir / f"predictions_{split_label}.csv", index=False)

    # Metrics CSV
    metrics_row = {
        "model": model_name,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    for split_label, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        for k, v in m.items():
            metrics_row[f"{k}_{split_label}"] = v
    pd.DataFrame([metrics_row]).to_csv(outdir / "metrics.csv", index=False)

    # Feature importance
    _export_feature_importance(pipe, features_used, model_name, outdir, plots_dir)

    if plots:
        _plot_pred_vs_actual(y_val.values, pred_val, model_name, val_m, plots_dir, target_name)
        _plot_residuals(y_val.values, pred_val, model_name, plots_dir)

    # Model card
    _write_model_card(model_name, features_used, metrics_row, outdir)

    if not quiet:
        _print_metrics_table(model_name, train_m, val_m, test_m)

    return result


# ---------------------------------------------------------------------------
# Output helpers (internal)
# ---------------------------------------------------------------------------

def _export_feature_importance(
    pipe: Pipeline,
    features_used: List[str],
    model_name: str,
    outdir: Path,
    plots_dir: Path,
) -> None:
    prep = pipe.named_steps["prep"]
    feat_names = get_feature_names(prep)
    model_obj = pipe.named_steps["model"]

    coefs = None
    label = "importance"
    if hasattr(model_obj, "coef_"):
        coefs = np.asarray(model_obj.coef_).ravel()
        label = "coefficient"
    elif hasattr(model_obj, "feature_importances_"):
        coefs = np.asarray(model_obj.feature_importances_).ravel()

    if coefs is None:
        return

    if len(feat_names) != len(coefs):
        feat_names = [f"f_{i}" for i in range(len(coefs))]

    imp = pd.DataFrame({"feature": feat_names, label: coefs, f"abs_{label}": np.abs(coefs)})
    imp = imp.sort_values(f"abs_{label}", ascending=False)
    imp.to_csv(outdir / "feature_importance.csv", index=False)

    top = imp.head(25).iloc[::-1]
    plt.figure(figsize=(12, 8))
    plt.barh(top["feature"], top[label])
    plt.title(f"{model_name} | Top {label}s")
    plt.xlabel(label.capitalize())
    plt.ylabel("Feature")
    save_fig(plots_dir / "top_feature_importance.png")


def _plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    val_m: Dict[str, float],
    plots_dir: Path,
    target_name: str = "target",
) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=10, alpha=0.4)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.title(
        f"{model_name} | Val: R²={val_m['r2']:.3f}  "
        f"RMSE={val_m['rmse']:.3f}  wMAPE={val_m['wmapE']:.3f}"
    )
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    save_fig(plots_dir / "val_pred_vs_actual.png")


def _plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    plots_dir: Path,
) -> None:
    resid = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(resid, bins=50, edgecolor="black", linewidth=0.4)
    plt.title(f"{model_name} | Val residuals (pred − actual)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    save_fig(plots_dir / "val_residuals_hist.png")


def _write_model_card(
    model_name: str,
    features: List[str],
    m: Dict[str, Any],
    outdir: Path,
) -> None:
    def f(key: str) -> str:
        v = m.get(key)
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

    card = [
        f"Model : {model_name}",
        f"Features : {', '.join(features)}",
        "",
        f"Train/Val/Test sizes : {m['n_train']}/{m['n_val']}/{m['n_test']}",
        "",
        "            R²        RMSE       MAE        wMAPE",
        f"  Train   {f('r2_train'):>8}   {f('rmse_train'):>8}   {f('mae_train'):>8}   {f('wmapE_train'):>8}",
        f"  Val     {f('r2_val'):>8}   {f('rmse_val'):>8}   {f('mae_val'):>8}   {f('wmapE_val'):>8}",
        f"  Test    {f('r2_test'):>8}   {f('rmse_test'):>8}   {f('mae_test'):>8}   {f('wmapE_test'):>8}",
        "",
        "Saved artefacts:",
        "  model.joblib, metrics.csv, feature_importance.csv",
        "  predictions_train.csv, predictions_val.csv, predictions_test.csv",
        "  plots/val_pred_vs_actual.png, plots/val_residuals_hist.png",
    ]
    write_text(outdir / "MODEL_CARD.txt", "\n".join(card))


def _print_metrics_table(
    name: str,
    train_m: Dict[str, float],
    val_m: Dict[str, float],
    test_m: Dict[str, float],
) -> None:
    tbl = Table(title=f"[key]{name}[/key] — Regression Metrics", title_style="title", expand=False)
    tbl.add_column("Split", style="key")
    tbl.add_column("R²", justify="right")
    tbl.add_column("RMSE", justify="right")
    tbl.add_column("MAE", justify="right")
    tbl.add_column("wMAPE", justify="right")
    for label, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        tbl.add_row(
            label,
            f"{m['r2']:.4f}",
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['wmapE']:.4f}",
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train regression models on a preprocessed CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python regress_train.py --csv data_clean.csv --models rfr
              python regress_train.py --csv data_clean.csv --models ridge lasso rfr \\
                  --split 0.7 0.15 0.15 --results_dir ./results
        """),
    )
    p.add_argument("--csv", required=True, help="Path to preprocessed CSV")
    p.add_argument("--models", nargs="+", required=True, choices=MODEL_CHOICES,
                   help="Model(s) to train")
    p.add_argument("--features", nargs="+", default=None,
                   help="Feature columns (default: all columns except --target)")
    p.add_argument("--exclude_features", nargs="+", default=None,
                   help="Columns to drop before training (simpler than listing every --features)")
    p.add_argument("--target", default=None,
                   help="Target column (default: last column in CSV)")
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train/val/test fractions (must sum to 1.0)")
    p.add_argument("--results_dir", default=".", help="Parent folder for results_<timestamp>/")
    p.add_argument("--max_rows", type=int, default=None, help="Subsample after shuffle")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Skip shuffle (use when CSV is already randomised)")
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

    features = args.features
    target = args.target

    t0 = time.perf_counter()

    df, target = load_and_prepare(
        csv_path,
        features=features,
        target=target,
        exclude=args.exclude_features,
        max_rows=args.max_rows,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    feature_cols = [c for c in df.columns if c != target]

    console.print(Panel(
        f"CSV     : [path]{csv_path}[/path]\n"
        f"Models  : [key]{', '.join(args.models)}[/key]\n"
        f"Features: [dim]{', '.join(feature_cols)}[/dim]\n"
        f"Target  : [key]{target}[/key]\n"
        f"Split   : {split[0]:.0%} / {split[1]:.0%} / {split[2]:.0%}",
        title="regress_train",
        border_style="cyan",
    ))

    console.print(f"Loaded [key]{len(df):,}[/key] rows, [key]{df.shape[1]}[/key] columns")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target, split)
    console.print(
        f"Split → train [key]{len(X_train):,}[/key]  "
        f"val [key]{len(X_val):,}[/key]  "
        f"test [key]{len(X_test):,}[/key]"
    )

    run_id = now_stamp()
    run_dir = Path(args.results_dir).resolve() / f"results_{run_id}"
    safe_mkdir(run_dir)

    summary_rows = []

    for model_name in args.models:
        model_dir = run_dir / "models" / model_name
        console.print(f"\n[title]Training {model_name}…[/title]")

        result = train_and_evaluate(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            outdir=model_dir,
            random_state=args.seed,
            target_name=target,
        )

        if result.get("skipped"):
            console.print(f"[warn]Skipped {model_name}:[/warn] {result.get('reason')}")
            continue

        summary_rows.append({
            "model": model_name,
            "r2_val": result.get("r2_val"),
            "rmse_val": result.get("rmse_val"),
            "mae_val": result.get("mae_val"),
            "wmapE_val": result.get("wmapE_val"),
        })

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values("r2_val", ascending=False)
        summary.to_csv(run_dir / "model_summary.csv", index=False)

    elapsed = time.perf_counter() - t0
    console.print(f"\n[ok]Done[/ok] in {elapsed:.1f}s — results in [path]{run_dir}[/path]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
