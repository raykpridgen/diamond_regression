#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="$SCRIPT_DIR/sweep_output.log"
SHUTDOWN=false

for arg in "$@"; do
    case "$arg" in
        --unattended) SHUTDOWN=true ;;
    esac
done

if "$SHUTDOWN" && [ -z "${_SWEEPS_INNER:-}" ]; then
    export _SWEEPS_INNER=1
    if command -v tmux &>/dev/null; then
        tmux new-session -d -s sweeps "bash '$0' --unattended 2>&1 | tee '$LOGFILE'"
        echo "Running in tmux session 'sweeps' — log: $LOGFILE"
        echo "Reattach with: tmux attach -t sweeps"
    else
        nohup bash "$0" --unattended > "$LOGFILE" 2>&1 &
        echo "Running in background (PID $!) — log: $LOGFILE"
    fi
    echo "Machine will shut down when sweeps finish."
    exit 0
fi

cd "$SCRIPT_DIR"

COMMON=(
    --csv diamonds.csv
    --models rfr elasticnet knn
    --n_estimators 50 100 300 800
    --max_depth 5 10 20 30 0
    --min_samples_leaf 1 2 4 8 16
    --exclude_features number
    --target price
    --split 0.7 0.15 0.15
    --sweep_features
    --n_neighbors 2 5 10 30
    --knn_weights uniform distance
    --l1_ratio 0.1 0.33 0.66 0.85 1
    --alpha 0.1 0.3 0.5 1.0 1.5 2.0
)

SIMPLE=(
    --csv diamonds.csv
    --models rfr knn
    --n_estimators 800 1200 1600 2000
    --max_depth 20 50 70 0
    --min_samples_leaf 1 2 4
    --features clarity color carat y x
    --target price
    --split 0.7 0.15 0.15
    --sweep_features
    --n_neighbors 1 2 3 5 10 30
    --knn_weights uniform distance
)

echo "=== Run 1/3: dimensions ==="
python3 rf_param_train.py --run_name local "${SIMPLE[@]}"

# echo "=== Run 2/3: qualities ==="
# python3 rf_param_train.py --run_name qualities --features carat cut color clarity depth table "${COMMON[@]}"

# echo "=== Run 3/3: all_features ==="
# python3 rf_param_train.py --run_name all_features --features carat cut color clarity depth table x y z "${COMMON[@]}"

echo "=== All sweeps complete — aggregating run records ==="

python3 - <<'PYEOF'
import json, glob, csv, os

records = []
for f in sorted(glob.glob("run_records/run_record_*.json")):
    with open(f) as fh:
        records.append(json.load(fh))

if not records:
    print("No run records found in run_records/")
    raise SystemExit(0)

# ---- aggregate_summary.csv: one row per run, best-overall metrics ----
summary_rows = []
for r in records:
    best = r.get("best_overall") or {}
    metrics = best.get("metrics") or {}
    row = {
        "run_name":       r.get("run_name", ""),
        "run_id":         r.get("run_id", ""),
        "timestamp":      r.get("timestamp", ""),
        "features":       ", ".join(r.get("data", {}).get("features", [])),
        "n_features":     r.get("data", {}).get("n_features"),
        "best_model":     best.get("model", ""),
        "r2_val":         metrics.get("r2_val"),
        "rmse_val":       metrics.get("rmse_val"),
        "mae_val":        metrics.get("mae_val"),
        "wmapE_val":      metrics.get("wmapE_val"),
        "r2_test":        metrics.get("r2_test"),
        "rmse_test":      metrics.get("rmse_test"),
        "mae_test":       metrics.get("mae_test"),
        "wmapE_test":     metrics.get("wmapE_test"),
        "total_combos":   r.get("counts", {}).get("total_combos"),
        "successful":     r.get("counts", {}).get("successful"),
        "errors":         r.get("counts", {}).get("errors"),
        "elapsed_s":      r.get("elapsed_s"),
    }
    summary_rows.append(row)

with open("run_records/aggregate_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader()
    w.writerows(summary_rows)

# ---- aggregate_best_per_model.csv: best config per model across all runs ----
bpm_rows = []
for r in records:
    for bpm in r.get("best_per_model", []):
        metrics = bpm.get("metrics", {})
        params  = bpm.get("params", {})
        row = {
            "run_name":     r.get("run_name", ""),
            "run_id":       r.get("run_id", ""),
            "model":        bpm.get("model", ""),
            "feature_set":  bpm.get("feature_set", ""),
            "n_features":   bpm.get("n_features"),
            "n_combos":     bpm.get("n_combos_evaluated"),
        }
        row.update({k: v for k, v in metrics.items()})
        row.update({f"param_{k}": v for k, v in params.items()})
        bpm_rows.append(row)

if bpm_rows:
    all_keys = list(dict.fromkeys(k for row in bpm_rows for k in row))
    with open("run_records/aggregate_best_per_model.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(bpm_rows)

# ---- aggregate_feature_importance.csv: importance from each run's best model ----
imp_rows = []
for r in records:
    for feat in r.get("feature_importance", []):
        row = {"run_name": r.get("run_name", ""), "run_id": r.get("run_id", "")}
        row.update(feat)
        imp_rows.append(row)

if imp_rows:
    all_keys = list(dict.fromkeys(k for row in imp_rows for k in row))
    with open("run_records/aggregate_feature_importance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(imp_rows)

print(f"Aggregated {len(records)} run(s) into run_records/:")
print(f"  aggregate_summary.csv              ({len(summary_rows)} runs)")
print(f"  aggregate_best_per_model.csv       ({len(bpm_rows)} entries)")
print(f"  aggregate_feature_importance.csv   ({len(imp_rows)} entries)")
PYEOF

if "$SHUTDOWN"; then
    echo "Shutting down now..."
    sudo shutdown -h now
fi
