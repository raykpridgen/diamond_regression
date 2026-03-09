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
    --n_estimators 50 100 200 800 1200
    --max_depth 5 10 20 30 0
    --min_samples_leaf 1 2 4 8 32
    --exclude_features number
    --target price
    --split 0.7 0.15 0.15
    --sweep_features
    --n_neighbors 2 5 10 30
    --knn_weights uniform distance
    --l1_ratio 0.1 0.33 0.66 0.85 1
    --alpha 0.1 0.3 0.5 1.0 1.5 2.0
)

echo "=== Run 1/3: x y z ==="
python3 rf_param_train.py --features x y z "${COMMON[@]}"

echo "=== Run 2/3: carat cut color clarity depth table ==="
python3 rf_param_train.py --features carat cut color clarity depth table "${COMMON[@]}"

echo "=== Run 3/3: all 9 features ==="
python3 rf_param_train.py --features carat cut color clarity depth table x y z "${COMMON[@]}"

echo "=== All sweeps complete ==="

if "$SHUTDOWN"; then
    echo "Shutting down now..."
    sudo shutdown -h now
fi
