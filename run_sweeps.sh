#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

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

echo "=== All sweeps complete — shutting down ==="
sudo shutdown -h now
