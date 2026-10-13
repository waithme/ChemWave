#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR=${DATA_DIR:-${CHEMWAVE_DATA_DIR:-$PROJECT_DIR/../ablation_experiment/data/experiment2}}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/target_balanced_runs}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" run_final_model.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda
