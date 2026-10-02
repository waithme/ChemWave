#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR=${DATA_DIR:-/home/jinxia1/Desktop/another_cliff_model/downstream_data/experiment2}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/target_balanced_runs}

cd "$PROJECT_DIR"
exec /home/jinxia1/miniconda3/envs/temp_env/bin/python run_final_model.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda
