#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-houston}"
MODEL="${2:-hyperprompt_coop}"
SEED="${3:-42}"
RUN_ID="${4:-exp001}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ARGS=(
  --dataset "${DATASET}"
  --model "${MODEL}"
  --seed "${SEED}"
  --run-id "${RUN_ID}"
)

python train.py "${ARGS[@]}"
python test.py "${ARGS[@]}"
