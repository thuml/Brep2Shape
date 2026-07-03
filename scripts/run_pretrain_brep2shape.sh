#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-brep2shape}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DATASET_DIR="${DATASET_DIR:-/path/to/brep2shape_data}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-brep2shape}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-4}"
export DROPOUT="${DROPOUT:-0.0}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export SEED="${SEED:-0}"

export GRAPH_NUM_LAYERS="${GRAPH_NUM_LAYERS:-6}"
export SURFACE_NUM_LAYERS="${SURFACE_NUM_LAYERS:-3}"
export SAMPLE_NUM="${SAMPLE_NUM:-3}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

export DESC="${DESC:-brep2shape_dropout_${DROPOUT}_epochs_${EPOCHS}_topo${GRAPH_NUM_LAYERS}_geo${SURFACE_NUM_LAYERS}_seed${SEED}_sample${SAMPLE_NUM}}"

bash scripts/pretrain.sh
