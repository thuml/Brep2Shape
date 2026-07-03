#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-brep2shape}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

PRETRAIN_RUN_DIR="${PRETRAIN_RUN_DIR:-$PWD/results/brep2shape_pretraining}"

export PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$PRETRAIN_RUN_DIR/last.ckpt}"
export DATASET_DIR="${DATASET_DIR:-/path/to/classification_data}"
export NUM_CLASSES="${NUM_CLASSES:-10}"

export TASK="classification"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-brep2shape_finetuning_mechcad_classification}"
export METHOD="${METHOD:-dual}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-2e-4}"
export DROPOUT="${DROPOUT:-0.1}"
export ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.1}"
export HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export SEED="${SEED:-0}"

# These values must match the pretrained encoder architecture.
export GRAPH_NUM_LAYERS="${GRAPH_NUM_LAYERS:-6}"
export EDGE_NUM_LAYERS="${EDGE_NUM_LAYERS:-3}"
export SURFACE_NUM_LAYERS="${SURFACE_NUM_LAYERS:-3}"
export CURVE_NUM_HEADS="${CURVE_NUM_HEADS:-4}"
export SURFACE_NUM_HEADS="${SURFACE_NUM_HEADS:-4}"
export GRAPH_NUM_HEADS="${GRAPH_NUM_HEADS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

export DESC="${DESC:-mechcad_classification_topo${GRAPH_NUM_LAYERS}_geo${SURFACE_NUM_LAYERS}_seed${SEED}}"

if [ ! -f "$PRETRAIN_CHECKPOINT" ]; then
  echo "Pretrained checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 1
fi

if [ ! -f "$DATASET_DIR/datasplit.json" ]; then
  echo "Classification split not found: $DATASET_DIR/datasplit.json" >&2
  exit 1
fi

bash scripts/finetune_with_pretrain.sh
