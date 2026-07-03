#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_DIR:?Set DATASET_DIR to a directory containing datasplit.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
GPU_NUM=${#CUDA_DEVICES[@]}
if [ -z "${ACCELERATOR:-}" ]; then
  if [ "$GPU_NUM" -gt 1 ]; then
    ACCELERATOR="ddp"
  else
    ACCELERATOR="gpu"
  fi
fi

BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-4}"
LEARNING_RATE="${LEARNING_RATE:-$(python3 -c "print($BASE_LEARNING_RATE * ($BATCH_SIZE / $BASE_BATCH_SIZE) * $GPU_NUM)")}"
SAMPLE_NUM="${SAMPLE_NUM:-3}"
EPOCHS="${EPOCHS:-100}"
DROPOUT="${DROPOUT:-0.0}"
GRAPH_NUM_LAYERS="${GRAPH_NUM_LAYERS:-6}"
SURFACE_NUM_LAYERS="${SURFACE_NUM_LAYERS:-3}"
SEED="${SEED:-0}"
DESC="${DESC:-brep2shape_dropout_${DROPOUT}_epochs_${EPOCHS}_seed_${SEED}}"

python pretrain.py train \
  --dataset_dir "$DATASET_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "${NUM_WORKERS:-8}" \
  --max_epochs "$EPOCHS" \
  --graph_num_layers "$GRAPH_NUM_LAYERS" \
  --edge_num_layers "$SURFACE_NUM_LAYERS" \
  --surface_num_layers "$SURFACE_NUM_LAYERS" \
  --curve_num_heads "${CURVE_NUM_HEADS:-4}" \
  --surface_num_heads "${SURFACE_NUM_HEADS:-4}" \
  --graph_num_heads "${GRAPH_NUM_HEADS:-4}" \
  --curve_hidden_dim "${CURVE_HIDDEN_DIM:-128}" \
  --surface_hidden_dim "${SURFACE_HIDDEN_DIM:-128}" \
  --graph_hidden_dim "${GRAPH_HIDDEN_DIM:-128}" \
  --dim_feedforward "${DIM_FEEDFORWARD:-128}" \
  --mlp_hidden_dim "${MLP_HIDDEN_DIM:-128}" \
  --dropout "$DROPOUT" \
  --act "${ACT:-gelu}" \
  --curve_emb_dim "${CURVE_EMB_DIM:-64}" \
  --surface_emb_dim "${SURFACE_EMB_DIM:-64}" \
  --graph_emb_dim "${GRAPH_EMB_DIM:-128}" \
  --u_samples "$SAMPLE_NUM" \
  --v_samples "$SAMPLE_NUM" \
  --add_positional_encoding \
  --seed "$SEED" \
  --accelerator "$ACCELERATOR" \
  --scheduler "${SCHEDULER:-cosine}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --betas 0.9 0.95 \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --use_class_token \
  --use_layer_norm \
  --norm_first \
  --use_node_bias \
  --use_edge_bias \
  --experiment_name "${EXPERIMENT_NAME:-brep2shape_pretraining_uv_prediction}" \
  --desc "$DESC"
