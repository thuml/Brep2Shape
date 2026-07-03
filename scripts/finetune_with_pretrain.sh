#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_DIR:?Set DATASET_DIR to a directory containing datasplit_new.json}"
: "${PRETRAIN_CHECKPOINT:?Set PRETRAIN_CHECKPOINT to a pretrained .ckpt file}"

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

METHOD="${METHOD:-dual}"
TASK="${TASK:-segmentation}"
NUM_CLASSES="${NUM_CLASSES:-25}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-2e-4}"
LEARNING_RATE="${LEARNING_RATE:-$(python3 -c "print($BASE_LEARNING_RATE * ($BATCH_SIZE / $BASE_BATCH_SIZE) * $GPU_NUM)")}"
DROPOUT="${DROPOUT:-0.1}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DESC="${DESC:-${METHOD}_pretrained_dropout_${DROPOUT}_lr_${LEARNING_RATE}}"

case "$TASK" in
  classification|segmentation)
    ENTRYPOINT="${TASK}.py"
    ;;
  *)
    echo "Unknown TASK '$TASK'. Use TASK=classification or TASK=segmentation." >&2
    exit 1
    ;;
esac

python "$ENTRYPOINT" train \
  --num_classes "$NUM_CLASSES" \
  --dataset_dir "$DATASET_DIR" \
  --method "$METHOD" \
  --desc "$DESC" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "${NUM_WORKERS:-8}" \
  --max_epochs "$EPOCHS" \
  --graph_num_layers "${GRAPH_NUM_LAYERS:-6}" \
  --edge_num_layers "${EDGE_NUM_LAYERS:-3}" \
  --surface_num_layers "${SURFACE_NUM_LAYERS:-3}" \
  --curve_num_heads "${CURVE_NUM_HEADS:-4}" \
  --surface_num_heads "${SURFACE_NUM_HEADS:-4}" \
  --graph_num_heads "${GRAPH_NUM_HEADS:-4}" \
  --curve_hidden_dim "${CURVE_HIDDEN_DIM:-128}" \
  --surface_hidden_dim "${SURFACE_HIDDEN_DIM:-128}" \
  --graph_hidden_dim "${GRAPH_HIDDEN_DIM:-128}" \
  --dim_feedforward "${DIM_FEEDFORWARD:-128}" \
  --dropout "$DROPOUT" \
  --attention_dropout "$ATTENTION_DROPOUT" \
  --head_dropout "$HEAD_DROPOUT" \
  --act "${ACT:-gelu}" \
  --curve_emb_dim "${CURVE_EMB_DIM:-64}" \
  --surface_emb_dim "${SURFACE_EMB_DIM:-64}" \
  --graph_emb_dim "${GRAPH_EMB_DIM:-128}" \
  --use_edge_bias \
  --use_node_bias \
  --add_positional_encoding \
  --scheduler "${SCHEDULER:-cosine}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --betas 0.9 0.95 \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --seed "${SEED:-0}" \
  --accelerator "$ACCELERATOR" \
  --experiment_name "${EXPERIMENT_NAME:-segmentation}" \
  --use_class_token \
  --use_layer_norm \
  --norm_first \
  --pretrain_checkpoint "$PRETRAIN_CHECKPOINT"
