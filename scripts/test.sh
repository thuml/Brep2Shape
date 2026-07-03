#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_DIR:?Set DATASET_DIR to a directory containing datasplit_new.json}"
: "${CHECKPOINT:?Set CHECKPOINT to a .ckpt file, or a checkpoint directory for segmentation}"

TASK="${TASK:-classification}"
NUM_CLASSES="${NUM_CLASSES:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "$TASK" = "classification" ]; then
  python classification.py test \
    --num_classes "$NUM_CLASSES" \
    --dataset_dir "$DATASET_DIR" \
    --dataset "${DATASET:-fusiongallery}" \
    --method "${METHOD:-dual}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --seed "${SEED:-0}" \
    --checkpoint "$CHECKPOINT" \
    --experiment_name "${EXPERIMENT_NAME:-classification}" \
    --accelerator "${ACCELERATOR:-gpu}"
elif [ "$TASK" = "segmentation" ]; then
  python segmentation.py test \
    --num_classes "$NUM_CLASSES" \
    --dataset_dir "$DATASET_DIR" \
    --dataset "${DATASET:-fusiongallery}" \
    --method "${METHOD:-dual}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --seed "${SEED:-0}" \
    --checkpoint "$CHECKPOINT" \
    --experiment_name "${EXPERIMENT_NAME:-segmentation}" \
    --accelerator "${ACCELERATOR:-gpu}"
else
  echo "Unknown TASK '$TASK'. Use TASK=classification or TASK=segmentation." >&2
  exit 1
fi
