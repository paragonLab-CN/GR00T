#!/usr/bin/env bash
# Train LeWM on LIBERO-Spatial no-noops (front camera)
#
# Spatial-specific experiment. Does NOT modify any LIBERO-90 configs or outputs.
#
# Usage:
#   bash FLARE-VLA/commands/spatial/02_train_lewm_spatial.sh
#
# Output:
#   le-wm/outputs/{timestamp}/  — training logs + checkpoints

set -euo pipefail

cd /root/autodl-tmp/le-wm

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

echo "============================================"
echo "  LeWM-Spatial Training"
echo "  Dataset: libero_spatial_no_noops"
echo "  Camera:  observation.images.image (front)"
echo "  Frames:  52970"
echo "  Epochs:  50"
echo "  BS:      64"
echo "============================================"

python train.py data=libero_spatial \
  trainer.max_epochs=50 \
  loader.batch_size=64 \
  loader.num_workers=0 \
  loader.pin_memory=false \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  wandb.enabled=false \
  2>&1 | tee logs/libero_spatial_bs64_epoch50.log
