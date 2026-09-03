#!/usr/bin/env bash
# Train LeWM from scratch on LIBERO-Plus data (front camera).
#
# Does NOT modify any vanilla LIBERO configs or outputs.
#
# Output: le-wm/stablewm_data/checkpoints/<run>/weights_epoch_*.pt
#
# Usage: bash FLARE-VLA/commands/libero_plus/02_train_lewm.sh

set -euo pipefail

cd /root/autodl-tmp/le-wm

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

PY=/root/autodl-tmp/envs/LeWorldModel/bin/python

mkdir -p logs

echo "============================================"
echo "  LeWM-Libero-Plus Training"
echo "  Dataset: libero_plus_lerobot__observation_images_front"
echo "  Camera:  front (observation.images.front)"
echo "  Epochs:  50"
echo "  BS:      64"
echo "============================================"

# num_workers>0 parallelizes H5 chunk reads/decompression across processes
# (the 121GB H5 cannot fit in page cache, so single-process reads are
# disk-bound; measured 0.4 it/s with the old chunks, 1.4 it/s with the
# fast-chunk layout, num_workers=0).
# output_model_name doubles as the SaveCkptCallback run_name, which names the
# checkpoint dir: <STABLEWM_HOME>/checkpoints/<run_name>/weights_epoch_N.pt
"${PY}" train.py data=libero_plus \
  subdir=libero_plus_lewm_full \
  output_model_name=libero_plus_lewm_full \
  trainer.max_epochs=50 \
  loader.batch_size=64 \
  loader.num_workers=6 \
  loader.pin_memory=true \
  loader.persistent_workers=true \
  loader.prefetch_factor=3 \
  wandb.enabled=false \
  2>&1 | tee logs/libero_plus_bs64_epoch50.log

echo ""
echo "Checkpoints: /root/autodl-tmp/le-wm/stablewm_data/checkpoints/libero_plus_lewm_full/"
