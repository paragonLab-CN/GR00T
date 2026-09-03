#!/usr/bin/env bash
cd /root/autodl-tmp/le-wm

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

python train.py data=libero90 \
  trainer.max_epochs=50 \
  loader.batch_size=64 \
  loader.num_workers=0 \
  loader.pin_memory=false \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  wandb.enabled=false \
  2>&1 | tee logs/libero90_bs64_full.log
