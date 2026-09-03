#!/usr/bin/env bash
cd /root/autodl-tmp/le-wm

export LIBERO90_RAW=/root/autodl-tmp/lerobot-dataset-to-HDF5/datasets/libero_data/libero_90
export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data

python scripts/convert_libero90_to_lewm_h5.py \
  --src-dir "$LIBERO90_RAW" \
  --out "$STABLEWM_HOME/libero90_image_train_v2.h5" \
  --camera agentview \
  --image-size 224
