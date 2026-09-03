#!/usr/bin/env bash
# Extract GT latents for LIBERO-Spatial using the trained LeWM checkpoint.
#
# Spatial-specific — does NOT read or modify any LIBERO-90 outputs.
#
# Usage:  bash FLARE-VLA/commands/spatial/03_extract_gt_latent.sh

set -euo pipefail

cd /root/autodl-tmp/le-wm

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "  Spatial GT Latent Extraction"
echo "  Checkpoint: weights_epoch_50.pt"
echo "  Horizon: 8"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/spatial/03_extract_gt_latent_spatial.py

echo ""
echo "============================================"
echo "  Verification"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python \
  /root/autodl-tmp/FLARE-VLA/scripts/spatial/03_verify_gt_latent_spatial.py
