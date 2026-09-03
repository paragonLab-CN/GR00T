#!/usr/bin/env bash
# Extract GT latents for LIBERO-Plus using the LeWM checkpoint trained on
# libero_plus data.
#
# NOTE: Update CHECKPOINT_PATH inside
# scripts/libero_plus/03_extract_gt_latent_libero_plus.py to the actual
# weights_epoch_N.pt produced by 02_train_lewm.sh before running.
#
# Usage: bash FLARE-VLA/commands/libero_plus/03_extract_gt_latent.sh

set -euo pipefail

cd /root/autodl-tmp/le-wm

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "  Libero-Plus GT Latent Extraction"
echo "  Checkpoint: (see script constant)"
echo "  Horizon: 8"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/03_extract_gt_latent_libero_plus.py

echo ""
echo "Done. Output: FLARE-VLA/outputs/libero_plus/gt_latents/"
