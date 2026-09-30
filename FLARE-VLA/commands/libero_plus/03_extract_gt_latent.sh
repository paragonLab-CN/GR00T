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

readonly LEWM_ROOT="/root/autodl-tmp/le-wm"
readonly FLARE_ROOT="/root/autodl-tmp/FLARE-VLA"
readonly LEWM_CHECKPOINT="${LEWM_ROOT}/stablewm_data/checkpoints/libero_plus_lewm_full/weights_epoch_50.pt"
readonly GT_LATENT_OUTPUT="${FLARE_ROOT}/outputs/libero_plus/gt_latents/libero_plus_epoch50_fixed_h8_framebank_trainpipe.h5"

echo "============================================"
echo "  Libero-Plus GT Latent Extraction (epoch-50)"
echo "  Checkpoint: ${LEWM_CHECKPOINT}"
echo "  Output:     ${GT_LATENT_OUTPUT}"
echo "  Horizon: 8"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/03_extract_gt_latent_libero_plus.py \
  --checkpoint-path "${LEWM_CHECKPOINT}" \
  --output-path "${GT_LATENT_OUTPUT}"

echo ""
echo "Done. Output: ${GT_LATENT_OUTPUT}"
