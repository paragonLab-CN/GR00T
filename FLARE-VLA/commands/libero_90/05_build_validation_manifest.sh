#!/usr/bin/env bash
cd /root/autodl-tmp/Isaac-GR00T

export OMP_NUM_THREADS=8

uv run --no-sync python \
  scripts/flare/03_build_validation_manifest.py \
  --dataset-root /root/autodl-tmp/lerobot_data/libero_90_lerobot \
  --gt-h5 /root/autodl-tmp/le-wm/outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5 \
  --output-dir outputs/flare_validation_500 \
  --num-episodes 100 \
  --windows-per-episode 5 \
  --history-size 3 \
  --horizon 8 \
  --candidate-k 4 \
  --base-seed 2026
