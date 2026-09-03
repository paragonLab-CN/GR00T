#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Isaac-GR00T

export OMP_NUM_THREADS=8

uv run --no-sync python \
  scripts/flare/03_build_full_manifest.py \
  --dataset-root /root/autodl-tmp/lerobot_data/libero_90_lerobot \
  --gt-h5 /root/autodl-tmp/le-wm/outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5 \
  --episode-selection-manifest outputs/flare_validation_500/validation_manifest.jsonl \
  --output-dir outputs/flare_selected100_all_windows \
  --history-size 3 \
  --horizon 8 \
  --candidate-k 4 \
  --base-seed 2026 \
  --all-legal-windows
