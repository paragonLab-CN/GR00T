#!/usr/bin/env bash
cd /root/autodl-tmp/Isaac-GR00T

export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export UV_LINK_MODE=copy

uv run --no-sync python \
  scripts/flare/04_generate_validation_candidates.py \
  --manifest outputs/flare_validation_500/validation_manifest.jsonl \
  --dataset-path /root/autodl-tmp/lerobot_data/libero_90_lerobot \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --output-dir outputs/flare_validation_500 \
  --embodiment-tag LIBERO_PANDA \
  --video-backend torchcodec \
  --action-horizon 8 \
  2>&1 | tee outputs/flare_validation_500/candidate_generation_full.log
