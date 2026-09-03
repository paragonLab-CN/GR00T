#!/usr/bin/env bash
# Build sparse LIBERO-Spatial candidate-window manifest.
#
# Strategy: stride-8 + boundary + gripper transition densification
# + action change densification + extra cap at 20 per episode.
#
# Produces:
#   spatial_sparse_manifest.jsonl              sparse window index
#   spatial_sparse_manifest_summary.json        summary + coverage stats
#   spatial_sparse_manifest_episode_summary.csv per-episode breakdown
#   spatial_sparse_manifest_task_summary.csv    per-task breakdown
#
# Spatial-specific. Does NOT read or modify any LIBERO-90 outputs.
#
# Usage:
#   bash FLARE-VLA/commands/spatial/03_build_sparse_manifest.sh

set -euo pipefail

/root/autodl-tmp/envs/LeWorldModel/bin/python3 -u \
  /root/autodl-tmp/FLARE-VLA/scripts/spatial/03_build_sparse_manifest.py \
  --dataset-root /root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot \
  --gt-h5 /root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/spatial_epoch50_h8_framebank_trainpipe.h5 \
  --output-dir /root/autodl-tmp/FLARE-VLA/outputs/spatial/manifests \
  --history-size 3 \
  --horizon 8 \
  --stride 8 \
  --candidate-k 4 \
  --base-seed 2026 \
  --gripper-radius 2 \
  --action-change-topk 3 \
  --action-change-radius 1 \
  --max-extra-windows-per-episode 20

echo ""
echo "Done. Manifest at:"
echo "  FLARE-VLA/outputs/spatial/manifests/spatial_sparse_manifest.jsonl"
