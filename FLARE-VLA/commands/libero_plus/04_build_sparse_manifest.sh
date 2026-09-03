#!/usr/bin/env bash
# Build sparse LIBERO-Plus candidate-window manifest.
#
# Same strategy as spatial: stride-8 + boundary + gripper transition
# densification + action change densification + extra cap at 20 per episode.
#
# NOTE: 03_build_sparse_manifest.py writes "spatial_sparse_*" filenames and
# "spatial_sparse_XXXXXXXXX" sample_ids regardless of dataset. This is
# intentional: downstream candidate generation and scoring only read the
# manifest content (sample_id is opaque), and keeping one shared manifest
# builder avoids forking the dense logic. The libero_plus manifests live in
# their own output dir under outputs/libero_plus/manifests/.
#
# Produces:
#   spatial_sparse_manifest.jsonl / _summary.json / _episode_summary.csv / _task_summary.csv
#
# Usage: bash FLARE-VLA/commands/libero_plus/04_build_sparse_manifest.sh

set -euo pipefail

/root/autodl-tmp/envs/LeWorldModel/bin/python3 -u \
  /root/autodl-tmp/FLARE-VLA/scripts/spatial/03_build_sparse_manifest.py \
  --dataset-root /root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot \
  --gt-h5 /root/autodl-tmp/FLARE-VLA/outputs/libero_plus/gt_latents/libero_plus_epoch9_h8_framebank_trainpipe.h5 \
  --output-dir /root/autodl-tmp/FLARE-VLA/outputs/libero_plus/manifests \
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
echo "  FLARE-VLA/outputs/libero_plus/manifests/spatial_sparse_manifest.jsonl"
