#!/usr/bin/env bash
cd /root/autodl-tmp/le-wm

mkdir -p outputs/gt_latents

python -u extract_gt_latent_all_v3.py 2>&1 | \
  tee outputs/gt_latents/extract_all_v3.log

python verify_gt_latent_all_v3.py
