#!/usr/bin/env bash
cd /root/autodl-tmp/le-wm

export OLD_LATENT=/root/autodl-tmp/le-wm/outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe.h5
export EP_MAP=/root/autodl-tmp/le-wm/outputs/lerobot_to_train_episode_map_gripper_fixed.jsonl
export NEW_LATENT=/root/autodl-tmp/le-wm/outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5
export LEROBOT90=/root/autodl-tmp/lerobot_data/libero_90_lerobot

python reorder_gt_latent_to_lerobot.py \
  --input-h5 "$OLD_LATENT" \
  --mapping "$EP_MAP" \
  --output-h5 "$NEW_LATENT"

python verify_reordered_gt_latent.py \
  --original-h5 "$OLD_LATENT" \
  --reordered-h5 "$NEW_LATENT" \
  --mapping "$EP_MAP" \
  --episodes-jsonl "$LEROBOT90/meta/episodes.jsonl"
