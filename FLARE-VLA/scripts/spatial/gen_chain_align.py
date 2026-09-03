#!/usr/bin/env python3
"""Generator: writes the full chain alignment script to the target path."""
import json
from pathlib import Path

OUTPUT = Path("/root/autodl-tmp/FLARE-VLA/scripts/spatial/04_verify_chain_alignment_spatial.py")

# Build the script piece by piece, avoiding escaping issues
lines = []
lines.append("#!/usr/bin/env python3")
lines.append('"""Three-way chain alignment: LeRobot -> LeWM H5 -> GT Latent Framebank."""')
lines.append("")
lines.append("import argparse, json, subprocess, sys, time")
lines.append("from datetime import datetime, timezone")
lines.append("from pathlib import Path")
lines.append("import h5py, numpy as np, pandas as pd, torch")
lines.append("from PIL import Image")
lines.append("")
lines.append("PROJECT_ROOT = Path('/root/autodl-tmp/le-wm')")
lines.append("sys.path.insert(0, str(PROJECT_ROOT))")
lines.append("import hydra, stable_pretraining as spt, stable_worldmodel as swm")
lines.append("from hydra import compose, initialize_config_dir")
lines.append("from omegaconf import open_dict")
lines.append("from utils import get_img_preprocessor")
lines.append("")
lines.append("# Paths & config")
lines.append("DEFAULT_LEROBOT_DIR = Path('/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot')")
lines.append("DEFAULT_TRAIN_H5 = Path('/root/autodl-tmp/le-wm/stablewm_data/libero_spatial_no_noops_1.0.0_lerobot__observation_images_image.h5')")
lines.append("DEFAULT_LATENT_H5 = Path('/root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/spatial_epoch50_h8_framebank_trainpipe.h5')")
lines.append("DEFAULT_FRAME_MAP = Path('/root/autodl-tmp/FLARE-VLA/outputs/spatial/mapping/spatial_lerobot_to_h5_frame_map.parquet')")
lines.append("DEFAULT_CHECKPOINT = Path('/root/autodl-tmp/le-wm/stablewm_data/checkpoints/lewm/weights_epoch_50.pt')")
lines.append("DEFAULT_OUTPUT_DIR = Path('/root/autodl-tmp/FLARE-VLA/outputs/spatial/alignment')")
lines.append("FUTURE_HORIZON, IMAGE_SIZE = 8, 224")
lines.append("NATIVE_H, NATIVE_W = 256, 256")
lines.append("ATOL, LATENT_ATOL, MIN_COSINE = 1e-6, 5e-3, 0.9999")
lines.append("MIN_WINDOWS, MIN_REENCODE, NUM_TASKS, SEED = 200, 20, 10, 42")
lines.append("DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'")

with open(OUTPUT, 'w') as f:
    f.write('\n'.join(lines))
    f.write('\n')

print(f"Wrote {len(lines)} lines to {OUTPUT}")
