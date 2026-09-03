#!/usr/bin/env python3
"""Three-way chain alignment: LeRobot -> LeWM H5 -> GT Latent Framebank."""

import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from PIL import Image

PROJECT_ROOT = Path('/root/autodl-tmp/le-wm')
sys.path.insert(0, str(PROJECT_ROOT))
import hydra, stable_pretraining as spt, stable_worldmodel as swm
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from utils import get_img_preprocessor

# Paths & config
DEFAULT_LEROBOT_DIR = Path('/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot')
DEFAULT_TRAIN_H5 = Path('/root/autodl-tmp/le-wm/stablewm_data/libero_spatial_no_noops_1.0.0_lerobot__observation_images_image.h5')
DEFAULT_LATENT_H5 = Path('/root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/spatial_epoch50_h8_framebank_trainpipe.h5')
DEFAULT_FRAME_MAP = Path('/root/autodl-tmp/FLARE-VLA/outputs/spatial/mapping/spatial_lerobot_to_h5_frame_map.parquet')
DEFAULT_CHECKPOINT = Path('/root/autodl-tmp/le-wm/stablewm_data/checkpoints/lewm/weights_epoch_50.pt')
DEFAULT_OUTPUT_DIR = Path('/root/autodl-tmp/FLARE-VLA/outputs/spatial/alignment')
FUTURE_HORIZON, IMAGE_SIZE = 8, 224
NATIVE_H, NATIVE_W = 256, 256
ATOL, LATENT_ATOL, MIN_COSINE = 1e-6, 5e-3, 0.9999
MIN_WINDOWS, MIN_REENCODE, NUM_TASKS, SEED = 200, 20, 10, 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
