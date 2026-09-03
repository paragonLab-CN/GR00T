#!/usr/bin/env python3
import os

# Must be set before importing torch/OpenMP users.
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"

from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path("/root/autodl-tmp/le-wm")
CKPT_PATH = ROOT / "stablewm_data/checkpoints/lewm/weights_epoch_50.pt"
EXPECTED_SHA256 = "e26368be25b6c8f391f78e798bc06e14627697a69d519ac9063c0e41946f9267"


def _find_state_dict(obj: Any):
    if not isinstance(obj, dict):
        return None, None

    for key in ("model_state_dict", "state_dict", "model"):
        value = obj.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return key, value

    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return "<top-level>", obj

    # One extra nested level, for wrappers such as {'module': {'state_dict': ...}}.
    for outer_key, outer_value in obj.items():
        if isinstance(outer_value, dict):
            for inner_key in ("model_state_dict", "state_dict", "model"):
                value = outer_value.get(inner_key)
                if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                    return f"{outer_key}.{inner_key}", value
    return None, None


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def main(cfg: DictConfig) -> None:
    print("========== Top-level training config ==========")
    for key in ("history_size", "num_preds", "img_size", "seed"):
        print(f"{key:14s}: {OmegaConf.select(cfg, key)}")

    print("\n========== Dataset config ==========")
    dataset_cfg = cfg.data.dataset
    for key in ("name", "num_steps", "frameskip", "keys_to_load", "keys_to_cache"):
        value = OmegaConf.select(dataset_cfg, key)
        print(f"{key:14s}: {value}")

    print("\n========== Model config ==========")
    print(OmegaConf.to_yaml(cfg.model, resolve=True))

    print("\n========== Checkpoint ==========")
    print("checkpoint path :", CKPT_PATH)
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(CKPT_PATH)

    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")

    if isinstance(ckpt, dict):
        print("top-level keys  :", list(ckpt.keys()))
        for key in ("format", "epoch", "run_name", "checkpoint_sha256"):
            if key in ckpt:
                print(f"{key:16s}: {ckpt[key]}")
    else:
        print("checkpoint type :", type(ckpt).__name__)

    state_key, state_dict = _find_state_dict(ckpt)
    print("state dict key  :", state_key)
    if state_dict is None:
        raise RuntimeError("Could not locate a tensor state_dict in the checkpoint")

    print("state tensors   :", len(state_dict))
    print("\n========== Important parameter shapes ==========")
    matched = 0
    for key, value in state_dict.items():
        if any(name in key for name in ("action_encoder", "projector", "pred_proj", "predictor")):
            print(f"{key}: {tuple(value.shape)}")
            matched += 1
    if matched == 0:
        print("[WARN] No matching parameter names found; print the first 30 keys below:")
        for key in list(state_dict.keys())[:30]:
            value = state_dict[key]
            print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
