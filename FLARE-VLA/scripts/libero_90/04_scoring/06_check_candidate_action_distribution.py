#!/usr/bin/env python3
"""Compare GR00T candidate action distribution against LeWM training actions.

The script does not run GR00T or LeWM. It reads saved candidate .pt files,
converts gripper values from LeRobot to LeWM convention, then reports raw and
z-score out-of-distribution rates relative to the LeWM frame_action dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def lerobot_to_lewm(actions: torch.Tensor) -> torch.Tensor:
    out = actions.clone().float()
    g = out[..., 6]
    if torch.any(g < -1e-5) or torch.any(g > 1.0 + 1e-5):
        raise ValueError(
            f"LeRobot gripper outside [0,1]: min={float(g.min())}, max={float(g.max())}"
        )
    out[..., 6] = 1.0 - 2.0 * torch.clamp(g, 0.0, 1.0)
    return out


def quantiles(x: np.ndarray, qs: list[float]) -> np.ndarray:
    return np.quantile(x, qs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--gt-h5", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    files = sorted(args.candidate_dir.glob("sample_*.pt"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No candidate files in {args.candidate_dir}")

    candidate_chunks = []
    gt_chunks = []
    for path in files:
        d = load_torch(path)
        cand_lr = torch.as_tensor(d["candidate_actions_concat"]).float()
        if cand_lr.ndim != 4 or cand_lr.shape[-1] != 7:
            raise RuntimeError(f"Unexpected candidate shape in {path}: {tuple(cand_lr.shape)}")
        candidate_chunks.append(lerobot_to_lewm(cand_lr).reshape(-1, 7))

        gt_lr = torch.as_tensor(d["gt_actions_concat"]).float()
        if gt_lr.ndim != 2 or gt_lr.shape[-1] != 7:
            raise RuntimeError(f"Unexpected GT shape in {path}: {tuple(gt_lr.shape)}")
        gt_chunks.append(lerobot_to_lewm(gt_lr).reshape(-1, 7))

    candidate = torch.cat(candidate_chunks, dim=0).numpy()
    gt_saved = torch.cat(gt_chunks, dim=0).numpy()

    with h5py.File(args.gt_h5, "r") as h5:
        train = np.asarray(h5["frame_action"][:], dtype=np.float32)
    train = train[np.isfinite(train).all(axis=1)]

    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    train_min = train.min(axis=0)
    train_max = train.max(axis=0)
    q001, q01, q99, q999 = quantiles(train, [0.001, 0.01, 0.99, 0.999])

    z = (candidate - mean) / std
    gt_z = (gt_saved - mean) / std

    records = []
    for i, name in enumerate(ACTION_NAMES):
        records.append({
            "dim": i,
            "name": name,
            "train_mean": float(mean[i]),
            "train_std": float(std[i]),
            "train_min": float(train_min[i]),
            "train_q001": float(q001[i]),
            "train_q01": float(q01[i]),
            "train_q99": float(q99[i]),
            "train_q999": float(q999[i]),
            "train_max": float(train_max[i]),
            "candidate_min": float(candidate[:, i].min()),
            "candidate_mean": float(candidate[:, i].mean()),
            "candidate_max": float(candidate[:, i].max()),
            "candidate_outside_train_minmax_rate": float(
                np.mean((candidate[:, i] < train_min[i]) | (candidate[:, i] > train_max[i]))
            ),
            "candidate_outside_q001_q999_rate": float(
                np.mean((candidate[:, i] < q001[i]) | (candidate[:, i] > q999[i]))
            ),
            "candidate_outside_q01_q99_rate": float(
                np.mean((candidate[:, i] < q01[i]) | (candidate[:, i] > q99[i]))
            ),
            "candidate_abs_z_gt3_rate": float(np.mean(np.abs(z[:, i]) > 3)),
            "candidate_abs_z_gt5_rate": float(np.mean(np.abs(z[:, i]) > 5)),
            "candidate_abs_z_gt10_rate": float(np.mean(np.abs(z[:, i]) > 10)),
            "candidate_abs_z_max": float(np.abs(z[:, i]).max()),
            "saved_gt_abs_z_gt3_rate": float(np.mean(np.abs(gt_z[:, i]) > 3)),
            "saved_gt_abs_z_max": float(np.abs(gt_z[:, i]).max()),
        })

    df = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "candidate_action_distribution.csv"
    json_path = args.output_dir / "candidate_action_distribution_summary.json"
    df.to_csv(csv_path, index=False)

    summary = {
        "format": "flare_candidate_action_distribution_v1",
        "candidate_files": len(files),
        "candidate_rows": int(candidate.shape[0]),
        "saved_gt_rows": int(gt_saved.shape[0]),
        "training_rows": int(train.shape[0]),
        "action_names": ACTION_NAMES,
        "any_dimension_outside_train_minmax_rate": float(
            np.mean(np.any((candidate < train_min) | (candidate > train_max), axis=1))
        ),
        "any_dimension_outside_q001_q999_rate": float(
            np.mean(np.any((candidate < q001) | (candidate > q999), axis=1))
        ),
        "any_dimension_abs_z_gt3_rate": float(np.mean(np.any(np.abs(z) > 3, axis=1))),
        "any_dimension_abs_z_gt5_rate": float(np.mean(np.any(np.abs(z) > 5, axis=1))),
        "any_dimension_abs_z_gt10_rate": float(np.mean(np.any(np.abs(z) > 10, axis=1))),
        "per_dimension": records,
        "recommendation_rule": {
            "keep_no_clipping": "all |z|>5 rates approximately zero and min/max exceedance negligible",
            "run_clipping_ablation": "non-negligible |z|>5 or values outside training support",
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== Candidate action distribution ==========")
    print("candidate files:", len(files))
    print("candidate rows :", candidate.shape[0])
    print("training rows  :", train.shape[0])
    print(df.to_string(index=False))
    print("\nAny-dim outside train min/max:", summary["any_dimension_outside_train_minmax_rate"])
    print("Any-dim outside q0.1%-q99.9%:", summary["any_dimension_outside_q001_q999_rate"])
    print("Any-dim |z| > 3:", summary["any_dimension_abs_z_gt3_rate"])
    print("Any-dim |z| > 5:", summary["any_dimension_abs_z_gt5_rate"])
    print("Any-dim |z| > 10:", summary["any_dimension_abs_z_gt10_rate"])
    print("CSV :", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
