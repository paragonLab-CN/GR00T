#!/usr/bin/env python3
"""
验证使用训练数据管线重新生成的全量 latent 文件。
"""

from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path("/root/autodl-tmp/le-wm")

FULL_H5 = (
    PROJECT_ROOT / "outputs/gt_latents/"
    "libero90_epoch50_h8_framebank_trainpipe.h5"
)

SINGLE_PT = (
    PROJECT_ROOT / "outputs/gt_latent_test/"
    "libero90_sample0_h8_zgt.pt"
)

NUM_RANDOM_WINDOWS = 100
SEED = 42

MAX_ABS_TOL = 5e-3
MEAN_ABS_TOL = 1e-3
MIN_COSINE = 0.9999


def load_pt(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    if not FULL_H5.exists():
        raise FileNotFoundError(FULL_H5)
    if not SINGLE_PT.exists():
        raise FileNotFoundError(SINGLE_PT)

    single = load_pt(SINGLE_PT)
    single_z = torch.as_tensor(single["z_gt"]).float()

    with h5py.File(FULL_H5, "r") as f:
        print("========== 新全量文件 ==========")
        print("path                   :", FULL_H5)
        print("format                 :", f.attrs.get("format"))
        print("preprocessing_pipeline :", f.attrs.get("preprocessing_pipeline"))
        print("checkpoint_epoch       :", f.attrs.get("checkpoint_epoch"))
        print("checkpoint_sha256      :", f.attrs.get("checkpoint_sha256"))
        print("complete               :", f.attrs.get("complete"))
        print("frames_written         :", f.attrs.get("frames_written"))
        print("total_frames           :", f.attrs.get("total_frames"))

        if not bool(f.attrs.get("complete", False)):
            raise RuntimeError("新全量文件尚未 complete。")

        required = [
            "frame_latent",
            "frame_action",
            "window_current_row",
            "window_future_row",
            "window_episode_id",
            "window_step_idx",
            "episode_id",
            "ep_offset",
            "ep_len",
        ]
        missing = [key for key in required if key not in f]
        if missing:
            raise KeyError(f"缺少 datasets：{missing}")

        frame_latent = f["frame_latent"]
        future_rows = f["window_future_row"]
        current_rows = f["window_current_row"]
        window_episode = f["window_episode_id"]

        print("\n========== 形状 ==========")
        print("frame_latent      :", frame_latent.shape)
        print("frame_action      :", f["frame_action"].shape)
        print("window_future_row :", future_rows.shape)

        first_rows = np.asarray(
            future_rows[0], dtype=np.int64
        )
        full_first = torch.from_numpy(
            np.asarray(frame_latent[first_rows])
        ).unsqueeze(0).float()

        diff = (full_first - single_z).abs()
        cosine = torch.nn.functional.cosine_similarity(
            full_first.reshape(-1, full_first.shape[-1]),
            single_z.reshape(-1, single_z.shape[-1]),
            dim=-1,
            eps=1e-12,
        )
        relative_l2 = (
            torch.linalg.vector_norm(full_first - single_z)
            / torch.linalg.vector_norm(single_z).clamp_min(1e-12)
        )

        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        min_cosine = float(cosine.min())
        numerical_ok = (
            max_abs <= MAX_ABS_TOL
            and mean_abs <= MEAN_ABS_TOL
            and min_cosine >= MIN_COSINE
        )

        print("\n========== 与单样本数值一致性 ==========")
        print("single z shape :", tuple(single_z.shape))
        print("full z shape   :", tuple(full_first.shape))
        print("max abs error  :", max_abs)
        print("mean abs error :", mean_abs)
        print("relative L2    :", float(relative_l2))
        print("min cosine     :", min_cosine)
        print("mean cosine    :", float(cosine.mean()))
        print("numerical check:", numerical_ok)

        if not numerical_ok:
            raise RuntimeError(
                "新全量第一个窗口与单样本的差异超过允许的 "
                "CUDA/批量浮点误差阈值。"
            )

        episode_ids = np.asarray(
            f["episode_id"][:], dtype=np.int64
        )
        ep_offsets = np.asarray(
            f["ep_offset"][:], dtype=np.int64
        )
        ep_lengths = np.asarray(
            f["ep_len"][:], dtype=np.int64
        )
        episode_map = {
            int(ep): (int(offset), int(length))
            for ep, offset, length in zip(
                episode_ids, ep_offsets, ep_lengths
            )
        }

        rng = np.random.default_rng(SEED)
        count = min(NUM_RANDOM_WINDOWS, len(future_rows))
        indices = rng.choice(
            len(future_rows),
            size=count,
            replace=False,
        )

        print("\n========== 随机窗口检查 ==========")
        for idx in indices:
            current = int(current_rows[idx])
            future = np.asarray(
                future_rows[idx], dtype=np.int64
            )
            episode = int(window_episode[idx])

            expected = current + np.arange(
                1, len(future) + 1, dtype=np.int64
            )
            if not np.array_equal(future, expected):
                raise RuntimeError(
                    f"窗口 {idx} future rows 不连续。"
                )

            offset, length = episode_map[episode]
            if int(future[-1]) >= offset + length:
                raise RuntimeError(
                    f"窗口 {idx} 跨 episode。"
                )

            z = np.asarray(frame_latent[future])
            if not np.isfinite(z).all():
                raise RuntimeError(
                    f"窗口 {idx} latent 有 NaN/Inf。"
                )

        print(f"随机检查 {count} 个窗口：通过。")

        current0 = int(current_rows[0])
        gt_actions0 = np.asarray(
            f["frame_action"][current0:current0 + 8]
        )
        print("\n========== 第0窗口 ==========")
        print("future rows      :", first_rows.tolist())
        print("z_gt shape      :", tuple(full_first.shape))
        print("gt_actions shape:", gt_actions0.shape)

    print("\n所有新全量验证均已通过。")


if __name__ == "__main__":
    main()
