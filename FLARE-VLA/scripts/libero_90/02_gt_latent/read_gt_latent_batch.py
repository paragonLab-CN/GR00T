#!/usr/bin/env python3
"""
演示后续训练/评分时，如何从 frame-bank H5 读取一个 batch 的 Z_gt。

这不是全量提取脚本，只是读取示例。
"""

from pathlib import Path

import h5py
import numpy as np
import torch


H5_PATH = Path(
    "/root/autodl-tmp/le-wm/outputs/gt_latents/"
    "libero90_epoch50_h8_framebank.h5"
)


def load_gt_batch(window_indices):
    """
    输入：
        window_indices: 长度 B 的窗口编号
    返回：
        z_gt:       [B, 8, 192]
        gt_actions: [B, 8, 7]
        metadata:   episode_id / step_idx / current_row
    """
    window_indices = np.asarray(
        window_indices, dtype=np.int64
    )

    with h5py.File(H5_PATH, "r") as f:
        # h5py 的 fancy indexing 要求索引递增。
        order = np.argsort(window_indices)
        sorted_idx = window_indices[order]
        inverse = np.argsort(order)

        future_rows = np.asarray(
            f["window_future_row"][sorted_idx]
        )[inverse]
        current_rows = np.asarray(
            f["window_current_row"][sorted_idx]
        )[inverse]
        episode_ids = np.asarray(
            f["window_episode_id"][sorted_idx]
        )[inverse]
        step_indices = np.asarray(
            f["window_step_idx"][sorted_idx]
        )[inverse]

        z_list = []
        action_list = []

        horizon = future_rows.shape[1]
        for current, future in zip(
            current_rows, future_rows
        ):
            z_list.append(
                np.asarray(f["frame_latent"][future])
            )
            action_list.append(
                np.asarray(
                    f["frame_action"][
                        current:current + horizon
                    ]
                )
            )

    z_gt = torch.from_numpy(
        np.stack(z_list)
    ).float()
    gt_actions = torch.from_numpy(
        np.stack(action_list)
    ).float()

    metadata = {
        "window_idx": torch.from_numpy(
            window_indices
        ),
        "episode_id": torch.from_numpy(
            episode_ids
        ),
        "step_idx": torch.from_numpy(
            step_indices
        ),
        "current_row": torch.from_numpy(
            current_rows
        ),
    }
    return z_gt, gt_actions, metadata


if __name__ == "__main__":
    z_gt, gt_actions, metadata = load_gt_batch(
        [0, 10, 100]
    )

    print("z_gt       :", tuple(z_gt.shape))
    print("gt_actions :", tuple(gt_actions.shape))
    print("metadata   :", metadata)
