#!/usr/bin/env python3
"""
诊断：
为什么单样本 z_gt 与全量 frame-bank 的第一个窗口不一致。

不会修改或删除任何文件。

运行：
    conda activate LeWorldModel
    cd /root/autodl-tmp/le-wm
    python diagnose_gt_latent_mismatch.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/le-wm")
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from utils import get_img_preprocessor


DATA_ROOT = PROJECT_ROOT / "stablewm_data"
DATASET_NAME = "libero90_image_train_v2"
SOURCE_H5 = DATA_ROOT / f"{DATASET_NAME}.h5"

CHECKPOINT_PATH = (
    DATA_ROOT / "checkpoints/lewm/weights_epoch_50.pt"
)

SINGLE_PT = (
    PROJECT_ROOT / "outputs/gt_latent_test/"
    "libero90_sample0_h8_zgt.pt"
)

FULL_H5 = (
    PROJECT_ROOT / "outputs/gt_latents/"
    "libero90_epoch50_h8_framebank.h5"
)

FUTURE_HORIZON = 8
FRAMESKIP = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pt(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def version_of(module) -> str:
    return str(getattr(module, "__version__", "unknown"))


def compare(name: str, a, b, atol=1e-6, rtol=1e-5) -> bool:
    a = torch.as_tensor(a).detach().cpu().float()
    b = torch.as_tensor(b).detach().cpu().float()

    print(f"\n--- {name} ---")
    print("shape A          :", tuple(a.shape))
    print("shape B          :", tuple(b.shape))

    if a.shape != b.shape:
        print("result           : FAIL (shape 不一致)")
        return False

    diff = (a - b).abs()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)

    print("max abs error    :", float(diff.max()))
    print("mean abs error   :", float(diff.mean()))
    print("allclose         :", ok)
    return bool(ok)


def load_config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "config/train"),
    ):
        cfg = compose(
            config_name="lewm",
            overrides=["data=libero90"],
        )
    return cfg


def make_transform(cfg):
    image_transform = get_img_preprocessor(
        source="pixels",
        target="pixels",
        img_size=int(cfg.img_size),
    )
    return spt.data.transforms.Compose(image_transform)


def build_model(cfg, action_dim: int):
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            FRAMESKIP * int(action_dim)
        )

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = load_pt(CHECKPOINT_PATH)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model = model.to(DEVICE)
    model.eval()
    model.requires_grad_(False)

    return model, checkpoint


def encode(model, pixels: torch.Tensor) -> torch.Tensor:
    pixels = torch.as_tensor(pixels).float()

    if pixels.ndim != 4:
        raise RuntimeError(
            f"encode 输入应为 [T,C,H,W]，实际 {tuple(pixels.shape)}"
        )

    with torch.inference_mode():
        result = model.encode(
            {"pixels": pixels.unsqueeze(0).to(DEVICE)}
        )["emb"]

    return result.detach().cpu()


def find_action_matches(
    full_file: h5py.File,
    target_actions: np.ndarray,
):
    """
    在所有合法窗口中，查找与单样本保存的 8 步真实动作一致的窗口。
    """
    current_rows_ds = full_file["window_current_row"]
    frame_action_ds = full_file["frame_action"]

    target_actions = np.asarray(target_actions)
    matches = []
    chunk_size = 65536

    for begin in range(0, len(current_rows_ds), chunk_size):
        end = min(begin + chunk_size, len(current_rows_ds))
        rows = np.asarray(current_rows_ds[begin:end], dtype=np.int64)

        first_actions = np.asarray(frame_action_ds[rows])
        first_mask = np.all(
            np.isclose(
                first_actions,
                target_actions[0][None, :],
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            ),
            axis=1,
        )

        candidate_rows = rows[first_mask]

        for row in candidate_rows:
            row = int(row)
            if row + len(target_actions) > len(frame_action_ds):
                continue

            actions = np.asarray(
                frame_action_ds[
                    row:row + len(target_actions)
                ]
            )
            if np.allclose(
                actions,
                target_actions,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            ):
                matches.append(row)

    return matches


def main():
    required = [
        SOURCE_H5,
        CHECKPOINT_PATH,
        SINGLE_PT,
        FULL_H5,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少文件：\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    print("========== 环境和文件 ==========")
    print("torch version                :", torch.__version__)
    print("stable_worldmodel version    :", version_of(swm))
    print("stable_pretraining version   :", version_of(spt))
    print("device                       :", DEVICE)
    print("checkpoint sha256            :", sha256(CHECKPOINT_PATH))
    print("single PT modified time      :", SINGLE_PT.stat().st_mtime)
    print("full H5 modified time        :", FULL_H5.stat().st_mtime)

    cfg = load_config()
    single = load_pt(SINGLE_PT)

    # ------------------------------------------------------------
    # 阶段1：短暂打开源 H5，只读取模型需要的 action_dim，然后关闭。
    # 注意：不能让它与 HDF5Dataset 对同一源文件同时保持打开，
    # 因为 HDF5Dataset 内部会使用 swmr=True。
    # ------------------------------------------------------------
    with h5py.File(SOURCE_H5, "r") as source:
        action_dim = int(source["action"].shape[-1])

    model, checkpoint = build_model(cfg, action_dim)

    # ------------------------------------------------------------
    # 阶段2：读取全量文件元数据和动作匹配。FULL_H5 是另一个文件，
    # 不会与 SOURCE_H5 的 SWMR 标志冲突。
    # ------------------------------------------------------------
    with h5py.File(FULL_H5, "r") as full:
        print("\n========== 元数据 ==========")
        print("current checkpoint epoch     :", checkpoint.get("epoch"))
        print("single checkpoint epoch      :", single.get("checkpoint_epoch"))
        print("single sample_index          :", single.get("sample_index"))
        print("full checkpoint epoch        :", full.attrs.get("checkpoint_epoch"))
        print("full frames_written          :", full.attrs.get("frames_written"))
        print("full complete                :", full.attrs.get("complete"))

        target_actions = (
            single["gt_actions"].detach().cpu().numpy()
        )
        matches = find_action_matches(full, target_actions)

        print("\n========== 动作序列定位 ==========")
        print("匹配窗口 current_row 数量     :", len(matches))
        print("前20个匹配 current_row        :", matches[:20])

        if 0 in matches:
            print("结论                         : 单样本动作与原始 row 0:8 一致")
        elif matches:
            print(
                "结论                         : 单样本并不对应 row 0，"
                f"首个匹配 current_row={matches[0]}"
            )
        else:
            print(
                "结论                         : 全量 frame_action 中找不到"
                "单样本保存的完整 8 步动作序列"
            )

        first_future_rows = np.asarray(
            full["window_future_row"][0],
            dtype=np.int64,
        )
        z_full_first = torch.from_numpy(
            np.asarray(
                full["frame_latent"][first_future_rows]
            )
        ).unsqueeze(0).float()

    # ------------------------------------------------------------
    # 阶段3：完全复现单样本 HDF5Dataset 路线。
    # 此时 SOURCE_H5 已经关闭，因此 HDF5Dataset 可以用 swmr=True 打开。
    # ------------------------------------------------------------
    dataset = swm.data.HDF5Dataset(
        name=DATASET_NAME,
        num_steps=FUTURE_HORIZON + 1,
        frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=None,
        cache_dir=str(DATA_ROOT),
    )
    dataset.transform = make_transform(cfg)

    sample = dataset[0]
    ds_pixels = torch.as_tensor(sample["pixels"]).float()
    ds_actions = torch.as_tensor(sample["action"]).float()

    print("\n========== HDF5Dataset 第0样本 ==========")
    print("dataset length               :", len(dataset))
    print("pixels shape                 :", tuple(ds_pixels.shape))
    print("actions shape                :", tuple(ds_actions.shape))

    clip_indices = getattr(dataset, "clip_indices", None)
    if clip_indices is not None and len(clip_indices) > 0:
        print("dataset.clip_indices[0]      :", clip_indices[0])
    else:
        print("dataset.clip_indices[0]      : unavailable")

    # 主动关闭 HDF5Dataset 内部以 swmr=True 打开的源文件。
    # 这样后面才能再以普通 h5py.File(..., "r") 方式打开同一文件。
    dataset_h5 = getattr(dataset, "h5_file", None)
    if dataset_h5 is not None:
        dataset_h5.close()
        dataset.h5_file = None

    # ------------------------------------------------------------
    # 阶段4：直接读取原始 H5 row 0:9。
    # ------------------------------------------------------------
    with h5py.File(SOURCE_H5, "r") as source:
        raw_pixels = np.asarray(
            source["pixels"][:FUTURE_HORIZON + 1]
        )
        raw_actions = np.asarray(
            source["action"][:FUTURE_HORIZON]
        )

    seq_result = make_transform(cfg)(
        {"pixels": raw_pixels.copy()}
    )
    raw_sequence_pixels = torch.as_tensor(
        seq_result["pixels"]
    ).float()

    individual_transform = make_transform(cfg)
    individual_pixels = []
    for image in raw_pixels:
        output = individual_transform(
            {"pixels": image.copy()}
        )
        individual_pixels.append(
            torch.as_tensor(output["pixels"]).float()
        )
    raw_individual_pixels = torch.stack(
        individual_pixels, dim=0
    )

    print("\n========== 原始数据形状 ==========")
    print("raw pixels                   :", raw_pixels.shape)
    print("sequence transform           :", tuple(raw_sequence_pixels.shape))
    print("individual transform         :", tuple(raw_individual_pixels.shape))

    actions_ds_vs_raw = compare(
        "HDF5Dataset actions[:8] vs 原始 H5 row 0:8",
        ds_actions[:FUTURE_HORIZON],
        raw_actions,
    )

    pixels_ds_vs_seq = compare(
        "HDF5Dataset pixels vs 原始9帧整体预处理",
        ds_pixels,
        raw_sequence_pixels,
    )

    pixels_ds_vs_ind = compare(
        "HDF5Dataset pixels vs 原始9帧逐帧预处理",
        ds_pixels,
        raw_individual_pixels,
    )

    pixels_seq_vs_ind = compare(
        "原始9帧整体预处理 vs 逐帧预处理",
        raw_sequence_pixels,
        raw_individual_pixels,
    )

    # ------------------------------------------------------------
    # 阶段5：用当前 checkpoint 重新编码。
    # ------------------------------------------------------------
    z_saved_single = torch.as_tensor(
        single["z_gt"]
    ).float()

    z_fresh_dataset = encode(
        model,
        ds_pixels[1:FUTURE_HORIZON + 1],
    )
    z_fresh_sequence = encode(
        model,
        raw_sequence_pixels[1:FUTURE_HORIZON + 1],
    )
    z_fresh_individual = encode(
        model,
        raw_individual_pixels[1:FUTURE_HORIZON + 1],
    )

    saved_vs_fresh = compare(
        "已保存 single z_gt vs 当前checkpoint重新编码dataset[0]",
        z_saved_single,
        z_fresh_dataset,
    )
    fresh_ds_vs_seq = compare(
        "当前checkpoint：dataset路线 vs 整体预处理路线",
        z_fresh_dataset,
        z_fresh_sequence,
    )
    fresh_ds_vs_ind = compare(
        "当前checkpoint：dataset路线 vs 逐帧预处理路线",
        z_fresh_dataset,
        z_fresh_individual,
    )
    full_vs_ind = compare(
        "全量第一个窗口 vs 当前checkpoint逐帧重新编码",
        z_full_first,
        z_fresh_individual,
    )
    full_vs_fresh_ds = compare(
        "全量第一个窗口 vs 当前checkpoint dataset[0]重新编码",
        z_full_first,
        z_fresh_dataset,
    )

    print("\n========== 自动诊断 ==========")

    if not actions_ds_vs_raw:
        print(
            "[A] HDF5Dataset[0] 并不对应原始 H5 的 row 0 开始。"
        )
        print(
            "    应修正验证脚本的窗口映射，而不是重跑全量 latent。"
        )
    elif not pixels_ds_vs_ind:
        print(
            "[B] HDF5Dataset 序列预处理与全量脚本逐帧预处理不同。"
        )
        print(
            "    全量提取必须改成与训练/单样本完全相同的预处理路线后重跑。"
        )
    elif not saved_vs_fresh:
        print(
            "[C] 单样本 PT 与当前 checkpoint 重新计算结果不一致。"
        )
        print(
            "    单样本文件可能过期，或 checkpoint 在单样本生成后被覆盖。"
        )
        print(
            "    先重新运行 extract_gt_latent_one.py，再执行验证。"
        )
    elif not full_vs_ind:
        print(
            "[D] 全量 H5 与当前 checkpoint 逐帧重新计算结果不一致。"
        )
        print(
            "    全量文件可能来自旧 checkpoint，或断点续跑时复用了旧数据。"
        )
        print(
            "    需要换新 OUTPUT_PATH 或设置 OVERWRITE=True 后重跑全量。"
        )
    elif not full_vs_fresh_ds:
        print(
            "[E] 全量逐帧结果正确，但与 dataset 序列路线仍有差异。"
        )
        print(
            "    重点检查预处理的序列维度行为。"
        )
    else:
        print(
            "[OK] 当前重新计算的所有关键结果一致。"
        )
        print(
            "     原验证失败更可能来自旧的单样本文件，"
            "重新生成 single PT 后即可。"
        )

    print("\n========== 关键布尔结果 ==========")
    print("actions dataset == raw row0 :", actions_ds_vs_raw)
    print("pixels dataset == sequence  :", pixels_ds_vs_seq)
    print("pixels dataset == individual:", pixels_ds_vs_ind)
    print("sequence == individual      :", pixels_seq_vs_ind)
    print("saved single == fresh ds    :", saved_vs_fresh)
    print("fresh ds == fresh sequence  :", fresh_ds_vs_seq)
    print("fresh ds == fresh individual:", fresh_ds_vs_ind)
    print("full == fresh individual    :", full_vs_ind)
    print("full == fresh dataset       :", full_vs_fresh_ds)


if __name__ == "__main__":
    main()
