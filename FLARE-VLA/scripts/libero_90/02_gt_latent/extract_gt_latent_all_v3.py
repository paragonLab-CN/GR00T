#!/usr/bin/env python3
"""
使用与训练阶段完全相同的 HDF5Dataset + transform 数据管线，
重新提取 LIBERO-90 全量逐帧真值 latent。

重要：
- 不覆盖旧的错误 latent 文件；
- 从旧文件复制已经验证正确的 episode/window/action 元数据；
- 只重新计算 frame_latent；
- 正式全量计算前，先进行：
  1) num_steps=1 与训练式 num_steps=9 像素一致性测试；
  2) 与已通过的单样本 z_gt 一致性测试；
- 支持中断后继续。

运行：
    conda activate LeWorldModel
    cd /root/autodl-tmp/le-wm
    python -u extract_gt_latent_all_v3.py 2>&1 | \
      tee outputs/gt_latents/extract_all_v3.log
"""

from __future__ import annotations

import hashlib
import sys
import time
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
from torch.utils.data import DataLoader, Dataset

from utils import get_img_preprocessor


DATA_ROOT = PROJECT_ROOT / "stablewm_data"
DATASET_NAME = "libero90_image_train_v2"
SOURCE_H5 = DATA_ROOT / f"{DATASET_NAME}.h5"

CHECKPOINT_PATH = (
    DATA_ROOT / "checkpoints/lewm/weights_epoch_50.pt"
)

# 已验证 episode/window/action 正确，但 frame_latent 预处理错误的旧文件。
OLD_METADATA_H5 = (
    PROJECT_ROOT / "outputs/gt_latents/"
    "libero90_epoch50_h8_framebank.h5"
)

# 使用训练数据管线重新生成的新文件。
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs/gt_latents/"
    "libero90_epoch50_h8_framebank_trainpipe.h5"
)

SINGLE_PT = (
    PROJECT_ROOT / "outputs/gt_latent_test/"
    "libero90_sample0_h8_zgt.pt"
)

FUTURE_HORIZON = 8
FRAMESKIP = 1
BATCH_SIZE = 128
NUM_WORKERS = 0
PRINT_EVERY = 20

# CUDA/TF32/不同批量工作负载下，数学上相同的 FP32 前向不保证逐位一致。
# 这些阈值远小于此前错误预处理造成的差异：
#   错误旧管线 mean≈1.18, max≈4.34
# 当前正确管线跨进程差异：
#   mean≈3.0e-4, max≈9.7e-4
MAX_ABS_TOL = 5e-3
MEAN_ABS_TOL = 1e-3
MIN_COSINE = 0.9999

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DTYPE = np.float32

# False：文件存在时按 items_written 断点续跑。
# True：删除新输出文件并从头开始。不会删除 OLD_METADATA_H5。
OVERWRITE = False


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


def check_paths():
    required = [
        PROJECT_ROOT / "config/train/lewm.yaml",
        PROJECT_ROOT / "config/train/data/libero90.yaml",
        SOURCE_H5,
        CHECKPOINT_PATH,
        OLD_METADATA_H5,
        SINGLE_PT,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少以下文件：\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def load_config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "config/train"),
    ):
        return compose(
            config_name="lewm",
            overrides=["data=libero90"],
        )


def make_transform(cfg):
    image_transform = get_img_preprocessor(
        source="pixels",
        target="pixels",
        img_size=int(cfg.img_size),
    )
    return spt.data.transforms.Compose(image_transform)


def build_swm_dataset(cfg, num_steps: int):
    """
    必须通过 stable_worldmodel.HDF5Dataset 读取，再应用与 train.py
    相同的图像 transform。不要直接用 h5py 读取 pixels 后手工变换。
    """
    dataset = swm.data.HDF5Dataset(
        name=DATASET_NAME,
        num_steps=int(num_steps),
        frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=None,
        cache_dir=str(DATA_ROOT),
    )
    dataset.transform = make_transform(cfg)
    return dataset


def close_swm_dataset(dataset):
    h5_file = getattr(dataset, "h5_file", None)
    if h5_file is not None:
        try:
            h5_file.close()
        finally:
            dataset.h5_file = None


def build_model(cfg, action_dim: int):
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            FRAMESKIP * int(action_dim)
        )

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = load_pt(CHECKPOINT_PATH)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "checkpoint 中不存在 model_state_dict。"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model = model.to(DEVICE)
    model.eval()
    model.requires_grad_(False)

    return model, checkpoint


def encode_pixels(model, pixels):
    """
    pixels:
      [T,C,H,W] 或 [B,T,C,H,W]
    返回：
      [1,T,D] 或 [B,T,D]
    """
    pixels = torch.as_tensor(pixels).float()
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 5:
        raise RuntimeError(
            f"pixels 应为 [B,T,C,H,W]，实际 {tuple(pixels.shape)}"
        )

    with torch.inference_mode():
        z = model.encode(
            {"pixels": pixels.to(DEVICE, non_blocking=True)}
        )["emb"]
    return z.detach().cpu()


def build_row_map(frame_dataset, ep_offsets, total_frames):
    clips = np.asarray(
        frame_dataset.clip_indices,
        dtype=np.int64,
    )

    if clips.ndim != 2 or clips.shape[1] != 2:
        raise RuntimeError(
            f"clip_indices 形状异常：{clips.shape}"
        )

    episode_positions = clips[:, 0]
    starts = clips[:, 1]

    if episode_positions.min() < 0:
        raise RuntimeError("clip_indices 存在负 episode 下标。")
    if episode_positions.max() >= len(ep_offsets):
        raise RuntimeError("clip_indices episode 下标越界。")

    row_map = ep_offsets[episode_positions] + starts

    expected = np.arange(total_frames, dtype=np.int64)
    if len(row_map) != total_frames:
        raise RuntimeError(
            f"num_steps=1 数据集长度 {len(row_map)} "
            f"不等于原始总帧数 {total_frames}。"
        )

    # 本项目数据必须逐 episode 连续排布。这个检查确保 dataset index
    # 与原始 H5 row 完全一一对应，避免 latent 写错行。
    if not np.array_equal(row_map, expected):
        first_bad = int(np.flatnonzero(row_map != expected)[0])
        raise RuntimeError(
            "num_steps=1 的 dataset index 与原始 H5 row 不一致："
            f"首个错误位置 index={first_bad}, "
            f"row_map={row_map[first_bad]}, expected={first_bad}"
        )

    return row_map


class FrameDatasetWithRow(Dataset):
    def __init__(
        self,
        base_dataset,
        row_map,
        start_item: int,
    ):
        self.base_dataset = base_dataset
        self.row_map = row_map
        self.start_item = int(start_item)

    def __len__(self):
        return len(self.base_dataset) - self.start_item

    def __getitem__(self, local_idx):
        item_idx = self.start_item + int(local_idx)
        sample = self.base_dataset[item_idx]

        pixels = torch.as_tensor(sample["pixels"]).float()
        # num_steps=1，必须为 [1,C,H,W]。
        if pixels.ndim != 4 or pixels.shape[0] != 1:
            raise RuntimeError(
                f"item {item_idx} pixels 应为 [1,C,H,W]，"
                f"实际 {tuple(pixels.shape)}"
            )

        return pixels, int(self.row_map[item_idx]), item_idx


def latent_metrics(reference, candidate):
    reference = torch.as_tensor(reference).float()
    candidate = torch.as_tensor(candidate).float()

    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"latent shape 不一致：{tuple(reference.shape)} "
            f"vs {tuple(candidate.shape)}"
        )

    diff = (reference - candidate).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    ref_flat = reference.reshape(-1, reference.shape[-1])
    cand_flat = candidate.reshape(-1, candidate.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(
        ref_flat,
        cand_flat,
        dim=-1,
        eps=1e-12,
    )
    min_cosine = float(cosine.min())
    mean_cosine = float(cosine.mean())

    rel_l2 = float(
        torch.linalg.vector_norm(reference - candidate)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    )

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "min_cosine": min_cosine,
        "mean_cosine": mean_cosine,
        "relative_l2": rel_l2,
    }


def print_latent_metrics(title, reference, candidate):
    metrics = latent_metrics(reference, candidate)

    print(f"\n========== {title} ==========")
    print("reference shape :", tuple(reference.shape))
    print("candidate shape :", tuple(candidate.shape))
    print("max abs error   :", metrics["max_abs"])
    print("mean abs error  :", metrics["mean_abs"])
    print("relative L2     :", metrics["relative_l2"])
    print("min cosine      :", metrics["min_cosine"])
    print("mean cosine     :", metrics["mean_cosine"])

    passed = (
        metrics["max_abs"] <= MAX_ABS_TOL
        and metrics["mean_abs"] <= MEAN_ABS_TOL
        and metrics["min_cosine"] >= MIN_COSINE
    )
    print("numerical check :", passed)
    return passed, metrics


def preflight_check(cfg, model, frame_dataset):
    """
    全量运行前验证：
    A. num_steps=1 逐帧读取后拼接，与训练式 num_steps=9 的像素完全一致；
    B. 当前进程重算结果与之前 single z_gt 在严格数值阈值内一致；
    C. 用正式 BATCH_SIZE 模拟第一批前向，确认批量工作负载只带来微小浮点差异。
    """
    sequence_dataset = build_swm_dataset(
        cfg,
        num_steps=FUTURE_HORIZON + 1,
    )

    try:
        reference = sequence_dataset[0]
        reference_pixels = torch.as_tensor(
            reference["pixels"]
        ).float()

        frame_list = []
        for idx in range(FUTURE_HORIZON + 1):
            one = frame_dataset[idx]
            pixels = torch.as_tensor(one["pixels"]).float()
            frame_list.append(pixels[0])
        stacked_pixels = torch.stack(frame_list, dim=0)

        pixel_diff = (
            reference_pixels - stacked_pixels
        ).abs()

        print("\n========== 正式提取前像素管线检查 ==========")
        print("reference pixels :", tuple(reference_pixels.shape))
        print("stacked pixels   :", tuple(stacked_pixels.shape))
        print("max abs error    :", float(pixel_diff.max()))
        print("mean abs error   :", float(pixel_diff.mean()))

        # 像素数据管线必须完全相同，不在这里放宽。
        if not torch.equal(reference_pixels, stacked_pixels):
            raise RuntimeError(
                "num_steps=1 逐帧 HDF5Dataset 输出与 "
                "num_steps=9 训练式输出不完全一致，停止全量提取。"
            )

        future_pixels = stacked_pixels[
            1:FUTURE_HORIZON + 1
        ]

        # 路线1：与单样本脚本相同的 [1,8,C,H,W] 前向。
        fresh_z = encode_pixels(
            model,
            future_pixels,
        ).float()

        single = load_pt(SINGLE_PT)
        single_z = torch.as_tensor(
            single["z_gt"]
        ).float()

        single_ok, _ = print_latent_metrics(
            "当前进程结果 vs 已保存单样本",
            single_z,
            fresh_z,
        )

        if not single_ok:
            raise RuntimeError(
                "像素已经完全一致，但当前重算 latent 与 single z_gt "
                "超过允许的 CUDA 浮点误差阈值，停止全量提取。"
            )

        # 路线2：模拟正式全量脚本的第一批 [B,1,C,H,W]。
        actual_batch_size = min(BATCH_SIZE, len(frame_dataset))
        batch_frames = []
        for idx in range(actual_batch_size):
            one = frame_dataset[idx]
            pixels = torch.as_tensor(one["pixels"]).float()
            batch_frames.append(pixels[0])
        batch_pixels = torch.stack(batch_frames, dim=0)

        with torch.inference_mode():
            batch_z = model.encode(
                {
                    "pixels": batch_pixels.unsqueeze(1).to(
                        DEVICE,
                        non_blocking=True,
                    )
                }
            )["emb"][:, 0].detach().cpu().float()

        # 第一批中的 row 1..8 对应第一个窗口的未来帧。
        batch_first_window_z = batch_z[
            1:FUTURE_HORIZON + 1
        ].unsqueeze(0)

        batch_ok, _ = print_latent_metrics(
            f"序列前向 vs 正式 batch_size={actual_batch_size} 前向",
            fresh_z,
            batch_first_window_z,
        )

        if not batch_ok:
            raise RuntimeError(
                "正式 batch 工作负载导致的 latent 差异超过阈值，"
                "停止全量提取。请降低 BATCH_SIZE 或启用更高精度设置。"
            )

        # 同一进程、同一输入、同一工作负载重复一次。
        repeated_z = encode_pixels(
            model,
            future_pixels,
        ).float()
        repeat_ok, _ = print_latent_metrics(
            "同一进程重复前向",
            fresh_z,
            repeated_z,
        )

        if not repeat_ok:
            raise RuntimeError(
                "同一进程、同一输入的重复前向不稳定，停止全量提取。"
            )

        print(
            "\n预检查全部通过：像素完全一致，latent 差异处于"
            "正常 CUDA/批量浮点误差范围内，可以开始全量提取。"
        )
    finally:
        close_swm_dataset(sequence_dataset)


def create_output(checkpoint, checkpoint_hash):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and OVERWRITE:
        OUTPUT_PATH.unlink()

    if OUTPUT_PATH.exists():
        output = h5py.File(OUTPUT_PATH, "a")

        expected = {
            "source_h5": str(SOURCE_H5),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "checkpoint_sha256": checkpoint_hash,
            "future_horizon": FUTURE_HORIZON,
            "preprocessing_pipeline": (
                "stable_worldmodel.HDF5Dataset(num_steps=1)"
                "+training_get_img_preprocessor"
            ),
        }
        for key, value in expected.items():
            actual = output.attrs.get(key)
            if actual != value:
                output.close()
                raise RuntimeError(
                    f"已有新输出属性不一致：{key}={actual!r}, "
                    f"本次={value!r}。请更换路径或设 OVERWRITE=True。"
                )
        return output

    output = h5py.File(OUTPUT_PATH, "w")

    # 旧文件中这些元数据已经通过边界验证，直接复制；
    # 唯独不复制旧的 frame_latent。
    with h5py.File(OLD_METADATA_H5, "r") as old:
        for key in old.keys():
            if key == "frame_latent":
                continue
            old.copy(key, output)

        for key, value in old.attrs.items():
            if key in {
                "complete",
                "frames_written",
                "latent_dim",
                "format",
                "latent_layer",
            }:
                continue
            output.attrs[key] = value

    output.attrs["format"] = "lewm_gt_latent_framebank_v2"
    output.attrs["source_h5"] = str(SOURCE_H5)
    output.attrs["checkpoint_path"] = str(CHECKPOINT_PATH)
    output.attrs["checkpoint_epoch"] = int(
        checkpoint.get("epoch", -1)
    )
    output.attrs["checkpoint_sha256"] = checkpoint_hash
    output.attrs["future_horizon"] = FUTURE_HORIZON
    output.attrs["latent_layer"] = "encoder_cls_then_projector"
    output.attrs["preprocessing_pipeline"] = (
        "stable_worldmodel.HDF5Dataset(num_steps=1)"
        "+training_get_img_preprocessor"
    )
    output.attrs["items_written"] = 0
    output.attrs["frames_written"] = 0
    output.attrs["complete"] = False
    output.flush()

    return output


def encode_all(
    model,
    frame_dataset,
    row_map,
    output,
    total_frames,
):
    start_item = int(output.attrs.get("items_written", 0))

    if bool(output.attrs.get("complete", False)):
        print("新输出已 complete=True，无需重复计算。")
        return

    if start_item < 0 or start_item > total_frames:
        raise RuntimeError(
            f"items_written={start_item} 非法。"
        )

    wrapped = FrameDatasetWithRow(
        base_dataset=frame_dataset,
        row_map=row_map,
        start_item=start_item,
    )
    loader = DataLoader(
        wrapped,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )

    print("\n========== 开始全量重算 ==========")
    print("device             :", DEVICE)
    print("batch size         :", BATCH_SIZE)
    print("total frames       :", total_frames)
    print("resume item        :", start_item)
    print("remaining          :", total_frames - start_item)
    print("output             :", OUTPUT_PATH)

    begin = time.time()
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_id, (pixels, rows, item_indices) in enumerate(loader):
            # pixels: [B,1,C,H,W]，已经由 HDF5Dataset + 训练 transform 得到。
            pixels = pixels.to(
                DEVICE,
                non_blocking=True,
            )

            z = model.encode({"pixels": pixels})["emb"][:, 0]
            z = z.detach().cpu()

            if z.ndim != 2:
                raise RuntimeError(
                    f"frame latent 应为 [B,D]，实际 {tuple(z.shape)}"
                )
            if not torch.isfinite(z).all():
                raise RuntimeError("latent 中出现 NaN 或 Inf。")

            rows_np = rows.numpy().astype(np.int64)
            items_np = item_indices.numpy().astype(np.int64)

            expected_items = np.arange(
                int(items_np[0]),
                int(items_np[0]) + len(items_np),
                dtype=np.int64,
            )
            if not np.array_equal(items_np, expected_items):
                raise RuntimeError("DataLoader item 顺序不连续。")
            if not np.array_equal(rows_np, items_np):
                raise RuntimeError(
                    "已验证应有 row == dataset item，但当前 batch 不满足。"
                )

            if "frame_latent" not in output:
                latent_dim = int(z.shape[1])
                chunk_rows = min(total_frames, 8192)
                output.create_dataset(
                    "frame_latent",
                    shape=(total_frames, latent_dim),
                    dtype=OUTPUT_DTYPE,
                    chunks=(chunk_rows, latent_dim),
                )
                output.attrs["latent_dim"] = latent_dim
                output.flush()
            else:
                latent_dim = int(output.attrs["latent_dim"])
                if int(z.shape[1]) != latent_dim:
                    raise RuntimeError(
                        f"latent dim {z.shape[1]} != {latent_dim}"
                    )

            write_start = int(rows_np[0])
            write_end = int(rows_np[-1]) + 1

            output["frame_latent"][write_start:write_end] = (
                z.numpy().astype(OUTPUT_DTYPE, copy=False)
            )

            next_item = int(items_np[-1]) + 1
            output.attrs["items_written"] = next_item
            output.attrs["frames_written"] = write_end
            output.flush()

            if (
                batch_id % PRINT_EVERY == 0
                or batch_id == total_batches - 1
            ):
                elapsed = max(time.time() - begin, 1e-6)
                processed = next_item - start_item
                speed = processed / elapsed
                percent = 100.0 * next_item / total_frames
                print(
                    f"[{batch_id + 1:>6}/{total_batches}] "
                    f"frame {next_item:>9}/{total_frames} "
                    f"({percent:6.2f}%) | "
                    f"{speed:8.1f} frame/s"
                )

    if int(output.attrs["items_written"]) != total_frames:
        raise RuntimeError(
            "循环结束但 items_written != total_frames。"
        )

    output.attrs["complete"] = True
    output.flush()

    print("\n========== 新全量 latent 完成 ==========")
    print("frame_latent :", tuple(output["frame_latent"].shape))
    print("complete     :", bool(output.attrs["complete"]))
    print("output       :", OUTPUT_PATH)


def main():
    check_paths()
    cfg = load_config()
    checkpoint_hash = sha256(CHECKPOINT_PATH)

    # 先普通读取 episode 元数据，然后关闭，避免 SWMR 冲突。
    with h5py.File(SOURCE_H5, "r") as source:
        total_frames = int(source["pixels"].shape[0])
        action_dim = int(source["action"].shape[-1])
        ep_offsets = np.asarray(
            source["ep_offset"][:],
            dtype=np.int64,
        )

    model, checkpoint = build_model(cfg, action_dim)

    # num_steps=1 是逐帧 frame-bank，但仍完整经过训练使用的
    # HDF5Dataset 内部读取逻辑和同一个 transform。
    frame_dataset = build_swm_dataset(cfg, num_steps=1)

    try:
        row_map = build_row_map(
            frame_dataset,
            ep_offsets,
            total_frames,
        )

        print("========== 数据映射检查 ==========")
        print("source total frames :", total_frames)
        print("dataset length      :", len(frame_dataset))
        print("row map first/last  :",
              int(row_map[0]), int(row_map[-1]))
        print("checkpoint sha256   :", checkpoint_hash)

        # 非常重要：预检查不通过时，绝不开始全量重算。
        preflight_check(cfg, model, frame_dataset)

        output = create_output(
            checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash,
        )
        try:
            encode_all(
                model=model,
                frame_dataset=frame_dataset,
                row_map=row_map,
                output=output,
                total_frames=total_frames,
            )
        finally:
            output.close()
    finally:
        close_swm_dataset(frame_dataset)


if __name__ == "__main__":
    main()
