import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def read_video_with_cv2(video_path: Path):
    """
    读取 mp4 视频，返回 RGB frames:
        frames: uint8, [T, H, W, 3]
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "Missing cv2. Install it with:\n"
            "  python -m pip install opencv-python-headless"
        ) from e

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return np.stack(frames, axis=0)


def frames_to_tensor_224(frames_uint8: np.ndarray):
    """
    输入:
        frames_uint8: [T, H, W, 3], uint8, RGB

    输出:
        frames_tensor: [T, 3, 224, 224], float32, 0~1
    """
    frames = torch.from_numpy(frames_uint8).float()  # [T, H, W, 3]

    if frames.max() > 2.0:
        frames = frames / 255.0

    frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]

    frames = F.interpolate(
        frames,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    return frames


def build_windows(frames, actions, states, action_horizon=16, history=1, stride=1, max_windows=0):
    """
    构造 LeWM 训练窗口。

    frames:
        [L, 3, 224, 224]

    actions:
        [L, 7]

    states:
        [L, 8]

    输出:
        pixels:        [N, history, 3, 224, 224]
        actions:       [N, action_horizon, 7]
        future_pixels: [N, action_horizon, 3, 224, 224]
        states:        [N, history, 8]
    """
    L = frames.shape[0]

    windows_pixels = []
    windows_actions = []
    windows_future_pixels = []
    windows_states = []

    # 需要保证:
    # 当前历史帧结束在 t
    # actions 取 t : t+action_horizon
    # future_pixels 取 t+1 : t+1+action_horizon
    start_t = history - 1
    end_t = L - action_horizon - 1

    for t in range(start_t, end_t + 1, stride):
        hist_start = t - history + 1
        hist_end = t + 1

        pixels_t = frames[hist_start:hist_end]  # [history, 3, 224, 224]
        states_t = states[hist_start:hist_end]  # [history, 8]

        action_seq = actions[t : t + action_horizon]  # [action_horizon, 7]

        future_seq = frames[t + 1 : t + 1 + action_horizon]  # [action_horizon, 3, 224, 224]

        if pixels_t.shape[0] != history:
            continue
        if action_seq.shape[0] != action_horizon:
            continue
        if future_seq.shape[0] != action_horizon:
            continue

        windows_pixels.append(pixels_t)
        windows_actions.append(action_seq)
        windows_future_pixels.append(future_seq)
        windows_states.append(states_t)

        if max_windows > 0 and len(windows_pixels) >= max_windows:
            break

    if len(windows_pixels) == 0:
        raise RuntimeError("No valid windows built. Check episode length and horizon.")

    pixels = torch.stack(windows_pixels, dim=0)
    action_seq = torch.stack(windows_actions, dim=0)
    future_pixels = torch.stack(windows_future_pixels, dim=0)
    state_seq = torch.stack(windows_states, dim=0)

    return pixels, action_seq, future_pixels, state_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/root/autodl-tmp/Isaac-GR00T/demo_data/libero_demo",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--camera",
        type=str,
        default="observation.images.image",
        choices=["observation.images.image", "observation.images.wrist_image"],
    )
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=16,
        help="For smoke test, only save a few windows. Use 0 for all windows.",
    )
    parser.add_argument("--save-output", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)

    parquet_path = (
        root
        / "data"
        / "chunk-000"
        / f"episode_{args.episode_index:06d}.parquet"
    )

    video_path = (
        root
        / "videos"
        / "chunk-000"
        / args.camera
        / f"episode_{args.episode_index:06d}.mp4"
    )

    print("Parquet:", parquet_path)
    print("Video:", video_path)

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    df = pd.read_parquet(parquet_path)

    print("Parquet shape:", df.shape)
    print("Columns:", list(df.columns))

    actions_np = np.stack(df["action"].to_numpy()).astype("float32")  # [L, 7]
    states_np = np.stack(df["observation.state"].to_numpy()).astype("float32")  # [L, 8]

    actions = torch.from_numpy(actions_np)
    states = torch.from_numpy(states_np)

    print("actions:", tuple(actions.shape), actions.dtype)
    print("states:", tuple(states.shape), states.dtype)

    frames_np = read_video_with_cv2(video_path)
    print("raw video frames:", frames_np.shape, frames_np.dtype)

    frames = frames_to_tensor_224(frames_np)
    print("frames:", tuple(frames.shape), frames.dtype)

    if frames.shape[0] != len(df):
        print(
            f"[WARNING] video frames ({frames.shape[0]}) != parquet rows ({len(df)}). "
            "Will use the shorter length."
        )
        L = min(frames.shape[0], len(df))
        frames = frames[:L]
        actions = actions[:L]
        states = states[:L]

    pixels, action_seq, future_pixels, state_seq = build_windows(
        frames=frames,
        actions=actions,
        states=states,
        action_horizon=args.action_horizon,
        history=args.history,
        stride=args.stride,
        max_windows=args.max_windows,
    )

    print("\n=== LeWM training windows ===")
    print("pixels:", tuple(pixels.shape), pixels.dtype)
    print("actions:", tuple(action_seq.shape), action_seq.dtype)
    print("future_pixels:", tuple(future_pixels.shape), future_pixels.dtype)
    print("states:", tuple(state_seq.shape), state_seq.dtype)

    save_path = Path(args.save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "root": str(root),
            "episode_index": args.episode_index,
            "camera": args.camera,
            "history": args.history,
            "action_horizon": args.action_horizon,
            "pixels": pixels,
            "actions": action_seq,
            "future_pixels": future_pixels,
            "states": state_seq,
            "note": (
                "Smoke-test LIBERO windows for LeWM. "
                "pixels=[N,H,3,224,224], actions=[N,T,7], "
                "future_pixels=[N,T,3,224,224]."
            ),
        },
        save_path,
    )

    print("\nSaved:", save_path)


if __name__ == "__main__":
    main()
