import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_CANDIDATES = {
    "agentview": [
        "obs/agentview_rgb",
        "obs/agentview_image",
        "obs/agentview",
        "obs/robot0_agentview_left_image",
        "obs/robot0_agentview_image",
    ],
    "wrist": [
        "obs/eye_in_hand_rgb",
        "obs/wrist_rgb",
        "obs/wrist_image",
        "obs/robot0_eye_in_hand_image",
        "obs/robot0_wrist_image",
    ],
}

STATE_CANDIDATES = [
    "obs/ee_states",
    "obs/joint_states",
    "obs/gripper_states",
    "obs/robot0_eef_pos",
    "obs/robot0_eef_quat",
    "obs/robot0_gripper_qpos",
]


def find_files(src_dir: Path):
    files = sorted(src_dir.rglob("*.hdf5")) + sorted(src_dir.rglob("*.h5"))
    files = [p for p in files if not p.name.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {src_dir}")
    return files


def get_demo_groups(f):
    if "data" in f and isinstance(f["data"], h5py.Group):
        root = f["data"]
        return [root[k] for k in sorted(root.keys()) if isinstance(root[k], h5py.Group)]

    demos = []
    for k in sorted(f.keys()):
        if isinstance(f[k], h5py.Group) and ("demo" in k.lower() or "traj" in k.lower()):
            demos.append(f[k])
    if demos:
        return demos

    return [f]


def get_dataset(g, path):
    try:
        obj = g[path]
        if isinstance(obj, h5py.Dataset):
            return obj
    except Exception:
        return None
    return None


def select_image_dataset(g, camera):
    candidates = IMAGE_CANDIDATES.get(camera, [])
    for p in candidates:
        ds = get_dataset(g, p)
        if ds is not None:
            return ds, p

    obs = g.get("obs", None)
    if isinstance(obs, h5py.Group):
        image_like = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 4:
                lname = name.lower()
                if "rgb" in lname or "image" in lname or "agentview" in lname or "eye" in lname or "wrist" in lname:
                    image_like.append(("obs/" + name, obj))
        obs.visititems(visitor)
        if image_like:
            return image_like[0][1], image_like[0][0]

    raise KeyError("Cannot find image dataset. Please inspect hdf5 keys and add the image key to IMAGE_CANDIDATES.")


def select_action_dataset(g):
    for p in ["actions", "action"]:
        ds = get_dataset(g, p)
        if ds is not None:
            return ds, p
    raise KeyError("Cannot find actions/action dataset.")


def normalize_image(img, size):
    img = np.asarray(img)

    if img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[-1] not in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    if img.shape[-1] > 3:
        img = img[..., :3]

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.shape[0] != size or img.shape[1] != size:
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)

    return img


def read_state(g, T):
    parts = []

    for p in STATE_CANDIDATES:
        ds = get_dataset(g, p)
        if ds is not None:
            arr = np.asarray(ds[:T], dtype=np.float32)
            arr = arr.reshape(arr.shape[0], -1)
            parts.append(arr)

    if parts:
        return np.concatenate(parts, axis=-1)

    ds = get_dataset(g, "states")
    if ds is not None:
        arr = np.asarray(ds[:T], dtype=np.float32)
        arr = arr.reshape(arr.shape[0], -1)
        return arr

    return np.zeros((T, 1), dtype=np.float32)


def scan_dataset(files, camera, max_files=None, max_demos_per_file=None):
    total_frames = 0
    total_episodes = 0
    max_state_dim = 1
    action_dim = None
    used_files = files[:max_files] if max_files else files

    for fp in tqdm(used_files, desc="Scanning"):
        with h5py.File(fp, "r") as f:
            demos = get_demo_groups(f)
            if max_demos_per_file:
                demos = demos[:max_demos_per_file]

            for g in demos:
                img_ds, _ = select_image_dataset(g, camera)
                act_ds, _ = select_action_dataset(g)

                T = min(img_ds.shape[0], act_ds.shape[0])
                if T <= 1:
                    continue

                act_shape = act_ds.shape
                cur_action_dim = int(np.prod(act_shape[1:]))
                action_dim = cur_action_dim if action_dim is None else action_dim

                st = read_state(g, min(T, 2))
                max_state_dim = max(max_state_dim, st.shape[-1])

                total_frames += T
                total_episodes += 1

    if total_frames == 0:
        raise RuntimeError("No valid frames found.")

    return used_files, total_frames, total_episodes, action_dim, max_state_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="agentview", choices=["agentview", "wrist"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-demos-per-file", type=int, default=None)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_files(src_dir)
    files, total_frames, total_episodes, action_dim, state_dim = scan_dataset(
        files,
        camera=args.camera,
        max_files=args.max_files,
        max_demos_per_file=args.max_demos_per_file,
    )

    print("Source dir:", src_dir)
    print("Output:", out_path)
    print("Files:", len(files))
    print("Episodes:", total_episodes)
    print("Frames:", total_frames)
    print("Action dim:", action_dim)
    print("State dim:", state_dim)
    print("Camera:", args.camera)
    print("Image size:", args.image_size)
    print("Pixel chunk frames:", args.chunk_frames)
    print("Pixel compression: gzip level 1")

    if action_dim != 7:
        print(f"[WARN] action_dim is {action_dim}, not 7. LeWM can still load it only if your config/model expects this dim.")

    ep_len = []
    ep_offset = []

    with h5py.File(out_path, "w") as out:
        pixel_chunk_frames = min(args.chunk_frames, total_frames)
        pixels = out.create_dataset(
            "pixels",
            shape=(total_frames, args.image_size, args.image_size, 3),
            dtype=np.uint8,
            # LeWM 默认读取 history_size(3) + num_preds(1) = 4 帧。
            # 让一个 chunk 对应 4 帧，避免读取一个样本时加载原来的 128 帧大块。
            chunks=(pixel_chunk_frames, args.image_size, args.image_size, 3),
            # GZIP 是无损压缩；level=1 优先保证读取速度。
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        action = out.create_dataset(
            "action",
            shape=(total_frames, action_dim),
            dtype=np.float32,
            chunks=(min(4096, total_frames), action_dim),
        )
        state = out.create_dataset(
            "state",
            shape=(total_frames, state_dim),
            dtype=np.float32,
            chunks=(min(4096, total_frames), state_dim),
        )
        proprio = out.create_dataset(
            "proprio",
            shape=(total_frames, state_dim),
            dtype=np.float32,
            chunks=(min(4096, total_frames), state_dim),
        )
        done = out.create_dataset("done", shape=(total_frames,), dtype=np.bool_)
        episode_idx = out.create_dataset("episode_idx", shape=(total_frames,), dtype=np.int64)
        index = out.create_dataset("index", shape=(total_frames,), dtype=np.int64)
        step_idx = out.create_dataset("step_idx", shape=(total_frames,), dtype=np.int64)
        task_index = out.create_dataset("task_index", shape=(total_frames,), dtype=np.int64)
        timestamp = out.create_dataset("timestamp", shape=(total_frames,), dtype=np.float32)

        cursor = 0
        epi = 0

        for task_id, fp in enumerate(tqdm(files, desc="Converting files")):
            with h5py.File(fp, "r") as f:
                demos = get_demo_groups(f)
                if args.max_demos_per_file:
                    demos = demos[:args.max_demos_per_file]

                for g in tqdm(demos, desc=fp.name, leave=False):
                    img_ds, img_key = select_image_dataset(g, args.camera)
                    act_ds, act_key = select_action_dataset(g)

                    T = min(img_ds.shape[0], act_ds.shape[0])
                    if T <= 1:
                        continue

                    act = np.asarray(act_ds[:T], dtype=np.float32).reshape(T, -1)
                    st = read_state(g, T)
                    if st.shape[-1] < state_dim:
                        pad = np.zeros((T, state_dim - st.shape[-1]), dtype=np.float32)
                        st = np.concatenate([st, pad], axis=-1)
                    elif st.shape[-1] > state_dim:
                        st = st[:, :state_dim]

                    start = cursor
                    end = cursor + T

                    for i in range(T):
                        pixels[start + i] = normalize_image(img_ds[i], args.image_size)

                    action[start:end] = act
                    state[start:end] = st
                    proprio[start:end] = st
                    done[start:end] = False
                    done[end - 1] = True
                    episode_idx[start:end] = epi
                    index[start:end] = np.arange(start, end)
                    step_idx[start:end] = np.arange(T)
                    task_index[start:end] = task_id
                    timestamp[start:end] = np.arange(T, dtype=np.float32)

                    ep_offset.append(start)
                    ep_len.append(T)

                    cursor = end
                    epi += 1

        out.create_dataset("ep_len", data=np.asarray(ep_len, dtype=np.int64))
        out.create_dataset("ep_offset", data=np.asarray(ep_offset, dtype=np.int64))

        out.attrs["source"] = str(src_dir)
        out.attrs["camera"] = args.camera
        out.attrs["image_size"] = args.image_size
        out.attrs["num_files"] = len(files)
        out.attrs["num_episodes"] = len(ep_len)
        out.attrs["num_frames"] = cursor

    print("Done:", out_path)
    print("Final frames:", cursor)
    print("Final episodes:", len(ep_len))


if __name__ == "__main__":
    main()
