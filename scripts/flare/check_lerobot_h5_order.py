import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def natural_key(name):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else name


def find_parquet_files(lerobot_dir):
    lerobot_dir = Path(lerobot_dir)
    files = sorted(lerobot_dir.glob("data/**/*.parquet"))
    if not files:
        files = sorted(lerobot_dir.glob("**/*.parquet"))
    return files


def load_lerobot_table(lerobot_dir, max_files=None):
    files = find_parquet_files(lerobot_dir)
    if max_files is not None:
        files = files[:max_files]

    print(f"[LeRobot] found {len(files)} parquet files")
    for f in files[:5]:
        print("  ", f)

    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    print("[LeRobot] columns:")
    for c in df.columns:
        print("  ", c)

    required = ["episode_index", "frame_index", "action"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"LeRobot parquet missing required column: {c}")

    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return df


def find_h5_action_key(demo_group):
    candidates = [
        "actions",
        "action",
    ]
    for k in candidates:
        if k in demo_group:
            return k
    raise KeyError(f"Cannot find action key in H5 demo group. keys={list(demo_group.keys())}")


def get_h5_demos(h5_path):
    f = h5py.File(h5_path, "r")

    if "data" in f:
        root = f["data"]
    else:
        root = f

    demo_names = [k for k in root.keys() if isinstance(root[k], h5py.Group)]
    demo_names = sorted(demo_names, key=natural_key)

    print(f"[H5] found {len(demo_names)} demos")
    print("[H5] first demos:", demo_names[:5])

    return f, root, demo_names


def to_np_action(x):
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack(arr)
    return arr.astype(np.float32)


def compare_one_episode(h5_root, h5_demo_name, lerobot_ep_df, compare_dim=7, n_check=20):
    demo = h5_root[h5_demo_name]
    h5_action_key = find_h5_action_key(demo)
    h5_actions = np.asarray(demo[h5_action_key]).astype(np.float32)

    lr_actions = np.stack(lerobot_ep_df["action"].map(to_np_action).to_list()).astype(np.float32)

    h5_len = len(h5_actions)
    lr_len = len(lr_actions)

    print(f"\nCompare {h5_demo_name} vs LeRobot episode {int(lerobot_ep_df['episode_index'].iloc[0])}")
    print("  H5 action key:", h5_action_key)
    print("  H5 action shape:", h5_actions.shape)
    print("  LeRobot action shape:", lr_actions.shape)
    print("  H5 length:", h5_len)
    print("  LeRobot length:", lr_len)

    min_len = min(h5_len, lr_len, n_check)
    dim = min(compare_dim, h5_actions.shape[-1], lr_actions.shape[-1])

    diffs = []
    for t in range(min_len):
        diff = np.max(np.abs(h5_actions[t, :dim] - lr_actions[t, :dim]))
        diffs.append(diff)

    diffs = np.asarray(diffs)
    print(f"  checked first {min_len} frames, first {dim} action dims")
    print("  max diff:", float(diffs.max()) if len(diffs) else None)
    print("  mean diff:", float(diffs.mean()) if len(diffs) else None)
    print("  first 10 diffs:", diffs[:10].tolist())

    same_length = h5_len == lr_len
    action_close = bool(len(diffs) > 0 and diffs.max() < 1e-4)

    return {
        "h5_demo": h5_demo_name,
        "lerobot_episode": int(lerobot_ep_df["episode_index"].iloc[0]),
        "h5_len": int(h5_len),
        "lerobot_len": int(lr_len),
        "same_length": same_length,
        "max_action_diff": float(diffs.max()) if len(diffs) else None,
        "mean_action_diff": float(diffs.mean()) if len(diffs) else None,
        "action_close": action_close,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", required=True, help="One LIBERO task .hdf5 file")
    parser.add_argument("--lerobot-dir", required=True, help="LeRobot dataset root dir")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--n-check", type=int, default=20)
    parser.add_argument("--compare-dim", type=int, default=7)
    parser.add_argument("--max-parquet-files", type=int, default=None)
    parser.add_argument("--out", default="outputs/flare_alignment_check/order_check_result.json")
    args = parser.parse_args()

    df = load_lerobot_table(args.lerobot_dir, args.max_parquet_files)
    h5_file, h5_root, h5_demos = get_h5_demos(args.h5_path)

    episode_ids = sorted(df["episode_index"].unique().tolist())
    print(f"[LeRobot] found {len(episode_ids)} episodes")
    print("[LeRobot] first episodes:", episode_ids[:10])

    n = min(args.num_episodes, len(h5_demos), len(episode_ids))
    results = []

    for i in range(n):
        h5_demo_name = h5_demos[i]
        ep_id = episode_ids[i]
        ep_df = df[df["episode_index"] == ep_id].sort_values("frame_index")
        results.append(
            compare_one_episode(
                h5_root=h5_root,
                h5_demo_name=h5_demo_name,
                lerobot_ep_df=ep_df,
                compare_dim=args.compare_dim,
                n_check=args.n_check,
            )
        )

    h5_file.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved result to {out}")

    all_same_len = all(r["same_length"] for r in results)
    all_action_close = all(r["action_close"] for r in results)

    print("\n========== FINAL JUDGEMENT ==========")
    print("all_same_len:", all_same_len)
    print("all_action_close:", all_action_close)

    if all_same_len and all_action_close:
        print("结论：至少在检查范围内，H5 demo_i frame_t 和 LeRobot episode_i frame_t 顺序一致。")
        print("可以暂时不用复杂映射，但仍建议保存 episode_index/frame_index。")
    elif all_action_close and not all_same_len:
        print("结论：前几帧动作接近，但 episode 长度不同。可能存在 no-op 删除或尾部截断。")
        print("不建议直接用 frame_id 对齐，需要记录/恢复 no-op 映射。")
    else:
        print("结论：顺序或动作定义不一致，不能假设 demo_i frame_t == episode_i frame_t。")
        print("必须建立映射，或者统一从同一个源重新转换 H5 和 LeRobot。")


if __name__ == "__main__":
    main()
