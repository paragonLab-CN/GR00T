import argparse
from pathlib import Path

import torch


ACTION_ORDER = [
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
]


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def print_tree(x, name="root", indent=0, max_depth=4):
    prefix = "  " * indent

    if indent > max_depth:
        print(f"{prefix}{name}: ...")
        return

    if torch.is_tensor(x):
        print(f"{prefix}{name}: Tensor shape={tuple(x.shape)}, dtype={x.dtype}")
    elif isinstance(x, dict):
        print(f"{prefix}{name}: dict keys={list(x.keys())}")
        for k, v in x.items():
            print_tree(v, name=str(k), indent=indent + 1, max_depth=max_depth)
    elif isinstance(x, list):
        print(f"{prefix}{name}: list len={len(x)}")
        for i, v in enumerate(x[:3]):
            print_tree(v, name=f"[{i}]", indent=indent + 1, max_depth=max_depth)
        if len(x) > 3:
            print(f"{prefix}  ...")
    elif isinstance(x, tuple):
        print(f"{prefix}{name}: tuple len={len(x)}")
        for i, v in enumerate(x[:3]):
            print_tree(v, name=f"({i})", indent=indent + 1, max_depth=max_depth)
    else:
        print(f"{prefix}{name}: {type(x).__name__} = {repr(x)[:120]}")


def pack_candidate_actions(candidate_dict):
    action_list = []
    for key in ACTION_ORDER:
        if key not in candidate_dict:
            raise KeyError(f"Missing action key: {key}")
        value = candidate_dict[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        action_list.append(value.float())

    return torch.cat(action_list, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    path = Path(args.input)
    data = safe_torch_load(path)

    print("Loaded:", path)
    print("Top-level keys:", list(data.keys()))

    print("\n=== Observations structure ===")
    print_tree(data["observations"], name="observations")

    print("\n=== Candidate actions ===")
    candidate_actions_7d = pack_candidate_actions(data["candidate_actions"])
    print("candidate_actions_7d:", tuple(candidate_actions_7d.shape))

    B, K, T, D = candidate_actions_7d.shape
    lewm_actions = candidate_actions_7d.reshape(B * K, T, D).contiguous()
    print("lewm_actions:", tuple(lewm_actions.shape))

    print("\nAlignment check:")
    print(f"One observations object corresponds to B={B} env(s).")
    print(f"For each env, K={K} candidate trajectories are sampled.")
    print(f"Therefore LeWM action batch should be B*K={B*K}.")


if __name__ == "__main__":
    main()
