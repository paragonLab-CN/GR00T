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


def repeat_tensor_first_dim_for_k(x: torch.Tensor, B: int, K: int):
    """
    如果 x 的第 0 维是 B，就复制成 B*K。
    例如:
        [1, 224, 224, 3] -> [4, 224, 224, 3]
        [1, 8]           -> [4, 8]
    """
    if x.ndim >= 1 and x.shape[0] == B:
        repeat_shape = [1, K] + [1] * (x.ndim - 1)
        return x.unsqueeze(1).repeat(*repeat_shape).reshape(B * K, *x.shape[1:]).contiguous()
    return x


def repeat_observations_for_candidates(obs, B: int, K: int):
    """
    递归复制 observations，使它和 lewm_actions 的 batch 维度一致。

    输入 obs:
        batch 维度通常是 B

    输出 repeated_obs:
        batch 维度变成 B*K
    """
    if torch.is_tensor(obs):
        return repeat_tensor_first_dim_for_k(obs, B, K)

    if isinstance(obs, dict):
        return {k: repeat_observations_for_candidates(v, B, K) for k, v in obs.items()}

    if isinstance(obs, list):
        # 如果这个 list 本身就是 batch 维度，例如 len(obs)=B
        if len(obs) == B:
            repeated = []
            for item in obs:
                for _ in range(K):
                    repeated.append(item)
            return repeated

        # 否则递归处理 list 内部元素
        return [repeat_observations_for_candidates(v, B, K) for v in obs]

    if isinstance(obs, tuple):
        return tuple(repeat_observations_for_candidates(v, B, K) for v in obs)

    return obs


def print_tree(x, name="root", indent=0, max_depth=3):
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
        for i, v in enumerate(x[:2]):
            print_tree(v, name=f"[{i}]", indent=indent + 1, max_depth=max_depth)
        if len(x) > 2:
            print(f"{prefix}  ...")
    else:
        print(f"{prefix}{name}: {type(x).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--save-output", type=str, required=True)
    args = parser.parse_args()

    data = safe_torch_load(args.input)

    observations = data["observations"]
    candidate_actions_7d = pack_candidate_actions(data["candidate_actions"])

    print("candidate_actions_7d:", tuple(candidate_actions_7d.shape))

    B, K, T, D = candidate_actions_7d.shape

    # [B, K, T, D] -> [B*K, T, D]
    lewm_actions = candidate_actions_7d.reshape(B * K, T, D).contiguous()

    # observations: [B, ...] -> [B*K, ...]
    lewm_observations = repeat_observations_for_candidates(observations, B, K)

    print("\n=== lewm_actions ===")
    print("lewm_actions:", tuple(lewm_actions.shape))

    print("\n=== lewm_observations structure ===")
    print_tree(lewm_observations, name="lewm_observations")

    save_path = Path(args.save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "source_file": str(args.input),
            "rollout_step": int(data["rollout_step"]),
            "B": B,
            "K": K,
            "T": T,
            "D": D,

            # LeWM-ready aligned inputs
            "lewm_actions": lewm_actions,
            "lewm_observations": lewm_observations,

            # Also keep original for debugging
            "candidate_actions_7d": candidate_actions_7d,
            "note": (
                "lewm_actions shape is [B*K, T, 7]. "
                "lewm_observations is repeated from the current observation "
                "so that each candidate action has its matching observation."
            ),
        },
        save_path,
    )

    print("\nSaved LeWM-ready obs-action batch to:", save_path)


if __name__ == "__main__":
    main()
