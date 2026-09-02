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


def pack_candidate_actions(candidate_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    把 GR00T dict 格式动作拼成完整 7 维动作。

    输入:
        candidate_dict:
            {
                "action.x":       [B, K, T, 1],
                "action.y":       [B, K, T, 1],
                ...
                "action.gripper": [B, K, T, 1],
            }

    输出:
        candidate_actions_7d: [B, K, T, 7]
    """
    missing_keys = [k for k in ACTION_ORDER if k not in candidate_dict]
    if missing_keys:
        raise KeyError(f"Missing action keys: {missing_keys}")

    action_list = []
    for key in ACTION_ORDER:
        value = candidate_dict[key]

        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        print(f"{key}: {tuple(value.shape)}")

        if value.ndim != 4:
            raise ValueError(f"{key} should have shape [B, K, T, 1], got {value.shape}")

        if value.shape[-1] != 1:
            raise ValueError(f"{key} last dim should be 1, got {value.shape}")

        action_list.append(value.float())

    candidate_actions_7d = torch.cat(action_list, dim=-1)

    return candidate_actions_7d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one saved step_xxxxxx.pt file.",
    )
    parser.add_argument(
        "--save-packed",
        type=str,
        default="",
        help="Optional path to save packed [B, K, T, 7] tensor.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = torch.load(input_path, map_location="cpu")

    print("Loaded:", input_path)
    print("Top-level keys:", data.keys())

    candidate_dict = data["candidate_actions"]

    print("\nOriginal candidate action fields:")
    candidate_actions_7d = pack_candidate_actions(candidate_dict)

    print("\nPacked candidate_actions_7d shape:", tuple(candidate_actions_7d.shape))

    B, K, T, D = candidate_actions_7d.shape
    print(f"B = {B}")
    print(f"K = {K}")
    print(f"T = {T}")
    print(f"D = {D}")

    # 给 LeWM 用时，通常把 K 条候选轨迹展开成 batch
    candidate_actions_flat = candidate_actions_7d.reshape(B * K, T, D)
    print("\nFor LeWM candidate_actions_flat shape:", tuple(candidate_actions_flat.shape))

    # 打印前两个候选轨迹的第一个动作，确认它们确实不同
    if K >= 2:
        print("\nCandidate 0 first action:", candidate_actions_7d[0, 0, 0])
        print("Candidate 1 first action:", candidate_actions_7d[0, 1, 0])
        diff = torch.max(torch.abs(candidate_actions_7d[0, 0] - candidate_actions_7d[0, 1]))
        print("max |candidate 0 - candidate 1|:", diff.item())

    if args.save_packed:
        save_path = Path(args.save_packed)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_file": str(input_path),
                "candidate_actions_7d": candidate_actions_7d,
                "candidate_actions_flat": candidate_actions_flat,
                "note": "candidate_actions_7d shape is [B, K, T, 7]; candidate_actions_flat shape is [B*K, T, 7]",
            },
            save_path,
        )
        print("\nSaved packed tensor to:", save_path)


if __name__ == "__main__":
    main()























