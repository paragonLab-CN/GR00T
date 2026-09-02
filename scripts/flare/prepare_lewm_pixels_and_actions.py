import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


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


def get_by_path(obj, path: str):
    """
    path 例如:
        observations::video.image
        observations::video.wrist_image
    """
    parts = path.split("::")

    if parts[0] in ("root", "observations"):
        parts = parts[1:]

    cur = obj
    for p in parts:
        if p.startswith("[") and p.endswith("]"):
            cur = cur[int(p[1:-1])]
        elif p.startswith("(") and p.endswith(")"):
            cur = cur[int(p[1:-1])]
        else:
            cur = cur[p]

    return cur


def pack_candidate_actions(candidate_dict):
    """
    把 GR00T 的 dict 动作拼成 [B, K, T, 7]
    """
    action_list = []

    for key in ACTION_ORDER:
        if key not in candidate_dict:
            raise KeyError(f"Missing action key: {key}")

        x = candidate_dict[key]

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        print(f"{key}: {tuple(x.shape)}")

        action_list.append(x.float())

    candidate_actions_7d = torch.cat(action_list, dim=-1)

    return candidate_actions_7d


def normalize_image_tensor(img):
    """
    把图像统一转成 [B, 3, 224, 224]，float，范围 0~1。

    你现在的输入是:
        [1, 1, 256, 256, 3]

    处理后变成:
        [1, 3, 224, 224]
    """
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)

    img = img.detach().cpu()

    print("raw image shape:", tuple(img.shape), img.dtype)

    # 当前你的图像是 [B, H, 256, 256, 3]
    # H 是历史帧数量，现在 H=1，所以取最后一帧
    if img.ndim == 5:
        img = img[:, -1]

    # [B, H, W, C] -> [B, C, H, W]
    if img.ndim == 4 and img.shape[-1] in (1, 3, 4):
        img = img[..., :3]
        img = img.permute(0, 3, 1, 2).contiguous()

    # [B, C, H, W]
    elif img.ndim == 4 and img.shape[1] in (1, 3, 4):
        img = img[:, :3].contiguous()

    else:
        raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")

    img = img.float()

    # uint8 图像一般是 0~255
    if img.max() > 2.0:
        img = img / 255.0

    # resize 到 LeWM 常用的 224x224
    img = F.interpolate(
        img,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help='Example: "observations::video.image"',
    )
    parser.add_argument("--save-output", type=str, required=True)
    args = parser.parse_args()

    data = safe_torch_load(args.input)

    observations = data["observations"]
    candidate_dict = data["candidate_actions"]

    print("Loaded:", args.input)
    print("Using image path:", args.image_path)

    print("\n=== Pack candidate actions ===")
    candidate_actions_7d = pack_candidate_actions(candidate_dict)

    print("\ncandidate_actions_7d:", tuple(candidate_actions_7d.shape))

    B, K, T, D = candidate_actions_7d.shape

    if D != 7:
        raise ValueError(f"Expected action dim 7, got {D}")

    print(f"B={B}, K={K}, T={T}, D={D}")

    print("\n=== Prepare pixels ===")
    raw_img = get_by_path(observations, args.image_path)
    pixels_bchw = normalize_image_tensor(raw_img)

    print("pixels_bchw:", tuple(pixels_bchw.shape), pixels_bchw.dtype)

    if pixels_bchw.shape[0] != B:
        raise ValueError(
            f"Image batch B={pixels_bchw.shape[0]} does not match action B={B}"
        )

    # LeWM rollout 需要:
    # info["pixels"] = [B, K, H, C, 224, 224]
    # 现在只有当前帧，所以 H=1
    pixels = pixels_bchw.unsqueeze(1).unsqueeze(2).repeat(
        1, K, 1, 1, 1, 1
    ).contiguous()

    # action_sequence = [B, K, T, 7]
    action_sequence = candidate_actions_7d.contiguous()

    print("\n=== LeWM-ready input ===")
    print("info['pixels']:", tuple(pixels.shape))
    print("action_sequence:", tuple(action_sequence.shape))

    save_path = Path(args.save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "source_file": args.input,
            "image_path": args.image_path,
            "info": {
                "pixels": pixels,
            },
            "action_sequence": action_sequence,
            "B": B,
            "K": K,
            "T": T,
            "D": D,
            "note": (
                "For LeWM rollout: info['pixels'] is [B,K,H,C,224,224], "
                "action_sequence is [B,K,T,7]. H=1 for current-frame test."
            ),
        },
        save_path,
    )

    print("\nSaved LeWM input to:", save_path)


if __name__ == "__main__":
    main()
