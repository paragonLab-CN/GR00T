import argparse
from pathlib import Path

import torch


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_image_like_tensor(x: torch.Tensor):
    """
    粗略判断一个 tensor 是否像图像。
    常见形状：
      [B, H, W, C]
      [B, C, H, W]
      [B, T, H, W, C]
      [B, T, C, H, W]
    """
    if not torch.is_tensor(x):
        return False

    shape = tuple(x.shape)

    if x.ndim < 3:
        return False

    # 最后一维是 3 或 4，常见 HWC/RGBA
    if shape[-1] in (1, 3, 4) and max(shape[-3:-1]) >= 32:
        return True

    # 倒数第三维是 1/3/4，常见 CHW
    if x.ndim >= 3 and shape[-3] in (1, 3, 4) and max(shape[-2:]) >= 32:
        return True

    return False


def walk(obj, path="root", results=None):
    if results is None:
        results = []

    if torch.is_tensor(obj):
        if is_image_like_tensor(obj):
            results.append((path, tuple(obj.shape), str(obj.dtype)))
        return results

    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}::{k}", results)
        return results

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}::[{i}]", results)
        return results

    if isinstance(obj, tuple):
        for i, v in enumerate(obj):
            walk(v, f"{path}::({i})", results)
        return results

    return results


def print_tree(obj, name="root", indent=0, max_depth=3):
    prefix = "  " * indent

    if indent > max_depth:
        print(f"{prefix}{name}: ...")
        return

    if torch.is_tensor(obj):
        print(f"{prefix}{name}: Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{prefix}{name}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            print_tree(v, str(k), indent + 1, max_depth)
    elif isinstance(obj, list):
        print(f"{prefix}{name}: list len={len(obj)}")
        for i, v in enumerate(obj[:3]):
            print_tree(v, f"[{i}]", indent + 1, max_depth)
        if len(obj) > 3:
            print(f"{prefix}  ...")
    elif isinstance(obj, tuple):
        print(f"{prefix}{name}: tuple len={len(obj)}")
        for i, v in enumerate(obj[:3]):
            print_tree(v, f"({i})", indent + 1, max_depth)
    else:
        print(f"{prefix}{name}: {type(obj).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    data = safe_torch_load(Path(args.input))

    print("Top-level keys:", list(data.keys()))

    print("\n=== Observation tree ===")
    print_tree(data["observations"], name="observations", max_depth=4)

    print("\n=== Possible image-like tensors ===")
    image_keys = walk(data["observations"], path="observations")

    if not image_keys:
        print("No image-like tensors found. Paste the observation tree output next.")
    else:
        for idx, (path, shape, dtype) in enumerate(image_keys):
            print(f"[{idx}] path={path}")
            print(f"    shape={shape}, dtype={dtype}")


if __name__ == "__main__":
    main()
