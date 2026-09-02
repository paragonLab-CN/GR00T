import argparse
import json
from pathlib import Path

import pandas as pd


def read_json_or_jsonl(path: Path, max_lines: int = 5):
    print(f"\n=== {path.name} ===")

    if not path.exists():
        print("Not found:", path)
        return

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    print("...")
                    break
                print(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            print("keys:", list(obj.keys()))
            print(json.dumps(obj, indent=2, ensure_ascii=False)[:3000])
        else:
            print(type(obj))
            print(str(obj)[:3000])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/root/autodl-tmp/Isaac-GR00T/demo_data/libero_demo",
    )
    parser.add_argument("--episode", type=str, default="episode_000000.parquet")
    args = parser.parse_args()

    root = Path(args.root)
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"
    video_dir = root / "videos" / "chunk-000"

    print("Dataset root:", root)
    print("Data dir:", data_dir)
    print("Meta dir:", meta_dir)
    print("Video dir:", video_dir)

    print("\n=== Parquet files ===")
    parquet_files = sorted(data_dir.glob("*.parquet"))
    for p in parquet_files[:10]:
        print(p.name)
    print("num parquet files:", len(parquet_files))

    print("\n=== Video files ===")
    video_files = sorted(video_dir.rglob("*"))
    video_files = [p for p in video_files if p.is_file()]
    for p in video_files[:20]:
        print(p.relative_to(root))
    print("num video files:", len(video_files))

    for name in [
        "info.json",
        "modality.json",
        "stats.json",
        "episodes.jsonl",
        "tasks.jsonl",
    ]:
        read_json_or_jsonl(meta_dir / name)

    parquet_path = data_dir / args.episode
    print(f"\n=== Read parquet: {parquet_path} ===")

    df = pd.read_parquet(parquet_path)

    print("shape:", df.shape)

    print("\nColumns:")
    for c in df.columns:
        print(" ", c, "| dtype:", df[c].dtype)

    print("\nHead:")
    print(df.head(3))

    print("\nFirst row detailed:")
    row = df.iloc[0]
    for c in df.columns:
        v = row[c]
        print(f"\n[{c}] type={type(v)}")
        s = repr(v)
        print(s[:1000])


if __name__ == "__main__":
    main()
