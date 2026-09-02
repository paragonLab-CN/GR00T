#!/usr/bin/env python3
"""LIBERO-plus environment smoke test.

Run with the libero_plus_uv venv python:
  LIBERO_CONFIG_PATH=<gr00t>/gr00t/eval/sim/LIBERO/libero_plus_config \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  <libero_plus_uv>/.venv/bin/python scripts/flare/smoke_test_libero_plus_env.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

GR00T_ROOT = Path("/root/autodl-tmp/Isaac-GR00T")
CLASSIFICATION_JSON = (
    GR00T_ROOT
    / "external_dependencies/LIBERO-plus/libero/libero/benchmark/task_classification.json"
)


def main() -> None:
    from libero.libero import benchmark, get_libero_path

    print("=== 1. benchmark dict ===")
    benchmark_dict = benchmark.get_benchmark_dict()
    print("available suites:", sorted(benchmark_dict.keys()))
    for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        suite = benchmark_dict[suite_name]()
        print(f"  {suite_name}: {suite.get_num_tasks()} tasks")

    print("\n=== 2. task_classification.json ===")
    classification = json.loads(CLASSIFICATION_JSON.read_text(encoding="utf-8"))
    for suite_name, entries in classification.items():
        print(f"  {suite_name}: {len(entries)} classified entries")

    print("\n=== 3. paths ===")
    for key in ["bddl_files", "init_states"]:
        print(f"  {key} -> {get_libero_path(key)}")

    # pick one task per perturbation category from libero_spatial
    spatial_entries = classification["libero_spatial"]
    by_category: dict[str, dict] = {}
    for entry in spatial_entries:
        by_category.setdefault(entry["category"], entry)

    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    print("\n=== 4. env smoke (one task per category, reset + 5 steps) ===")
    import gymnasium as gym
    import numpy as np

    for category, entry in sorted(by_category.items()):
        task = benchmark_dict["libero_spatial"]().get_task(entry["id"])
        print(f"\n[{category}] {task.name}")
        print(f"  language: {task.language}")
        register_libero_envs(task_filter=task.name)
        env = gym.make(f"libero_sim/{task.name}")
        obs, info = env.reset(seed=2026)
        obs_keys = set(obs.keys())
        expected = {
            "video.image",
            "video.wrist_image",
            "state.x",
            "state.gripper",
            "annotation.human.action.task_description",
        }
        missing = [k for k in expected if k not in obs_keys]
        if missing:
            raise RuntimeError(f"missing obs keys: {missing}")
        print(f"  obs keys OK: {sorted(obs_keys)}")
        print(f"  initial success: {info.get('success')}")
        action = {
            "action.x": np.array([0.0]),
            "action.y": np.array([0.0]),
            "action.z": np.array([0.0]),
            "action.roll": np.array([0.0]),
            "action.pitch": np.array([0.0]),
            "action.yaw": np.array([0.0]),
            "action.gripper": np.array([0.0]),
        }
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(action)
        print(f"  after 5 steps: success={info.get('success')}, image={obs['video.image'].shape}")
        env.close()

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
