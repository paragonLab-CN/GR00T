from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "flare" / "03_build_full_manifest.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_full_manifest", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_manifest_window_protocol_is_deterministic_and_deduplicated():
    module = load_script_module()
    actions = np.zeros((64, 7), dtype=np.float32)
    actions[:, 6] = 1.0
    actions[20:, 6] = 0.0
    actions[40:, 0] = 0.8
    action_std = np.ones((7,), dtype=np.float32)

    kwargs = dict(
        episode_index=7,
        actions=actions,
        action_std=action_std,
        first_valid_step=2,
        last_valid_step=55,
        stride=8,
        base_seed=2026,
        gripper_radius=2,
        action_change_topk=2,
        action_change_radius=1,
        max_extra_windows=20,
    )
    first = module.build_episode_window_starts(**kwargs)
    second = module.build_episode_window_starts(**kwargs)

    starts, reasons, _, phase, diagnostics = first
    assert starts == second[0]
    assert phase == second[3]
    assert starts == sorted(set(starts))
    assert all(2 <= step <= 55 for step in starts)
    assert 2 in starts and 55 in starts
    assert any("base_stride" in reasons[step] for step in starts)
    assert any("gripper_transition" in reasons[step] for step in starts)
    assert any("action_change" in reasons[step] for step in starts)
    assert diagnostics["gripper_events"] == 1


def test_episode_selection_manifest_returns_sorted_unique_ids(tmp_path):
    module = load_script_module()
    manifest_path = tmp_path / "selection.jsonl"
    rows = [
        {"episode_index": 9, "local_step": 2},
        {"episode_index": 3, "local_step": 8},
        {"episode_index": 9, "local_step": 16},
        {"episode_index": 5, "local_step": 4},
    ]
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert module.selected_episode_ids_from_manifest(
        manifest_path,
        dataset_episode_count=10,
    ) == [3, 5, 9]


def test_modified_python_files_compile():
    paths = [
        REPO_ROOT / "scripts" / "flare" / "03_build_full_manifest.py",
        REPO_ROOT / "gr00t" / "configs" / "finetune_config.py",
        REPO_ROOT / "gr00t" / "experiment" / "launch_finetune.py",
        REPO_ROOT.parent
        / "FLARE-VLA"
        / "scripts"
        / "05_distillation"
        / "08_build_distillation_dataset.py",
        REPO_ROOT.parent
        / "le-wm"
        / "scripts"
        / "flare"
        / "06_audit_teacher_quality.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
