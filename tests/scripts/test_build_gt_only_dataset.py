from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = (
    REPO_ROOT / "FLARE-VLA" / "scripts" / "05_distillation" / "10_build_gt_only_dataset.py"
)


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_gt_only_dataset", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_builds_exact_gt_only_sidecars(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder_module()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    rows = []
    for index in range(3):
        rows.append(
            {
                "format": "flare_distillation_index_v1",
                "distillation_index": index,
                "record_id": f"sample_{index:06d}:gt",
                "sample_id": f"sample_{index:06d}",
                "sample_index": index,
                "episode_index": index,
                "local_step": index,
                "task_index": index,
                "source": "gt",
                "teacher_row": None,
            }
        )
    for teacher_row in range(2):
        index = len(rows)
        rows.append(
            {
                "format": "flare_distillation_index_v1",
                "distillation_index": index,
                "record_id": f"sample_{teacher_row:06d}:teacher",
                "sample_id": f"sample_{teacher_row:06d}",
                "sample_index": teacher_row,
                "episode_index": teacher_row,
                "local_step": teacher_row,
                "task_index": teacher_row,
                "source": "teacher",
                "teacher_row": teacher_row,
            }
        )
    write_jsonl(source_dir / "distillation_index.jsonl", rows)

    torch.save(
        {
            "format": "flare_distillation_teacher_records_v1",
            "sample_ids": ["sample_000000", "sample_000001"],
            "sample_indices": torch.tensor([0, 1]),
            "episode_indices": torch.tensor([0, 1]),
            "local_steps": torch.tensor([0, 1]),
            "teacher_actions_lerobot": torch.zeros((2, 8, 7)),
            "teacher_horizon": 8,
            "processor_action_horizon": 16,
            "model_action_horizon": 40,
            "action_dim": 7,
            "action_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        source_dir / "teacher_records_enriched.pt",
    )
    write_json(
        source_dir / "distillation_summary.json",
        {
            "format": "flare_distillation_summary_v1",
            "status": "PASS",
            "gr00t_interface": {
                "teacher_horizon": 8,
                "processor_action_horizon": 16,
                "model_action_horizon": 40,
                "action_dim": 7,
            },
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER_PATH),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--expected-gt-count",
            "3",
        ],
    )
    builder.main()

    output_rows = [
        json.loads(line)
        for line in (output_dir / "distillation_index.jsonl").read_text().splitlines()
    ]
    assert len(output_rows) == 3
    assert [row["distillation_index"] for row in output_rows] == [0, 1, 2]
    assert {row["source"] for row in output_rows} == {"gt"}
    assert all(row["teacher_row"] is None for row in output_rows)

    teacher = torch.load(
        output_dir / "teacher_records_enriched.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert teacher["sample_ids"] == []
    assert tuple(teacher["teacher_actions_lerobot"].shape) == (0, 8, 7)
    assert teacher["teacher_horizon"] == 8
    assert teacher["processor_action_horizon"] == 16
    assert teacher["model_action_horizon"] == 40

    summary = json.loads((output_dir / "distillation_summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["protocol"]["gt_samples"] == 3
    assert summary["protocol"]["teacher_samples"] == 0
    assert summary["protocol"]["total_samples"] == 3
