#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def model_dir_report(path: Path) -> dict:
    index_path = path / "model.safetensors.index.json"
    refs = []
    missing = []
    if index_path.is_file():
        data = load_json(index_path)
        refs = sorted(set(data.get("weight_map", {}).values()))
        missing = [name for name in refs if not (path / name).is_file()]
    checkpoints = {}
    for ckpt in sorted(path.glob("checkpoint-*")):
        checkpoints[ckpt.name] = {
            "shards": sorted(p.name for p in ckpt.glob("model*.safetensors")),
            "has_index": (ckpt / "model.safetensors.index.json").is_file(),
        }
    return {
        "final_shards": sorted(p.name for p in path.glob("model*.safetensors")),
        "has_index": index_path.is_file(),
        "index_refs": refs,
        "missing_index_refs": missing,
        "checkpoints": checkpoints,
    }


def dataset_entries(config_path: Path) -> list[dict]:
    text = config_path.read_text(encoding="utf-8")
    # PyYAML cannot construct the Python object tags safely, so parse only the
    # human-readable conf.yaml when available.
    data = yaml.safe_load(text)
    return data["data"]["datasets"]


def main() -> None:
    manifest = ROOT / "Isaac-GR00T/outputs/flare_full_h8_s8/full_manifest.jsonl"
    manifest_summary_path = manifest.with_name("full_manifest_summary.json")
    candidate_dir = manifest.parent / "candidates"
    candidate_summary_path = manifest.parent / "candidate_generation_summary.json"

    score_root = ROOT / "le-wm/outputs/flare_full_h8_s8"
    validation_summary_path = score_root / "validation_summary.json"
    score_parts = score_root / "score_parts"
    audit_root = score_root / "teacher_quality_audit_high_medium"
    audit_csv = audit_root / "teacher_quality_audit.csv"
    audit_summary_path = audit_root / "teacher_quality_summary.json"

    sidecar_root = ROOT / "Isaac-GR00T/outputs/flare_full_teacher_high_medium_only"
    distill_index = sidecar_root / "distillation_index.jsonl"
    distill_summary_path = sidecar_root / "distillation_summary.json"

    gt_root = ROOT / "Isaac-GR00T/outputs/gr00t_full_gt_only_steps3000/gr00t_full_gt_only"
    flare_root = ROOT / "Isaac-GR00T/outputs/gr00t_full_gt_flare_steps3000/gr00t_full_gt_flare_h8_s8"

    manifest_summary = load_json(manifest_summary_path)
    candidate_summary = load_json(candidate_summary_path)
    validation_summary = load_json(validation_summary_path)
    audit_summary = load_json(audit_summary_path)
    distill_summary = load_json(distill_summary_path)

    df = pd.read_csv(audit_csv)
    selected = df[df["use_for_distillation"].astype(bool)].copy()
    selected["candidate_beats_oracle_mse"] = selected["best_mse_score"] < selected["oracle_mse_score"]
    selected["oracle_relative_improvement"] = (
        selected["oracle_mse_score"] - selected["best_mse_score"]
    ) / (selected["oracle_mse_score"].abs() + 1e-12)

    report = {
        "counts": {
            "manifest_lines": count_lines(manifest),
            "manifest_summary_windows": manifest_summary["selected_windows"],
            "candidate_files": len(list(candidate_dir.glob("full_*.pt"))),
            "candidate_summary_selected": candidate_summary["counters"]["selected"],
            "candidate_summary_completed": candidate_summary["counters"]["generated"]
            + candidate_summary["counters"]["skipped_valid"],
            "candidate_summary_failed": candidate_summary["counters"]["failed"],
            "score_part_files": len(list(score_parts.glob("full_*.pt"))),
            "validation_requested": validation_summary["requested_samples"],
            "validation_scored": validation_summary["scored_samples"],
            "distillation_index_lines": count_lines(distill_index),
            "distillation_teacher_samples": distill_summary["protocol"]["teacher_samples"],
        },
        "teacher_selection": {
            "directory_name": audit_root.name,
            "requested_confidence_levels": audit_summary["selected_confidence_levels"],
            "selected_count": int(len(selected)),
            "selected_confidence_counts": selected["confidence_level"].value_counts().sort_index().to_dict(),
            "candidate_beats_oracle_mse_count": int(selected["candidate_beats_oracle_mse"].sum()),
            "candidate_beats_oracle_mse_rate": float(selected["candidate_beats_oracle_mse"].mean()),
            "candidate_beats_oracle_by_5pct_count": int((selected["oracle_relative_improvement"] >= 0.05).sum()),
            "candidate_beats_oracle_by_5pct_rate": float((selected["oracle_relative_improvement"] >= 0.05).mean()),
            "gripper_full_match_rate": float((selected["teacher_gt_gripper_match_rate"] == 1.0).mean()),
            "gripper_at_least_75pct_match_rate": float((selected["teacher_gt_gripper_match_rate"] >= 0.75).mean()),
            "selected_tasks": int(selected["task_index"].nunique()),
            "selected_episodes": int(selected["episode_index"].nunique()),
            "selected_per_task_min": int(selected.groupby("task_index").size().min()),
            "selected_per_task_median": float(selected.groupby("task_index").size().median()),
            "selected_per_task_max": int(selected.groupby("task_index").size().max()),
            "selected_per_episode_max": int(selected.groupby("episode_index").size().max()),
        },
        "training": {
            "gt_datasets": dataset_entries(gt_root / "experiment_cfg/conf.yaml"),
            "flare_datasets": dataset_entries(flare_root / "experiment_cfg/conf.yaml"),
            "gt_model": model_dir_report(gt_root),
            "flare_model": model_dir_report(flare_root),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
