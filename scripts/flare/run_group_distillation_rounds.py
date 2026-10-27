#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run frozen-policy group-distillation rounds through existing pipeline commands.

Commands are argv arrays in a JSON plan, not shell strings. A dry run prints
the expanded commands without reading checkpoints or launching any process.
Actual runs use new round directories and fail rather than reuse stale outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.utils.checkpoint_snapshot import (  # noqa: E402
    checkpoint_snapshot_sha256,
    checkpoint_weights_sha256,
    file_sha256,
)


PLAN_FORMAT = "flare_group_round_plan_v1"
PROVENANCE_FORMAT = "flare_group_policy_provenance_v1"
_REQUIRED_STAGES = ("generate", "score", "prepare_teacher", "train")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _require_frozen_argument(argv: list[str], flag: str, value: str, stage: str) -> None:
    arguments = []
    for index, argument in enumerate(argv):
        if argument == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{stage} command has no value for {flag}")
            arguments.append(argv[index + 1])
        elif argument.startswith(f"{flag}="):
            arguments.append(argument.split("=", 1)[1])
    if arguments != [value]:
        raise ValueError(f"{stage} must pass exactly one {flag} with frozen {value}")


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError(f"Expected plan format {PLAN_FORMAT}")
    for field in ("initial_policy_checkpoint", "work_dir", "next_checkpoint"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            raise ValueError(f"Plan needs a nonempty {field}")
    for field, default in (("rounds", 2), ("start_round", 0)):
        value = plan.get(field, default)
        minimum = 1 if field == "rounds" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    commands = plan.get("commands")
    if not isinstance(commands, dict):
        raise ValueError("Plan commands must be an object of stage/argv-array lists")
    for stage in _REQUIRED_STAGES:
        if not commands.get(stage):
            raise ValueError(f"Plan needs nonempty {stage} commands")
    for stage, command_list in commands.items():
        if stage not in {*_REQUIRED_STAGES, "evaluate"} or not isinstance(command_list, list):
            raise ValueError(f"Unsupported command stage {stage}")
        for argv in command_list:
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                raise ValueError(f"{stage} commands must be nonempty arrays of strings")
    if not any("{policy_checkpoint}" in arg for argv in commands["generate"] for arg in argv):
        raise ValueError("Generation must reference the frozen {policy_checkpoint}")
    for stage in ("generate", "score"):
        for argv in commands[stage]:
            _require_frozen_argument(
                argv, "--policy-checkpoint-sha256", "{policy_checkpoint_sha256}", stage
            )
            policy_flag = "--model-path" if stage == "generate" else "--single-policy-checkpoint"
            _require_frozen_argument(argv, policy_flag, "{policy_checkpoint}", stage)
    if not any("{group_dataset_dir}" in arg for argv in commands["train"] for arg in argv):
        raise ValueError("Training must consume {group_dataset_dir}, not legacy top1 targets")
    if not any("{policy_checkpoint}" in arg for argv in commands["train"] for arg in argv):
        raise ValueError("Student training must initialize from {policy_checkpoint}")
    variables = plan.get("variables", {})
    if not isinstance(variables, dict) or not all(
        isinstance(v, (str, int, float)) for v in variables.values()
    ):
        raise ValueError("Plan variables must be scalar strings or numbers")
    builder_args = plan.get("group_builder_args", [])
    if not isinstance(builder_args, list) or not all(isinstance(v, str) for v in builder_args):
        raise ValueError("group_builder_args must be an argv array")
    reserved = {
        "--source-dir",
        "--distillation-dataset-dir",
        "--score-parts-dir",
        "--score-parts",
        "--output-dir",
        "--policy-provenance",
        "--round-id",
        "--dry-run",
    }
    if any(arg.split("=", 1)[0] in reserved for arg in builder_args):
        raise ValueError("group_builder_args may not override round-owned paths or provenance")


def _expand(argv: list[str], variables: dict[str, Any]) -> list[str]:
    try:
        return [arg.format_map(variables) for arg in argv]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid command placeholder: {exc}") from exc


def _commands(plan: dict[str, Any], stage: str, variables: dict[str, Any]) -> list[list[str]]:
    return [_expand(argv, variables) for argv in plan["commands"].get(stage, [])]


def _run_commands(commands: list[list[str]], *, cwd: Path, dry_run: bool) -> None:
    for argv in commands:
        print(shlex.join(argv), flush=True)
        if not dry_run:
            subprocess.run(argv, cwd=cwd, check=True)


def normalize_candidate_summary(variables: dict[str, Any]) -> None:
    """Adapt the existing full-manifest generator's summary path for the scorer."""
    summary = _read_json(Path(variables["generation_dir"]) / "candidate_generation_summary.json")
    if (
        summary.get("format") != "flare_candidate_generation_summary_v1"
        or summary.get("status") != "PASS"
    ):
        raise ValueError("Candidate generation did not produce a PASS summary")
    if summary.get("manifest_format") != "libero_plus_sparse_manifest_v1":
        raise ValueError("Use the full LIBERO-Plus manifest format for single-policy rounds")
    if Path(str(summary.get("model_path", ""))).resolve() != Path(variables["policy_checkpoint"]):
        raise ValueError("Generation summary does not match the frozen policy checkpoint")
    if summary.get("policy_checkpoint_sha256") != variables["policy_checkpoint_sha256"]:
        raise ValueError("Generation summary policy hash does not match the frozen checkpoint")
    summary["output_dir"] = variables["candidate_dir"]
    summary["round_id"] = int(variables["round_id"])
    _write_json(Path(variables["candidate_summary"]), summary)


def write_policy_provenance(variables: dict[str, Any]) -> dict[str, Any]:
    import torch

    records = []
    seen: set[str] = set()
    for path in sorted(Path(variables["score_parts_dir"]).glob("*.pt")):
        part = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(part, dict)
            or part.get("format") != "flare_lewm_libero_plus_score_part_v2"
        ):
            raise ValueError(f"Invalid score part: {path}")
        if part.get("policy_checkpoint_sha256") != variables["policy_checkpoint_sha256"]:
            raise ValueError(f"Score part policy hash does not match the frozen policy: {path}")
        sample_id = part.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate score-part sample_id: {path}")
        seen.add(sample_id)
        candidate_path = Path(str(part.get("candidate_path", ""))).resolve()
        if candidate_path.parent != Path(variables["candidate_dir"]):
            raise ValueError(f"Candidate is outside this round's generated directory: {sample_id}")
        if file_sha256(candidate_path) != part.get("candidate_sha256"):
            raise ValueError(f"Candidate content digest mismatch: {sample_id}")
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if not isinstance(candidate, dict) or Path(
            str(candidate.get("model_path", ""))
        ).resolve() != Path(variables["policy_checkpoint"]):
            raise ValueError(f"Candidate does not belong to the frozen policy: {sample_id}")
        if candidate.get("policy_checkpoint_sha256") != variables["policy_checkpoint_sha256"]:
            raise ValueError(f"Candidate policy hash does not match the frozen policy: {sample_id}")
        records.append({"sample_id": sample_id, "candidate_sha256": part["candidate_sha256"]})
    if not records:
        raise ValueError("No scored candidates found; refusing to create policy provenance")
    provenance = {
        "format": PROVENANCE_FORMAT,
        "round_id": int(variables["round_id"]),
        "policy_checkpoint_sha256": variables["policy_checkpoint_sha256"],
        "model_path": variables["policy_checkpoint"],
        "provenance_source": "fresh_frozen_policy_generation",
        "candidates": records,
    }
    _write_json(Path(variables["policy_provenance"]), provenance)
    return provenance


def run_rounds(plan: dict[str, Any], *, dry_run: bool = False) -> list[dict[str, Any]]:
    validate_plan(plan)
    if not dry_run and "REPLACE_WITH_" in json.dumps(plan):
        raise ValueError(
            "Fill the example plan's checkpoint/asset/calibration placeholders before execution"
        )
    cwd = Path(plan.get("cwd", REPO_ROOT)).expanduser().resolve()
    work_dir = Path(plan["work_dir"]).expanduser().resolve()
    policy = Path(plan["initial_policy_checkpoint"]).expanduser().resolve()
    plan_hash = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summaries: list[dict[str, Any]] = []
    first_round = plan.get("start_round", 0)
    for round_id in range(first_round, first_round + plan.get("rounds", 2)):
        round_dir = work_dir / f"round_{round_id:03d}"
        if not dry_run and round_dir.exists():
            raise FileExistsError(
                f"Round directory already exists; refusing stale reuse: {round_dir}"
            )
        policy_hash = "DRY_RUN_CHECKPOINT_HASH" if dry_run else checkpoint_snapshot_sha256(policy)
        policy_weights_hash = (
            "DRY_RUN_WEIGHTS_HASH" if dry_run else checkpoint_weights_sha256(policy)
        )
        values = {
            **plan.get("variables", {}),
            "repo_root": str(cwd),
            "round_id": round_id,
            "round_seed": int(plan.get("seed", 42)) + round_id,
            "round_dir": str(round_dir),
            "policy_checkpoint": str(policy),
            "policy_checkpoint_sha256": policy_hash,
            "generation_dir": str(round_dir / "generated"),
            "candidate_dir": str(round_dir / "generated" / "candidates"),
            "candidate_summary": str(round_dir / "candidate_summary_for_scoring.json"),
            "score_dir": str(round_dir / "scored"),
            "score_parts_dir": str(round_dir / "scored" / "score_parts"),
            "source_distillation_dir": str(round_dir / "top1_distillation"),
            "group_dataset_dir": str(round_dir / "group_distillation"),
            "training_output_dir": str(round_dir / "student"),
            "policy_provenance": str(round_dir / "policy_provenance.json"),
        }
        next_policy = Path(_expand([plan["next_checkpoint"]], values)[0]).expanduser().resolve()
        if not next_policy.is_relative_to(Path(values["training_output_dir"])):
            raise ValueError("next_checkpoint must be inside this round's new student directory")
        if next_policy == policy or policy.is_relative_to(Path(values["training_output_dir"])):
            raise ValueError("Student outputs may not overwrite the frozen policy checkpoint")
        values["next_policy_checkpoint"] = str(next_policy)
        build = [
            str(plan.get("group_python", sys.executable)),
            str(cwd / "scripts" / "flare" / "build_group_teacher.py"),
            "--source-dir",
            values["source_distillation_dir"],
            "--score-parts-dir",
            values["score_parts_dir"],
            "--output-dir",
            values["group_dataset_dir"],
            "--policy-provenance",
            values["policy_provenance"],
            "--round-id",
            str(round_id),
            *_expand(plan.get("group_builder_args", []), values),
        ]
        expanded = {stage: _commands(plan, stage, values) for stage in plan["commands"]}
        record = {
            "format": "flare_group_round_status_v1",
            "round_id": round_id,
            "policy_checkpoint": str(policy),
            "policy_checkpoint_sha256": policy_hash,
            "policy_weights_sha256": policy_weights_hash,
            "next_policy_checkpoint": str(next_policy),
            "plan_sha256": plan_hash,
            "round_seed": values["round_seed"],
            "status": "DRY_RUN" if dry_run else "RUNNING",
        }
        status_path = round_dir / "round_status.json"
        if not dry_run:
            round_dir.mkdir(parents=True)
            _write_json(status_path, record)
        try:
            _run_commands(expanded["generate"], cwd=cwd, dry_run=dry_run)
            if not dry_run:
                if checkpoint_snapshot_sha256(policy) != policy_hash:
                    raise ValueError("Behavior checkpoint changed during candidate generation")
                normalize_candidate_summary(values)
            _run_commands(expanded["score"], cwd=cwd, dry_run=dry_run)
            if not dry_run:
                write_policy_provenance(values)
            _run_commands(expanded["prepare_teacher"], cwd=cwd, dry_run=dry_run)
            _run_commands([build], cwd=cwd, dry_run=dry_run)
            if not dry_run and checkpoint_snapshot_sha256(policy) != policy_hash:
                raise ValueError("Behavior checkpoint changed before student training")
            _run_commands(expanded["train"], cwd=cwd, dry_run=dry_run)
            if not dry_run:
                if checkpoint_snapshot_sha256(policy) != policy_hash:
                    raise ValueError("Student training modified the frozen behavior checkpoint")
                next_hash = checkpoint_snapshot_sha256(next_policy)
                next_weights_hash = checkpoint_weights_sha256(next_policy)
                if next_weights_hash == policy_weights_hash:
                    raise ValueError(
                        "Student checkpoint weights did not change; refusing fake self-distillation refresh"
                    )
                record["next_policy_checkpoint_sha256"] = next_hash
                record["next_policy_weights_sha256"] = next_weights_hash
            _run_commands(expanded.get("evaluate", []), cwd=cwd, dry_run=dry_run)
            if not dry_run:
                record["status"] = "PASS"
                _write_json(status_path, record)
        except Exception as exc:
            if not dry_run:
                record["status"] = "FAILED"
                record["error_type"] = type(exc).__name__
                _write_json(status_path, record)
            raise
        summaries.append(record)
        policy = next_policy
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summaries = run_rounds(_read_json(args.plan), dry_run=args.dry_run)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
