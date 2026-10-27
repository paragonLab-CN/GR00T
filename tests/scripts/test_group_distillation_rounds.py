# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only snapshot, round-plan, provenance, and iterative-runner regressions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from gr00t.utils.checkpoint_snapshot import (
    checkpoint_snapshot_sha256,
    checkpoint_weights_sha256,
    file_sha256,
)
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "flare" / "run_group_distillation_rounds.py"
BUILDER_PATH = REPO_ROOT / "scripts" / "flare" / "build_group_teacher.py"
MANIFEST_FORMAT = "libero_plus_sparse_manifest_v1"
WM_HASH = "a" * 64


def _script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return _script(RUNNER_PATH, "group_round_runner_test")


def _json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(path: Path, value: float = 0.1) -> Path:
    path.mkdir(parents=True)
    torch.save({"parameter": torch.tensor(value)}, path / "model.pt")
    _json(path / "config.json", {"model_type": "synthetic_cpu_policy", "action_dim": 7})
    return path


def _identities():
    return [
        {"sample_id": "sample_002", "sample_index": 2, "episode_index": 2, "local_step": 3},
        {"sample_id": "sample_001", "sample_index": 1, "episode_index": 1, "local_step": 5},
    ]


def _plan(tmp_path: Path):
    return {
        "format": "flare_group_round_plan_v1",
        "initial_policy_checkpoint": str(tmp_path / "initial_policy"),
        "work_dir": str(tmp_path / "rounds"),
        "cwd": str(REPO_ROOT),
        "rounds": 2,
        "seed": 42,
        "next_checkpoint": "{training_output_dir}/checkpoint-1",
        "variables": {"manifest": str(tmp_path / "manifest.jsonl")},
        "group_builder_args": ["--oracle-max-mse", "0.05", "--max-paired-gap", "0.02"],
        "commands": {
            "generate": [
                [
                    "synthetic-stage",
                    "generate",
                    "--round-id",
                    "{round_id}",
                    "--seed",
                    "{round_seed}",
                    "--manifest",
                    "{manifest}",
                    "--model-path",
                    "{policy_checkpoint}",
                    "--policy-checkpoint-sha256",
                    "{policy_checkpoint_sha256}",
                    "--output-dir",
                    "{generation_dir}",
                ]
            ],
            "score": [
                [
                    "synthetic-stage",
                    "score",
                    "--single-policy-checkpoint",
                    "{policy_checkpoint}",
                    "--policy-checkpoint-sha256",
                    "{policy_checkpoint_sha256}",
                    "--candidates-dir",
                    "{candidate_dir}",
                    "--candidate-summary",
                    "{candidate_summary}",
                    "--output-dir",
                    "{score_dir}",
                ]
            ],
            "prepare_teacher": [
                [
                    "synthetic-stage",
                    "prepare_teacher",
                    "--score-parts-dir",
                    "{score_parts_dir}",
                    "--output-dir",
                    "{source_distillation_dir}",
                ]
            ],
            "train": [
                [
                    "synthetic-stage",
                    "train",
                    "--base-model-path",
                    "{policy_checkpoint}",
                    "--distillation-dataset-dir",
                    "{group_dataset_dir}",
                    "--output-dir",
                    "{training_output_dir}",
                    "--next-checkpoint",
                    "{next_policy_checkpoint}",
                ]
            ],
            "evaluate": [
                [
                    "synthetic-stage",
                    "evaluate",
                    "--model-path",
                    "{next_policy_checkpoint}",
                ]
            ],
        },
    }


def _argument(argv, flag):
    return argv[argv.index(flag) + 1]


class _SyntheticPipeline:
    """Fake expensive stages while exercising the real builder CLI and SGD."""

    def __init__(self, monkeypatch, plan):
        self.monkeypatch = monkeypatch
        self.plan = plan
        self.builder = _script(BUILDER_PATH, "group_round_builder_test")
        self.calls = []
        self.fail_stage = None
        self.mutate_policy_stage = None
        self.unchanged_student = False
        self.student_nonweight_change = None
        self.missing_student = False
        self.corrupt_candidates = None
        self.corrupt_score_parts = None
        self.corrupt_summary_hash = None

    def __call__(self, argv, *, cwd, check):
        assert check is True
        assert Path(cwd) == Path(self.plan["cwd"])
        if argv[0] == "synthetic-stage":
            stage = argv[1]
        else:
            assert Path(argv[1]) == BUILDER_PATH
            stage = "build_group"
        self.calls.append((stage, list(argv)))
        if stage == self.fail_stage:
            raise subprocess.CalledProcessError(13, argv)
        if stage == "build_group":
            with self.monkeypatch.context() as patch:
                patch.setattr(sys, "argv", argv[1:])
                self.builder.main()
        else:
            getattr(self, stage)(argv)
        if stage == self.mutate_policy_stage:
            policy_flag = "--model-path" if stage == "generate" else "--base-model-path"
            policy = Path(_argument(argv, policy_flag))
            torch.save({"parameter": torch.tensor(99.0)}, policy / "model.pt")
        return subprocess.CompletedProcess(argv, 0)

    def generate(self, argv):
        policy = Path(_argument(argv, "--model-path"))
        policy_hash = _argument(argv, "--policy-checkpoint-sha256")
        manifest = Path(_argument(argv, "--manifest"))
        generation = Path(_argument(argv, "--output-dir"))
        candidates = generation / "candidates"
        candidates.mkdir(parents=True)
        parameter = torch.load(policy / "model.pt", weights_only=True)["parameter"]
        for identity in _identities():
            offsets = torch.tensor([0.05, 0.15, 0.25])[:, None, None]
            actions = (parameter + offsets).expand(3, 4, 7).clone()
            payload = {
                **identity,
                "format": "flare_groot_manifest_candidates_v1",
                "manifest_format": MANIFEST_FORMAT,
                "manifest_sha256": file_sha256(manifest),
                "model_path": str(policy),
                "policy_checkpoint_sha256": policy_hash,
                "candidate_actions_concat": actions.unsqueeze(0),
            }
            if self.corrupt_candidates == "cached_model":
                payload["model_path"] = str(policy.parent / "old_cached_policy")
            elif self.corrupt_candidates == "stale_policy_hash":
                payload["policy_checkpoint_sha256"] = "0" * 64
            elif self.corrupt_candidates == "missing_policy_hash":
                del payload["policy_checkpoint_sha256"]
            torch.save(payload, candidates / f"{identity['sample_id']}.pt")
        _json(
            generation / "candidate_generation_summary.json",
            {
                "format": "flare_candidate_generation_summary_v1",
                "status": "PASS",
                "manifest_format": MANIFEST_FORMAT,
                "manifest": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "model_path": str(policy),
                "policy_checkpoint_sha256": policy_hash,
                "output_dir": str(generation),
                "counters": {"selected": 2, "generated": 2, "skipped_valid": 0, "failed": 0},
            },
        )
        if self.corrupt_summary_hash:
            summary_path = generation / "candidate_generation_summary.json"
            summary = _read_json(summary_path)
            if self.corrupt_summary_hash == "missing":
                del summary["policy_checkpoint_sha256"]
            else:
                summary["policy_checkpoint_sha256"] = "0" * 64
            _json(summary_path, summary)

    def score(self, argv):
        candidates = Path(_argument(argv, "--candidates-dir"))
        summary = _read_json(Path(_argument(argv, "--candidate-summary")))
        assert Path(summary["output_dir"]) == candidates
        assert summary["policy_checkpoint_sha256"] == _argument(argv, "--policy-checkpoint-sha256")
        assert summary["model_path"] == _argument(argv, "--single-policy-checkpoint")
        parts = Path(_argument(argv, "--output-dir")) / "score_parts"
        parts.mkdir(parents=True)
        for candidate_path in sorted(candidates.glob("*.pt")):
            candidate = torch.load(candidate_path, weights_only=False)
            part = {
                **{name: candidate[name] for name in _identities()[0]},
                "format": "flare_lewm_libero_plus_score_part_v2",
                "manifest_sha256": candidate["manifest_sha256"],
                "checkpoint_sha256": WM_HASH,
                "policy_checkpoint_sha256": summary["policy_checkpoint_sha256"],
                "round_id": summary["round_id"],
                "candidate_path": str(candidate_path),
                "candidate_sha256": file_sha256(candidate_path),
                "candidate_k": 3,
                "future_horizon": 4,
                "candidate_actions_lerobot": candidate["candidate_actions_concat"][0],
                "mse_per_step": torch.full((3, 4), 0.02),
                "oracle_mse_per_step": torch.full((4,), 0.01),
                "cosine_score": torch.tensor([0.2, 0.5, 0.9]),
                "cosine_per_step": torch.tensor([[0.2] * 4, [0.5] * 4, [0.9] * 4]),
                "mse_score": torch.tensor([0.02, 0.02, 0.02]),
            }
            if self.corrupt_score_parts == "candidate_hash":
                part["candidate_sha256"] = "0" * 64
            elif self.corrupt_score_parts == "scored_actions":
                part["candidate_actions_lerobot"] = part["candidate_actions_lerobot"].clone()
                part["candidate_actions_lerobot"][1] += 0.01
            elif self.corrupt_score_parts == "policy_hash":
                part["policy_checkpoint_sha256"] = "0" * 64
            elif self.corrupt_score_parts == "missing_policy_hash":
                del part["policy_checkpoint_sha256"]
            torch.save(part, parts / candidate_path.name)

    def prepare_teacher(self, argv):
        parts = Path(_argument(argv, "--score-parts-dir"))
        source = Path(_argument(argv, "--output-dir"))
        source.mkdir(parents=True)
        identities = _identities()
        score_parts = [
            torch.load(parts / f"{row['sample_id']}.pt", weights_only=False) for row in identities
        ]
        teacher = {
            "format": "flare_distillation_teacher_records_v1",
            "manifest_sha256": score_parts[0]["manifest_sha256"],
            "sample_ids": [row["sample_id"] for row in identities],
            "sample_indices": torch.tensor([row["sample_index"] for row in identities]),
            "episode_indices": torch.tensor([row["episode_index"] for row in identities]),
            "local_steps": torch.tensor([row["local_step"] for row in identities]),
            "teacher_actions_lerobot": torch.stack(
                [part["candidate_actions_lerobot"][2] for part in score_parts]
            ),
            "teacher_idx_primary": torch.full((2,), 2, dtype=torch.long),
            "teacher_horizon": 4,
            "processor_action_horizon": 16,
            "model_action_horizon": 40,
            "action_dim": 7,
            "action_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        }
        torch.save(teacher, source / "teacher_records_enriched.pt")
        index = []
        for source_name in ("gt", "teacher"):
            for teacher_row, identity in enumerate(identities):
                index.append(
                    {
                        **identity,
                        "format": "flare_distillation_index_v1",
                        "distillation_index": len(index),
                        "source": source_name,
                        "teacher_row": teacher_row if source_name == "teacher" else None,
                    }
                )
        (source / "distillation_index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in index), encoding="utf-8"
        )
        _json(
            source / "distillation_summary.json",
            {
                "format": "flare_distillation_summary_v1",
                "status": "PASS",
                "protocol": {"total_samples": 4, "gt_samples": 2, "teacher_samples": 2},
            },
        )

    def train(self, argv):
        policy = Path(_argument(argv, "--base-model-path"))
        next_policy = Path(_argument(argv, "--next-checkpoint"))
        group_dir = Path(_argument(argv, "--distillation-dataset-dir"))
        if self.missing_student:
            return
        if self.unchanged_student or self.student_nonweight_change:
            shutil.copytree(policy, next_policy)
            if self.student_nonweight_change:
                changed = next_policy / self.student_nonweight_change
                changed.parent.mkdir(parents=True, exist_ok=True)
                _json(changed, {"synthetic_student_only_change": True})
            return
        group = torch.load(group_dir / "teacher_records_enriched.pt", weights_only=False)[
            "group_teacher"
        ]
        assert group["policy_checkpoint_sha256"] == checkpoint_snapshot_sha256(policy)
        parameter = torch.nn.Parameter(
            torch.load(policy / "model.pt", weights_only=True)["parameter"].clone()
        )
        candidate_targets = group["candidate_actions_lerobot"].mean(dim=(2, 3))
        teacher_target = (candidate_targets * group["candidate_probabilities"]).sum(1).mean()
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        loss = 0.7 * (parameter - 0.1).square() + 0.3 * (parameter - teacher_target).square()
        loss.backward()
        optimizer.step()
        next_policy.mkdir(parents=True)
        torch.save({"parameter": parameter.detach()}, next_policy / "model.pt")
        shutil.copy2(policy / "config.json", next_policy / "config.json")
        torch.save(optimizer.state_dict(), next_policy / "optimizer.pt")

    def evaluate(self, argv):
        assert (Path(_argument(argv, "--model-path")) / "model.pt").is_file()


@pytest.fixture
def pipeline(tmp_path, monkeypatch, runner):
    plan = _plan(tmp_path)
    _checkpoint(Path(plan["initial_policy_checkpoint"]))
    Path(plan["variables"]["manifest"]).write_text(
        "".join(json.dumps(row) + "\n" for row in _identities()), encoding="utf-8"
    )
    stages = _SyntheticPipeline(monkeypatch, plan)
    monkeypatch.setattr(runner.subprocess, "run", stages)
    return runner, plan, stages


@pytest.mark.parametrize("identity", [checkpoint_snapshot_sha256, checkpoint_weights_sha256])
def test_snapshot_identity_is_independent_of_checkpoint_root(tmp_path, identity):
    original = _checkpoint(tmp_path / "original")
    moved = tmp_path / "moved" / "copy"
    shutil.copytree(original, moved)

    assert identity(original) == identity(moved)


@pytest.mark.parametrize(
    "name",
    [
        "model.pt",
        "config.json",
        "tokenizer.json",
        "merges.txt",
        "processor_config.json",
        "statistics.json",
        "model.safetensors.index.json",
        "processor/processor_config.json",
        "processor/statistics.json",
        "processor/embodiment_config.json",
        "processor/nested/custom_action_stats.json",
    ],
)
def test_snapshot_changes_for_weights_or_inference_configuration(tmp_path, name):
    path = _checkpoint(tmp_path / "policy")
    before = checkpoint_snapshot_sha256(path)
    changed = path / name
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_bytes(b"changed checkpoint content")

    assert checkpoint_snapshot_sha256(path) != before


@pytest.mark.parametrize("identity", [checkpoint_snapshot_sha256, checkpoint_weights_sha256])
def test_optimizer_and_trainer_state_do_not_change_snapshot_identity(tmp_path, identity):
    path = _checkpoint(tmp_path / "policy")
    before = identity(path)
    torch.save({"step": 123}, path / "optimizer.pt")
    (path / "training_args.bin").write_bytes(b"training-only arguments")
    _json(path / "trainer_state.json", {"global_step": 123})

    assert identity(path) == before
    torch.save({"step": 456}, path / "optimizer.pt")
    _json(path / "trainer_state.json", {"global_step": 456})
    assert identity(path) == before


@pytest.mark.parametrize(
    "name",
    [
        "wandb_config.json",
        "metadata.json",
        "experiment_cfg/metadata.json",
        "experiment_cfg/config.json",
        "experiment_cfg/processor_config.json",
        "experiment_cfg/processor/statistics.json",
        "logs/config.json",
    ],
)
@pytest.mark.parametrize("identity", [checkpoint_snapshot_sha256, checkpoint_weights_sha256])
def test_training_metadata_does_not_change_checkpoint_identity(tmp_path, name, identity):
    path = _checkpoint(tmp_path / "policy")
    before = identity(path)
    metadata = path / name
    metadata.parent.mkdir(parents=True, exist_ok=True)
    _json(metadata, {"global_step": 123})

    assert identity(path) == before
    _json(metadata, {"global_step": 456})
    assert identity(path) == before


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "tokenizer.json",
        "merges.txt",
        "processor_config.json",
        "statistics.json",
        "model.safetensors.index.json",
        "processor/processor_config.json",
        "processor/statistics.json",
        "processor/custom_action_stats.json",
    ],
)
def test_weights_identity_is_unchanged_by_inference_configuration(tmp_path, name):
    path = _checkpoint(tmp_path / "policy")
    before = checkpoint_weights_sha256(path)
    config = path / name
    config.parent.mkdir(parents=True, exist_ok=True)
    _json(config, {"changed_inference_configuration": True})

    assert checkpoint_weights_sha256(path) == before
    _json(config, {"changed_inference_configuration": False})
    assert checkpoint_weights_sha256(path) == before


@pytest.mark.parametrize(
    "name", ["model.pt", "weights.pt", "pytorch_model.bin", "model.safetensors"]
)
def test_weights_identity_changes_when_model_weight_contents_change(tmp_path, name):
    path = _checkpoint(tmp_path / "policy")
    weights = path / name
    weights.write_bytes(b"initial model weight contents")
    before = checkpoint_weights_sha256(path)
    weights.write_bytes(b"updated model weight contents")

    assert checkpoint_weights_sha256(path) != before


@pytest.mark.parametrize("identity", [checkpoint_snapshot_sha256, checkpoint_weights_sha256])
def test_snapshot_rejects_missing_checkpoint_and_directory_without_model_weights(
    tmp_path, identity
):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        identity(tmp_path / "missing")
    config_only = tmp_path / "config_only"
    config_only.mkdir()
    _json(config_only / "config.json", {"model_type": "test"})
    torch.save({"step": 1}, config_only / "optimizer.pt")
    with pytest.raises(ValueError, match="no supported model weights"):
        identity(config_only)


@pytest.mark.parametrize("identity", [checkpoint_snapshot_sha256, checkpoint_weights_sha256])
def test_single_file_snapshot_matches_content_digest(tmp_path, identity):
    path = tmp_path / "model.pt"
    path.write_bytes(b"a frozen checkpoint")

    assert identity(path) == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "wrong_format"),
        ("initial_policy_checkpoint", ""),
        ("work_dir", None),
        ("next_checkpoint", 3),
        ("rounds", 0),
        ("rounds", True),
        ("rounds", 1.5),
        ("start_round", -1),
        ("start_round", False),
        ("commands", None),
        ("variables", []),
        ("variables", {"bad_value": []}),
        ("group_builder_args", "--temperature 1"),
        ("group_builder_args", [1]),
    ],
)
def test_invalid_plan_fields_fail_before_execution(runner, tmp_path, field, value):
    plan = _plan(tmp_path)
    plan[field] = value
    with pytest.raises(ValueError):
        runner.validate_plan(plan)


@pytest.mark.parametrize(
    "case", ["missing_stage", "unknown_stage", "shell_string", "empty_argv", "numeric_arg"]
)
def test_invalid_command_structure_is_rejected(runner, tmp_path, case):
    plan = _plan(tmp_path)
    if case == "missing_stage":
        del plan["commands"]["train"]
    elif case == "unknown_stage":
        plan["commands"]["submit"] = [["unsafe-command"]]
    elif case == "shell_string":
        plan["commands"]["train"] = ["python train.py && echo done"]
    elif case == "empty_argv":
        plan["commands"]["train"] = [[]]
    else:
        plan["commands"]["train"] = [["python", 1]]
    with pytest.raises(ValueError):
        runner.validate_plan(plan)


@pytest.mark.parametrize(
    "stage,placeholder",
    [
        ("generate", "{policy_checkpoint}"),
        ("train", "{policy_checkpoint}"),
        ("train", "{group_dataset_dir}"),
    ],
)
def test_required_policy_and_group_placeholders_cannot_be_omitted(
    runner, tmp_path, stage, placeholder
):
    plan = _plan(tmp_path)
    plan["commands"][stage] = [
        [argument.replace(placeholder, "fixed-stale-path") for argument in argv]
        for argv in plan["commands"][stage]
    ]
    with pytest.raises(ValueError):
        runner.validate_plan(plan)


@pytest.mark.parametrize(
    "stage,flag,placeholder",
    [
        ("generate", "--model-path", "{policy_checkpoint}"),
        ("generate", "--policy-checkpoint-sha256", "{policy_checkpoint_sha256}"),
        ("score", "--policy-checkpoint-sha256", "{policy_checkpoint_sha256}"),
        ("score", "--single-policy-checkpoint", "{policy_checkpoint}"),
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "stale_value",
        "unrelated_placeholder",
        "duplicate_good",
        "duplicate_bad",
        "duplicate_equals",
    ],
)
def test_fresh_policy_flags_must_bind_to_the_current_round_placeholders(
    runner, tmp_path, stage, flag, placeholder, case
):
    plan = _plan(tmp_path)
    argv = plan["commands"][stage][0]
    index = argv.index(flag)
    if case == "missing":
        del argv[index : index + 2]
    elif case.startswith("duplicate"):
        if case == "duplicate_equals":
            argv.append(f"{flag}={placeholder}")
        else:
            argv.extend([flag, placeholder if case == "duplicate_good" else "stale-fixed-value"])
    else:
        argv[index + 1] = "stale-fixed-value"
        if case == "unrelated_placeholder":
            argv.extend(["--unrelated-option", placeholder])

    with pytest.raises(ValueError):
        runner.validate_plan(plan)


def test_fresh_policy_flags_support_equals_syntax_without_side_effects(runner, tmp_path):
    plan = _plan(tmp_path)
    flags = {
        "generate": ("--model-path", "--policy-checkpoint-sha256"),
        "score": ("--single-policy-checkpoint", "--policy-checkpoint-sha256"),
    }
    for stage, names in flags.items():
        argv = plan["commands"][stage][0]
        for flag in names:
            index = argv.index(flag)
            argv[index : index + 2] = [f"{flag}={argv[index + 1]}"]

    runner.validate_plan(plan)
    records = runner.run_rounds(plan, dry_run=True)

    assert len(records) == 2
    assert not Path(plan["work_dir"]).exists()


@pytest.mark.parametrize(
    "flag",
    [
        "--source-dir",
        "--distillation-dataset-dir",
        "--score-parts-dir",
        "--score-parts",
        "--output-dir",
        "--policy-provenance",
        "--round-id",
        "--dry-run",
    ],
)
@pytest.mark.parametrize("equals_syntax", [False, True])
def test_group_builder_arguments_cannot_override_owned_paths_or_provenance(
    runner, tmp_path, flag, equals_syntax
):
    plan = _plan(tmp_path)
    plan["group_builder_args"].append(f"{flag}=override" if equals_syntax else flag)
    with pytest.raises(ValueError, match="override"):
        runner.validate_plan(plan)


@pytest.mark.parametrize("command", [["{missing}"], ["unfinished {"]])
def test_unknown_or_invalid_command_placeholders_are_actionable_errors(runner, command):
    with pytest.raises(ValueError, match="Invalid command placeholder"):
        runner._expand(command, {})


@pytest.mark.parametrize(
    "next_path", ["/outside/checkpoint-1", "{training_output_dir}/../stale/checkpoint-1"]
)
def test_next_policy_must_be_owned_by_the_new_round_student_directory(runner, tmp_path, next_path):
    plan = _plan(tmp_path)
    plan["next_checkpoint"] = next_path
    with pytest.raises(ValueError, match="new student directory"):
        runner.run_rounds(plan, dry_run=True)
    assert not Path(plan["work_dir"]).exists()


def test_next_checkpoint_placeholder_errors_fail_before_creating_outputs(runner, tmp_path):
    plan = _plan(tmp_path)
    plan["next_checkpoint"] = "{unknown_student_dir}/checkpoint-1"
    with pytest.raises(ValueError, match="Invalid command placeholder"):
        runner.run_rounds(plan, dry_run=True)
    assert not Path(plan["work_dir"]).exists()


def test_round_owned_student_outputs_cannot_overwrite_frozen_policy(runner, tmp_path):
    plan = _plan(tmp_path)
    plan["initial_policy_checkpoint"] = str(
        Path(plan["work_dir"]) / "round_000" / "student" / "initial_policy"
    )
    with pytest.raises(ValueError, match="overwrite the frozen policy"):
        runner.run_rounds(plan, dry_run=True)
    assert not Path(plan["work_dir"]).exists()


def test_builder_comes_from_explicit_command_checkout(runner, tmp_path, capsys):
    plan = _plan(tmp_path)
    checkout = tmp_path / "another_checkout"
    plan["cwd"] = str(checkout)

    runner.run_rounds(plan, dry_run=True)

    printed = capsys.readouterr().out
    assert str(checkout / "scripts" / "flare" / "build_group_teacher.py") in printed
    assert str(BUILDER_PATH) not in printed
    assert not checkout.exists()


@pytest.mark.parametrize(
    "location",
    [
        "variable",
        "initial_policy_checkpoint",
        "work_dir",
        "cwd",
        "next_checkpoint",
        "command_argument",
        "group_builder_argument",
    ],
)
def test_unfilled_example_asset_values_are_rejected_before_checkpoint_loading(
    runner, tmp_path, monkeypatch, location
):
    plan = _plan(tmp_path)
    if location == "variable":
        plan["variables"]["asset"] = "REPLACE_WITH_DATASET_PATH"
    elif location == "command_argument":
        plan["commands"]["train"][0].extend(["--run-name", "REPLACE_WITH_RUN_NAME"])
    elif location == "group_builder_argument":
        plan["group_builder_args"].extend(["--temperature", "REPLACE_WITH_TEMPERATURE"])
    else:
        plan[location] = str(tmp_path / "REPLACE_WITH_LIBERO_FINETUNED_STUDENT")

    def forbidden(*args, **kwargs):
        raise AssertionError("unfilled example plan started checkpoint loading or execution")

    monkeypatch.setattr(runner, "checkpoint_snapshot_sha256", forbidden)
    monkeypatch.setattr(runner, "checkpoint_weights_sha256", forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="placeholders"):
        runner.run_rounds(plan)
    assert set(tmp_path.rglob("*")) == before


def test_dry_run_never_loads_checkpoints_writes_or_launches_subprocesses(
    runner, tmp_path, monkeypatch, capsys
):
    plan = _plan(tmp_path)
    plan["start_round"] = 2
    plan["variables"].update(
        {"policy_checkpoint": "/stale_override", "group_dataset_dir": "/stale_targets"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run attempted an execution side effect")

    for name in (
        "checkpoint_snapshot_sha256",
        "checkpoint_weights_sha256",
        "normalize_candidate_summary",
        "write_policy_provenance",
        "_write_json",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    before = set(tmp_path.rglob("*"))

    records = runner.run_rounds(plan, dry_run=True)

    assert set(tmp_path.rglob("*")) == before
    assert [record["status"] for record in records] == ["DRY_RUN", "DRY_RUN"]
    assert [record["round_id"] for record in records] == [2, 3]
    assert [record["round_seed"] for record in records] == [44, 45]
    assert records[1]["policy_checkpoint"] == records[0]["next_policy_checkpoint"]
    printed = capsys.readouterr().out
    assert "DRY_RUN_CHECKPOINT_HASH" in printed
    assert "/stale_override" not in printed
    assert "/stale_targets" not in printed


def test_two_cpu_rounds_refresh_policy_and_preserve_real_group_sidecars(pipeline):
    runner, plan, stages = pipeline
    initial_hash = checkpoint_snapshot_sha256(plan["initial_policy_checkpoint"])
    initial_weights_hash = checkpoint_weights_sha256(plan["initial_policy_checkpoint"])

    records = runner.run_rounds(plan)

    assert [record["status"] for record in records] == ["PASS", "PASS"]
    assert records[0]["policy_checkpoint_sha256"] == initial_hash
    assert records[0]["policy_weights_sha256"] == initial_weights_hash
    assert records[1]["policy_checkpoint"] == records[0]["next_policy_checkpoint"]
    assert records[1]["policy_checkpoint_sha256"] == records[0]["next_policy_checkpoint_sha256"]
    assert records[1]["policy_weights_sha256"] == records[0]["next_policy_weights_sha256"]
    assert records[1]["next_policy_checkpoint_sha256"] != initial_hash
    assert records[1]["next_policy_weights_sha256"] != initial_weights_hash
    stages_per_round = ["generate", "score", "prepare_teacher", "build_group", "train", "evaluate"]
    assert [stage for stage, _ in stages.calls] == stages_per_round * 2
    expected_plan_hash = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    old_parameter = torch.load(
        Path(plan["initial_policy_checkpoint"]) / "model.pt", weights_only=True
    )["parameter"]
    for record in records:
        round_dir = Path(plan["work_dir"]) / f"round_{record['round_id']:03d}"
        assert _read_json(round_dir / "round_status.json") == record
        assert record["plan_sha256"] == expected_plan_hash
        assert record["policy_weights_sha256"] == checkpoint_weights_sha256(
            record["policy_checkpoint"]
        )
        assert record["next_policy_weights_sha256"] == checkpoint_weights_sha256(
            record["next_policy_checkpoint"]
        )
        assert record["next_policy_weights_sha256"] != record["policy_weights_sha256"]
        source = round_dir / "top1_distillation"
        output = round_dir / "group_distillation"
        assert (source / "distillation_index.jsonl").read_bytes() == (
            output / "distillation_index.jsonl"
        ).read_bytes()
        original = torch.load(source / "teacher_records_enriched.pt", weights_only=False)
        enriched = torch.load(output / "teacher_records_enriched.pt", weights_only=False)
        group = enriched["group_teacher"]
        assert group["sample_ids"] == [row["sample_id"] for row in _identities()]
        assert group["round_id"] == record["round_id"]
        assert group["policy_checkpoint_sha256"] == record["policy_checkpoint_sha256"]
        assert group["world_model_checkpoint_sha256"] == WM_HASH
        assert group["candidate_actions_lerobot"].shape == (2, 3, 4, 7)
        assert group["prefix_lengths"].tolist() == [[4, 4, 4], [4, 4, 4]]
        assert torch.all(
            group["candidate_probabilities"][:, 2] > group["candidate_probabilities"][:, 0]
        )
        torch.testing.assert_close(group["candidate_probabilities"].sum(1), torch.ones(2))
        assert torch.equal(enriched["teacher_actions_lerobot"], original["teacher_actions_lerobot"])
        provenance = _read_json(round_dir / "policy_provenance.json")
        hashes = {row["sample_id"]: row["candidate_sha256"] for row in provenance["candidates"]}
        assert group["candidate_sha256"] == [hashes[sample_id] for sample_id in group["sample_ids"]]
        parameter = torch.load(
            Path(record["next_policy_checkpoint"]) / "model.pt", weights_only=True
        )["parameter"]
        assert parameter.item() > old_parameter.item()
        old_parameter = parameter


@pytest.mark.parametrize(
    "stage", ["generate", "score", "prepare_teacher", "build_group", "train", "evaluate"]
)
def test_any_failed_stage_records_failure_and_never_starts_the_next_round(pipeline, stage):
    runner, plan, stages = pipeline
    stages.fail_stage = stage

    with pytest.raises(subprocess.CalledProcessError):
        runner.run_rounds(plan)

    round_dir = Path(plan["work_dir"]) / "round_000"
    status = _read_json(round_dir / "round_status.json")
    assert status["status"] == "FAILED"
    assert status["error_type"] == "CalledProcessError"
    assert stages.calls[-1][0] == stage
    assert not (Path(plan["work_dir"]) / "round_001").exists()


@pytest.mark.parametrize(
    "stage,message",
    [
        ("generate", "changed during candidate generation"),
        ("train", "modified the frozen behavior checkpoint"),
    ],
)
def test_behavior_checkpoint_is_immutable_during_generation_and_training(pipeline, stage, message):
    runner, plan, stages = pipeline
    stages.mutate_policy_stage = stage

    with pytest.raises(ValueError, match=message):
        runner.run_rounds(plan)

    status = _read_json(Path(plan["work_dir"]) / "round_000" / "round_status.json")
    assert status["status"] == "FAILED"
    assert stages.calls[-1][0] == stage
    assert not (Path(plan["work_dir"]) / "round_001").exists()


@pytest.mark.parametrize("case", ["unchanged", "missing"])
def test_missing_or_identical_student_cannot_count_as_self_distillation_refresh(pipeline, case):
    runner, plan, stages = pipeline
    stages.unchanged_student = case == "unchanged"
    stages.missing_student = case == "missing"

    with pytest.raises(ValueError if case == "unchanged" else FileNotFoundError):
        runner.run_rounds(plan)

    status = _read_json(Path(plan["work_dir"]) / "round_000" / "round_status.json")
    assert status["status"] == "FAILED"
    assert stages.calls[-1][0] == "train"
    assert not (Path(plan["work_dir"]) / "round_001").exists()


@pytest.mark.parametrize(
    "name",
    [
        "wandb_config.json",
        "metadata.json",
        "experiment_cfg/metadata.json",
        "config.json",
        "processor_config.json",
        "statistics.json",
        "processor/processor_config.json",
    ],
)
def test_nonweight_student_changes_cannot_count_as_self_distillation_refresh(pipeline, name):
    runner, plan, stages = pipeline
    stages.student_nonweight_change = name
    initial_policy = Path(plan["initial_policy_checkpoint"])

    with pytest.raises(ValueError, match="did not change"):
        runner.run_rounds(plan)

    round_dir = Path(plan["work_dir"]) / "round_000"
    status = _read_json(round_dir / "round_status.json")
    next_policy = Path(status["next_policy_checkpoint"])
    assert (next_policy / name).is_file()
    assert status["status"] == "FAILED"
    assert status["error_type"] == "ValueError"
    assert status["policy_weights_sha256"] == checkpoint_weights_sha256(initial_policy)
    assert checkpoint_weights_sha256(next_policy) == checkpoint_weights_sha256(initial_policy)
    if name in {"wandb_config.json", "metadata.json", "experiment_cfg/metadata.json"}:
        assert checkpoint_snapshot_sha256(next_policy) == checkpoint_snapshot_sha256(initial_policy)
    else:
        assert checkpoint_snapshot_sha256(next_policy) != checkpoint_snapshot_sha256(initial_policy)
    assert stages.calls[-1][0] == "train"
    assert not any(stage == "evaluate" for stage, _ in stages.calls)
    assert not (Path(plan["work_dir"]) / "round_001").exists()


def test_existing_round_outputs_are_not_overwritten_or_reused(pipeline):
    runner, plan, stages = pipeline
    old_round = Path(plan["work_dir"]) / "round_000"
    old_round.mkdir(parents=True)
    sentinel = old_round / "round_status.json"
    sentinel.write_text("existing outputs must survive", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing stale reuse"):
        runner.run_rounds(plan)

    assert sentinel.read_text(encoding="utf-8") == "existing outputs must survive"
    assert not stages.calls


@pytest.mark.parametrize(
    "case,message,last_stage",
    [
        ("cached_model", "frozen policy", "score"),
        ("candidate_hash", "content digest mismatch", "score"),
        ("scored_actions", "All-K candidate actions", "build_group"),
        ("policy_hash", "(policy|snapshot|checkpoint)", "score"),
    ],
)
def test_stale_candidates_or_misaligned_scores_fail_provenance_before_training(
    pipeline, case, message, last_stage
):
    runner, plan, stages = pipeline
    if case == "cached_model":
        stages.corrupt_candidates = case
    else:
        stages.corrupt_score_parts = case

    with pytest.raises(ValueError, match=message):
        runner.run_rounds(plan)

    status = _read_json(Path(plan["work_dir"]) / "round_000" / "round_status.json")
    assert status["status"] == "FAILED"
    assert stages.calls[-1][0] == last_stage
    assert not any(stage == "train" for stage, _ in stages.calls)
    assert not (Path(plan["work_dir"]) / "round_001").exists()


@pytest.mark.parametrize("case", ["missing", "stale"])
def test_fresh_generation_summary_cannot_be_restamped_with_a_new_policy_hash(pipeline, case):
    runner, plan, stages = pipeline
    stages.corrupt_summary_hash = case

    with pytest.raises(ValueError, match="(hash|SHA|snapshot|provenance|checkpoint)"):
        runner.run_rounds(plan)

    assert [stage for stage, _ in stages.calls] == ["generate"]
    round_dir = Path(plan["work_dir"]) / "round_000"
    assert _read_json(round_dir / "round_status.json")["status"] == "FAILED"
    assert not (round_dir / "candidate_summary_for_scoring.json").exists()
    assert not (Path(plan["work_dir"]) / "round_001").exists()


@pytest.mark.parametrize("artifact", ["candidate", "score_part"])
@pytest.mark.parametrize("case", ["missing", "stale"])
def test_same_model_path_does_not_make_stale_or_missing_policy_hashes_fresh(
    pipeline, artifact, case
):
    runner, plan, stages = pipeline
    if artifact == "candidate":
        stages.corrupt_candidates = f"{case}_policy_hash"
    else:
        stages.corrupt_score_parts = "missing_policy_hash" if case == "missing" else "policy_hash"

    with pytest.raises(ValueError, match="(hash|SHA|snapshot|provenance|checkpoint|policy)"):
        runner.run_rounds(plan)

    assert [stage for stage, _ in stages.calls] == ["generate", "score"]
    round_dir = Path(plan["work_dir"]) / "round_000"
    assert _read_json(round_dir / "round_status.json")["status"] == "FAILED"
    assert not (round_dir / "policy_provenance.json").exists()
    assert not (Path(plan["work_dir"]) / "round_001").exists()


def test_builder_cli_rejects_abbreviated_owned_path_overrides(pipeline, monkeypatch, capsys):
    _, plan, stages = pipeline
    with monkeypatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            [
                str(BUILDER_PATH),
                "--source-dir",
                str(Path(plan["work_dir"]) / "source"),
                "--score-parts-dir",
                str(Path(plan["work_dir"]) / "parts"),
                "--output-dir",
                str(Path(plan["work_dir"]) / "owned_output"),
                "--policy-provenance",
                str(Path(plan["work_dir"]) / "provenance.json"),
                "--output=/stale_override",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            stages.builder.parse_args()
    assert exc.value.code == 2
    assert "unrecognized arguments: --output=/stale_override" in capsys.readouterr().err
    assert not Path(plan["work_dir"]).exists()
