from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from gr00t.data.dataset.flare_group_teacher import (
    GROUP_TEACHER_FORMAT,
    POLICY_PROVENANCE_FORMAT,
    GroupTeacherSelectionConfig,
    sample_group_candidate,
    select_group_teacher,
    validate_group_teacher,
)
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPO_ROOT / "scripts" / "flare" / "build_group_teacher.py"


def _builder():
    spec = importlib.util.spec_from_file_location("build_group_teacher", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**kwargs) -> GroupTeacherSelectionConfig:
    values = {"oracle_max_mse": 0.05, "max_paired_gap": 0.02}
    return GroupTeacherSelectionConfig(**(values | kwargs))


def _inputs(horizon=5):
    return (
        torch.zeros(3, horizon, 7),
        torch.full((3, horizon), 0.02),
        torch.full((horizon,), 0.01),
        torch.tensor([0.2, 0.5, 0.9]),
    )


def test_prefix_is_contiguous_and_oracle_gate_is_shared() -> None:
    actions, errors, oracle, scores = _inputs()
    oracle[2] = 0.5
    q, lengths = select_group_teacher(actions, errors, oracle, scores, _config())
    assert lengths.tolist() == [2, 2, 2]
    assert torch.all(q > 0)
    assert q.sum() == pytest.approx(1.0)
    assert q[2] > q[1] > q[0]


def test_paired_gap_accepts_equality_not_only_oracle_improvement() -> None:
    actions, errors, oracle, scores = _inputs()
    oracle[:] = 0.0
    errors[:] = 0.02
    q, lengths = select_group_teacher(actions, errors, oracle, scores, _config())
    assert lengths.tolist() == [5, 5, 5]
    assert q.sum() == pytest.approx(1.0)


def test_candidate_failure_does_not_restart_supervision() -> None:
    actions, errors, oracle, scores = _inputs()
    errors[0, 1] = 0.5
    actions[1, 4, 0] = 1.1
    q, lengths = select_group_teacher(actions, errors, oracle, scores, _config(min_prefix=2))
    assert lengths.tolist() == [0, 4, 5]
    assert q[0] == 0
    assert q[2] > q[1] > 0


def test_nonfinite_action_or_error_truncates_prefix() -> None:
    actions, errors, oracle, scores = _inputs()
    actions[0, 3, 0] = float("nan")
    errors[1, 2] = float("inf")
    q, lengths = select_group_teacher(actions, errors, oracle, scores, _config())
    assert lengths.tolist() == [3, 2, 5]
    assert torch.isfinite(q).all()


@pytest.mark.parametrize(
    "case", ["ties", "insufficient_span", "invalid_oracle", "single_eligible", "invalid_scores"]
)
def test_uninformative_groups_fall_back_to_gt(case: str) -> None:
    actions, errors, oracle, scores = _inputs()
    config = _config()
    if case == "ties":
        scores[:] = 0.5
    elif case == "insufficient_span":
        config = _config(min_score_span=1.0)
    elif case == "invalid_oracle":
        oracle[0] = float("nan")
    elif case == "single_eligible":
        errors[:2, 0] = 1.0
    else:
        scores[:] = float("nan")
    q, lengths = select_group_teacher(actions, errors, oracle, scores, config)
    assert not q.any()
    assert not lengths.any()
    assert sample_group_candidate(q) is None


def test_std_floor_prevents_amplifying_tiny_score_noise() -> None:
    actions, errors, oracle, _ = _inputs()
    scores = torch.tensor([0.0, 1e-10, 2e-10])
    q, _ = select_group_teacher(actions, errors, oracle, scores, _config(score_std_floor=0.001))
    assert torch.allclose(q, torch.full((3,), 1 / 3), atol=1e-5)


def test_shared_prefix_scoring_ignores_untrusted_tail_without_length_bias() -> None:
    actions, errors, oracle, _ = _inputs()
    errors[0, 2] = 0.5
    step_scores = torch.tensor(
        [
            [0.9, 0.9, -0.5, -0.5, -0.5],
            [0.5, 0.5, 0.8, 0.8, 0.8],
            [0.1, 0.1, 0.9, 0.9, 0.9],
        ]
    )
    scores = step_scores.mean(dim=1)
    shared_q, lengths = select_group_teacher(
        actions,
        errors,
        oracle,
        scores,
        _config(),
        score_per_step=step_scores,
        score_scope="shared-prefix",
    )
    full_q, _ = select_group_teacher(actions, errors, oracle, scores, _config())
    assert lengths.tolist() == [2, 5, 5]
    assert shared_q[0] > shared_q[1] > shared_q[2]
    assert full_q[0] < full_q[1]
    step_scores[:, :2] = 0.5
    tied_q, tied_lengths = select_group_teacher(
        actions,
        errors,
        oracle,
        scores,
        _config(),
        score_per_step=step_scores,
        score_scope="shared-prefix",
    )
    assert not tied_q.any()
    assert not tied_lengths.any()


def test_shared_prefix_does_not_require_finite_full_horizon_scores() -> None:
    actions, errors, oracle, scores = _inputs()
    step_scores = scores[:, None].expand(-1, 5).clone()
    scores[:] = float("nan")
    q, lengths = select_group_teacher(
        actions,
        errors,
        oracle,
        scores,
        _config(),
        score_per_step=step_scores,
        score_scope="shared-prefix",
    )
    assert q.sum() == pytest.approx(1.0)
    assert lengths.tolist() == [5, 5, 5]


def test_ablation_flags_preserve_action_and_finiteness_gates() -> None:
    actions, errors, oracle, scores = _inputs()
    oracle[:] = float("inf")
    actions[0, 4, 0] = 2.0
    config = _config(oracle_anchor=False, fixed_prefix=3)
    q, lengths = select_group_teacher(actions, errors, oracle, scores, config)
    assert lengths.tolist() == [3, 3, 3]
    assert q.sum() == pytest.approx(1.0)
    config = _config(oracle_anchor=False, fixed_prefix=5)
    _, lengths = select_group_teacher(actions, errors, oracle, scores, config)
    assert lengths.tolist() == [0, 5, 5]
    config = _config(oracle_anchor=False, fixed_prefix=5, action_min=None, action_max=None)
    _, lengths = select_group_teacher(actions, errors, oracle, scores, config)
    assert lengths.tolist() == [5, 5, 5]


def test_mc_sampling_prefers_higher_weight_and_is_reproducible() -> None:
    q = torch.tensor([0.1, 0.9])
    generator = torch.Generator().manual_seed(44)
    draws = [sample_group_candidate(q, generator) for _ in range(1024)]
    assert 0.85 < sum(draws) / len(draws) < 0.95
    replay = torch.Generator().manual_seed(44)
    assert draws[:32] == [sample_group_candidate(q, replay) for _ in range(32)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"temperature": 0},
        {"score_std_floor": 0},
        {"min_prefix": 0},
        {"min_score_span": -1},
        {"action_min": None},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        if kwargs:
            _config(**kwargs)
        else:
            GroupTeacherSelectionConfig()


def _write_fixture(tmp_path: Path):
    source = tmp_path / "source"
    parts = tmp_path / "score_parts"
    source.mkdir()
    parts.mkdir()
    sample_ids = ["sample_000002", "sample_000001"]
    identities = [2, 1]
    actions = torch.stack([torch.full((3, 4, 7), 0.2 + 0.1 * i) for i in range(2)])
    actions[:, 1] += 0.1
    actions[:, 2] += 0.2
    teacher = {
        "format": "flare_distillation_teacher_records_v1",
        "manifest_sha256": "d" * 64,
        "sample_ids": sample_ids,
        "sample_indices": torch.tensor(identities),
        "episode_indices": torch.tensor(identities),
        "local_steps": torch.tensor([3, 5]),
        "teacher_actions_lerobot": actions[:, 0].clone(),
        "teacher_idx_primary": torch.zeros(2, dtype=torch.long),
        "teacher_horizon": 4,
        "processor_action_horizon": 16,
        "model_action_horizon": 40,
        "action_dim": 7,
        "action_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    }
    torch.save(teacher, source / "teacher_records_enriched.pt")
    rows = []
    for source_name in ("gt", "teacher"):
        for index, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "format": "flare_distillation_index_v1",
                    "distillation_index": len(rows),
                    "sample_id": sample_id,
                    "sample_index": identities[index],
                    "episode_index": identities[index],
                    "local_step": [3, 5][index],
                    "source": source_name,
                    "teacher_row": index if source_name == "teacher" else None,
                }
            )
    (source / "distillation_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (source / "distillation_summary.json").write_text(
        json.dumps(
            {
                "format": "flare_distillation_summary_v1",
                "status": "PASS",
                "protocol": {"total_samples": 4, "gt_samples": 2, "teacher_samples": 2},
            }
        )
    )
    candidates = []
    for index, sample_id in enumerate(sample_ids):
        candidate_hash = hashlib.sha256(sample_id.encode()).hexdigest()
        candidates.append({"sample_id": sample_id, "candidate_sha256": candidate_hash})
        torch.save(
            {
                "format": "flare_lewm_libero_plus_score_part_v2",
                "sample_id": sample_id,
                "sample_index": identities[index],
                "episode_index": identities[index],
                "local_step": [3, 5][index],
                "manifest_sha256": "d" * 64,
                "checkpoint_sha256": "a" * 64,
                "candidate_sha256": candidate_hash,
                "candidate_k": 3,
                "future_horizon": 4,
                "candidate_actions_lerobot": actions[index],
                "mse_per_step": torch.full((3, 4), 0.02),
                "oracle_mse_per_step": torch.full((4,), 0.01),
                "cosine_score": torch.tensor([0.9, 0.7, 0.5]),
                "cosine_per_step": torch.tensor([[0.9] * 4, [0.7] * 4, [0.5] * 4]),
                "mse_score": torch.tensor([0.01, 0.02, 0.03]),
            },
            parts / f"{sample_id}.pt",
        )
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "format": POLICY_PROVENANCE_FORMAT,
                "round_id": 0,
                "policy_checkpoint_sha256": "b" * 64,
                "provenance_source": "synthetic test fixture",
                "candidates": candidates,
            }
        )
    )
    kwargs = dict(
        source_dir=source,
        score_parts_dir=parts,
        output_dir=tmp_path / "group_output",
        policy_provenance_path=provenance_path,
        selection_config=_config(),
    )
    return teacher, kwargs


def test_builder_preserves_v1_payload_and_teacher_row_order(tmp_path: Path) -> None:
    original, kwargs = _write_fixture(tmp_path)
    source_index = (kwargs["source_dir"] / "distillation_index.jsonl").read_bytes()
    builder = _builder()
    summary = builder.build_group_teacher(**kwargs)
    output = kwargs["output_dir"]
    result = torch.load(output / "teacher_records_enriched.pt", weights_only=False)
    assert result["format"] == original["format"]
    assert torch.equal(result["teacher_actions_lerobot"], original["teacher_actions_lerobot"])
    assert (output / "distillation_index.jsonl").read_bytes() == source_index
    group = result["group_teacher"]
    assert group["format"] == GROUP_TEACHER_FORMAT
    assert group["sample_ids"] == original["sample_ids"]
    assert group["candidate_actions_lerobot"][0, 0, 0, 0] == pytest.approx(0.2)
    assert group["candidate_actions_lerobot"][1, 0, 0, 0] == pytest.approx(0.3)
    assert summary["protocol"] == {"total_samples": 4, "gt_samples": 2, "teacher_samples": 2}
    assert validate_group_teacher(None, original) is None
    assert validate_group_teacher(group, original) is not None
    assert "group_teacher" not in torch.load(
        kwargs["source_dir"] / "teacher_records_enriched.pt", weights_only=False
    )
    with pytest.raises(FileExistsError):
        builder.build_group_teacher(**kwargs)


def test_builder_fallback_dry_run_and_mse_scoring(tmp_path: Path) -> None:
    _, kwargs = _write_fixture(tmp_path)
    path = kwargs["score_parts_dir"] / "sample_000001.pt"
    part = torch.load(path, weights_only=False)
    part["mse_score"] = torch.zeros(3)
    torch.save(part, path)
    summary = _builder().build_group_teacher(
        **kwargs, score_metric="mse", score_scope="full", dry_run=True
    )
    assert not kwargs["output_dir"].exists()
    assert summary["group_teacher"]["trusted_teacher_groups"] == 1
    assert summary["group_teacher"]["gt_fallback_groups"] == 1


def test_score_loader_keeps_only_compact_selected_fields(tmp_path: Path) -> None:
    _, kwargs = _write_fixture(tmp_path)
    builder = _builder()
    for path in kwargs["score_parts_dir"].glob("*.pt"):
        part = torch.load(path, weights_only=False)
        part["predicted_latents"] = torch.ones(3, 4, 192)
        part["gt_latents"] = torch.ones(4, 192)
        part["unrelated_diagnostic"] = "do not retain"
        torch.save(part, path)
    provenance, candidate_hashes = builder._load_provenance(kwargs["policy_provenance_path"])
    assert "candidates" not in provenance
    parts, wm_hash = builder._load_score_parts(
        kwargs["score_parts_dir"],
        provenance,
        candidate_hashes,
        keep_sample_ids={"sample_000002"},
    )
    assert set(parts) == {"sample_000002"}
    assert wm_hash == "a" * 64
    kept = parts["sample_000002"][1]
    assert "candidate_actions_lerobot" in kept
    assert "cosine_per_step" in kept
    assert "predicted_latents" not in kept
    assert "gt_latents" not in kept
    assert "unrelated_diagnostic" not in kept
    empty, _ = builder._load_score_parts(
        kwargs["score_parts_dir"], provenance, candidate_hashes, keep_sample_ids=set()
    )
    assert empty == {}


@pytest.mark.parametrize("case", ["duplicate", "missing", "cross_wm"])
def test_unselected_score_parts_still_have_full_provenance_validation(
    tmp_path: Path, case: str
) -> None:
    _, kwargs = _write_fixture(tmp_path)
    builder = _builder()
    provenance, candidate_hashes = builder._load_provenance(kwargs["policy_provenance_path"])
    path = kwargs["score_parts_dir"] / "sample_000001.pt"
    part = torch.load(path, weights_only=False)
    if case == "duplicate":
        torch.save(part, path.with_name("duplicate_unselected.pt"))
    elif case == "missing":
        path.unlink()
    else:
        part["checkpoint_sha256"] = "c" * 64
        torch.save(part, path)
    with pytest.raises(ValueError):
        builder._load_score_parts(
            kwargs["score_parts_dir"],
            provenance,
            candidate_hashes,
            keep_sample_ids={"sample_000002"},
        )


@pytest.mark.parametrize("input_name", ["source_dir", "score_parts_dir"])
def test_builder_refuses_output_nested_in_input(tmp_path: Path, input_name: str) -> None:
    _, kwargs = _write_fixture(tmp_path)
    kwargs["output_dir"] = kwargs[input_name] / "nested_output"
    with pytest.raises(ValueError, match="must not be nested"):
        _builder().build_group_teacher(**kwargs)
    assert not kwargs["output_dir"].exists()


def _attach_candidate_file(
    kwargs: dict, *, extra_horizon: int = 0, include_actions: bool = True
) -> Path:
    path = kwargs["score_parts_dir"] / "sample_000001.pt"
    part = torch.load(path, weights_only=False)
    candidate_dir = kwargs["score_parts_dir"].parent / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    candidate_path = candidate_dir / "sample_000001.pt"
    payload = {"model_path": "/frozen/policy"}
    if include_actions:
        actions = part["candidate_actions_lerobot"]
        if extra_horizon:
            actions = torch.cat(
                [actions, torch.zeros(actions.shape[0], extra_horizon, actions.shape[2])], dim=1
            )
        payload["candidate_actions_concat"] = actions.unsqueeze(0)
    torch.save(payload, candidate_path)
    part["candidate_path"] = str(candidate_path)
    part["candidate_sha256"] = _builder().sha256_file(candidate_path)
    torch.save(part, path)
    provenance = json.loads(kwargs["policy_provenance_path"].read_text())
    for row in provenance["candidates"]:
        if row["sample_id"] == "sample_000001":
            row["candidate_sha256"] = part["candidate_sha256"]
    kwargs["policy_provenance_path"].write_text(json.dumps(provenance))
    return path


@pytest.mark.parametrize("extra_horizon,include_actions", [(0, True), (2, True), (0, False)])
def test_hashed_candidate_actions_match_scoring_prefix_and_legacy_metadata(
    tmp_path: Path, extra_horizon: int, include_actions: bool
) -> None:
    _, kwargs = _write_fixture(tmp_path)
    _attach_candidate_file(kwargs, extra_horizon=extra_horizon, include_actions=include_actions)
    assert _builder().build_group_teacher(**kwargs)["status"] == "PASS"


def test_hashed_all_k_check_rejects_changed_unselected_candidate(tmp_path: Path) -> None:
    _, kwargs = _write_fixture(tmp_path)
    path = _attach_candidate_file(kwargs)
    part = torch.load(path, weights_only=False)
    part["candidate_actions_lerobot"][1] += 0.01
    torch.save(part, path)
    with pytest.raises(ValueError, match="All-K candidate actions"):
        _builder().build_group_teacher(**kwargs)
    assert not kwargs["output_dir"].exists()


@pytest.mark.parametrize(
    "case",
    [
        "missing_part",
        "duplicate_part",
        "candidate_hash",
        "wm_hash",
        "policy_hash",
        "round_id",
        "identity",
        "teacher_action",
        "duplicate_provenance",
        "missing_provenance",
    ],
)
def test_builder_rejects_misalignment_and_provenance_mix(tmp_path: Path, case: str) -> None:
    _, kwargs = _write_fixture(tmp_path)
    path = kwargs["score_parts_dir"] / "sample_000001.pt"
    part = torch.load(path, weights_only=False)
    if case == "missing_part":
        path.unlink()
    elif case == "duplicate_part":
        torch.save(part, path.with_name("duplicate.pt"))
    elif case == "candidate_hash":
        part["candidate_sha256"] = "c" * 64
    elif case == "wm_hash":
        part["checkpoint_sha256"] = "c" * 64
    elif case == "policy_hash":
        part["policy_checkpoint_sha256"] = "c" * 64
    elif case == "round_id":
        part["round_id"] = 1
    elif case == "identity":
        part["local_step"] = 999
    elif case == "teacher_action":
        part["candidate_actions_lerobot"][0] += 0.01
    else:
        provenance = json.loads(kwargs["policy_provenance_path"].read_text())
        if case == "duplicate_provenance":
            provenance["candidates"].append(provenance["candidates"][0])
        else:
            del provenance["policy_checkpoint_sha256"]
        kwargs["policy_provenance_path"].write_text(json.dumps(provenance))
    if case not in {"missing_part", "duplicate_part", "duplicate_provenance", "missing_provenance"}:
        torch.save(part, path)
    with pytest.raises((ValueError, KeyError)):
        _builder().build_group_teacher(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_builder_sanitizes_only_unsupervised_nonfinite_coordinates(tmp_path: Path) -> None:
    _, kwargs = _write_fixture(tmp_path)
    path = kwargs["score_parts_dir"] / "sample_000001.pt"
    part = torch.load(path, weights_only=False)
    part["candidate_actions_lerobot"][1, 2, 0] = float("nan")
    torch.save(part, path)
    _builder().build_group_teacher(**kwargs)
    result = torch.load(kwargs["output_dir"] / "teacher_records_enriched.pt", weights_only=False)
    group = result["group_teacher"]
    assert group["prefix_lengths"][1, 1] == 2
    assert group["candidate_actions_lerobot"][1, 1, 2, 0] == 0
    assert torch.isfinite(group["candidate_actions_lerobot"]).all()


@pytest.mark.parametrize(
    "case",
    [
        "negative_q",
        "q_sum",
        "zero_prefix",
        "inactive_prefix",
        "noninteger_prefix",
        "nonfinite_actions",
        "bad_hash",
        "identity",
    ],
)
def test_group_validation_rejects_corrupted_payload(tmp_path: Path, case: str) -> None:
    teacher, kwargs = _write_fixture(tmp_path)
    _builder().build_group_teacher(**kwargs)
    group = copy.deepcopy(
        torch.load(kwargs["output_dir"] / "teacher_records_enriched.pt", weights_only=False)[
            "group_teacher"
        ]
    )
    if case == "negative_q":
        group["candidate_probabilities"][0, 0] = -0.1
    elif case == "q_sum":
        group["candidate_probabilities"][0] *= 0.5
    elif case == "zero_prefix":
        group["prefix_lengths"][0, 0] = 0
    elif case == "inactive_prefix":
        group["candidate_probabilities"][0, 0] = 0
        group["candidate_probabilities"][0] /= group["candidate_probabilities"][0].sum()
    elif case == "noninteger_prefix":
        group["prefix_lengths"] = group["prefix_lengths"].float() - 0.5
    elif case == "nonfinite_actions":
        group["candidate_actions_lerobot"][0, 0, -1, 0] = float("nan")
    elif case == "bad_hash":
        group["policy_checkpoint_sha256"] = "mutable/path"
    else:
        group["sample_ids"] = list(reversed(group["sample_ids"]))
    with pytest.raises((ValueError, TypeError)):
        validate_group_teacher(group, teacher)


def test_cli_help_is_cpu_only() -> None:
    result = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--help"], text=True, capture_output=True, check=True
    )
    assert "--policy-provenance" in result.stdout
    assert "--no-oracle-anchor" in result.stdout
    assert "--fixed-prefix" in result.stdout
