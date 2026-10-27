"""CPU provenance tests with real H5/actions and stubbed inference imports."""

from __future__ import annotations

from enum import Enum
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from gr00t.utils.checkpoint_snapshot import checkpoint_snapshot_sha256
import h5py
import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "scripts" / "flare" / "04_generate_validation_candidates.py"
SCORER_PATH = REPO_ROOT / "FLARE-VLA" / "scripts" / "libero_plus" / "05_score_libero_plus.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_scorer(monkeypatch):
    # No LeWM model/resolver registration is needed for H5 preparation and
    # cached-score aggregation; tests never invoke its inference functions.
    monkeypatch.setitem(sys.modules, "stable_pretraining", ModuleType("stable_pretraining"))
    return _load_module("single_policy_scorer", SCORER_PATH)


def _load_generator(monkeypatch):
    for module_name, attribute in (
        ("gr00t.data.dataset.lerobot_episode_loader", "LeRobotEpisodeLoader"),
        ("gr00t.policy.gr00t_policy", "Gr00tPolicy"),
    ):
        module = ModuleType(module_name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, module_name, module)
    tags = ModuleType("gr00t.data.embodiment_tags")
    tags.EmbodimentTag = Enum("EmbodimentTag", {"LIBERO_PANDA": "libero_panda"})
    monkeypatch.setitem(sys.modules, tags.__name__, tags)
    helpers = ModuleType("generate_k_candidates_openloop")
    for name in (
        "_extract_gt_action_chunk",
        "_get_internal_noise_shape",
        "_infer_batch_size",
        "_to_saveable_cpu",
        "build_explicit_initial_noises",
        "prepare_observation_data",
        "set_seed",
    ):
        setattr(helpers, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    return _load_module("single_policy_generator", GENERATOR_PATH)


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "frozen_policy"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"synthetic CPU checkpoint contents")
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    return checkpoint


def _candidate_fixture(tmp_path: Path, scorer):
    checkpoint = _checkpoint(tmp_path)
    policy_hash = checkpoint_snapshot_sha256(checkpoint)
    row = {
        "format": scorer.FULL_MANIFEST_FORMAT,
        "sample_id": "sample_000000",
        "sample_index": 0,
        "suite": "libero_10",
        "gt_latent_key": "libero_10",
        "episode_index": 0,
        "lewm_episode_index": 0,
        "episode_length": 14,
        "local_step": 2,
        "global_index": 2,
        "window_index_h8": 0,
        "task_index": 7,
        "task_text": "Move the object",
        "history_rows": [0, 1, 2],
        "future_rows": list(range(3, 11)),
        "history_size": 3,
        "window_horizon": 8,
        "candidate_k": 4,
        "sample_seed": 100,
        "candidate_seeds": [100, 101, 102, 103],
        "noise_mode": "independent",
        "noise_scale": 1.0,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest_hash = scorer.sha256_file(manifest)
    actions = np.zeros((14, 7), dtype=np.float32)
    actions[:, 6] = 1.0
    latents = np.ones((14, 192), dtype=np.float32)
    h5_path = tmp_path / "framebank.h5"
    with h5py.File(h5_path, "w") as h5:
        h5["frame_action"] = actions
        h5["frame_latent"] = latents
        h5["window_current_row"] = np.asarray([2])
        h5["window_episode_id"] = np.asarray([0])
        h5["window_step_idx"] = np.asarray([2])
        h5["window_future_row"] = np.asarray([row["future_rows"]])
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate_path = candidate_dir / f"{row['sample_id']}.pt"
    payload = {
        **row,
        "format": scorer.CANDIDATE_FORMAT,
        "manifest_format": scorer.FULL_MANIFEST_FORMAT,
        "manifest_sha256": manifest_hash,
        "model_path": str(checkpoint),
        "policy_checkpoint_sha256": policy_hash,
        "decoded_action_horizon": 8,
        "candidate_actions_concat": torch.zeros(1, 4, 8, 7),
        "gt_actions_concat": torch.from_numpy(actions[2:10].copy()),
    }
    torch.save(payload, candidate_path)
    kwargs = dict(
        row=row,
        candidate_file=candidate_path,
        candidate_sha256=scorer.sha256_file(candidate_path),
        ep_offsets=np.asarray([0, 14]),
        candidate_k=4,
        history_size=3,
        horizon=8,
        manifest_sha256=manifest_hash,
    )
    return checkpoint, policy_hash, manifest, h5_path, kwargs


def _prepare(
    scorer,
    h5_path: Path,
    kwargs: dict,
    checkpoint: Path | None = None,
    policy_hash: str | None = None,
):
    with h5py.File(h5_path, "r") as h5:
        return scorer.prepare_sample(
            **kwargs,
            h5=h5,
            single_policy_checkpoint=checkpoint,
            policy_checkpoint_sha256=policy_hash,
        )


def _make_part(scorer, prepared: dict, *, wm_hash: str = "c" * 64, gt_hash: str = "d" * 64):
    target = prepared["gt_latents"]
    predicted = target.unsqueeze(0).repeat(4, 1, 1)
    predicted[:, :, 0] += torch.arange(4)[:, None] * 0.1
    return scorer.make_score_part(
        prepared=prepared,
        predicted=predicted,
        oracle_predicted=target.clone(),
        checkpoint_sha256=wm_hash,
        manifest_sha256=prepared["manifest_sha256"],
        gt_h5_sha256=gt_hash,
        scoring_seconds=0.0,
        save_predicted_latents=False,
    )


def test_prepare_sample_single_policy_is_opt_in_and_retains_h5_checks(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = _load_scorer(monkeypatch)
    checkpoint, policy_hash, _, h5_path, kwargs = _candidate_fixture(tmp_path, scorer)
    with pytest.raises(RuntimeError, match="wrong suite checkpoint"):
        _prepare(scorer, h5_path, kwargs)
    prepared = _prepare(scorer, h5_path, kwargs, checkpoint, policy_hash)
    assert prepared["policy_checkpoint_sha256"] == policy_hash
    assert prepared["candidate_actions_lerobot"].shape == (4, 8, 7)
    assert prepared["all_action_sequences"].shape == (5, 10, 7)
    with h5py.File(h5_path, "r+") as h5:
        h5["window_current_row"][0] = 3
    with pytest.raises(RuntimeError, match="window_current_row"):
        _prepare(scorer, h5_path, kwargs, checkpoint, policy_hash)


@pytest.mark.parametrize("case", ["wrong_path", "wrong_hash", "missing_hash", "wrong_manifest"])
def test_single_policy_prepare_rejects_bad_provenance(
    tmp_path: Path, monkeypatch, case: str
) -> None:
    scorer = _load_scorer(monkeypatch)
    checkpoint, policy_hash, _, h5_path, kwargs = _candidate_fixture(tmp_path, scorer)
    payload = torch.load(kwargs["candidate_file"], weights_only=False)
    if case == "wrong_path":
        payload["model_path"] = str(tmp_path / "another_policy")
    elif case == "wrong_hash":
        payload["policy_checkpoint_sha256"] = "e" * 64
    elif case == "missing_hash":
        del payload["policy_checkpoint_sha256"]
    else:
        payload["manifest_sha256"] = "e" * 64
    torch.save(payload, kwargs["candidate_file"])
    with pytest.raises(RuntimeError):
        _prepare(scorer, h5_path, kwargs, checkpoint, policy_hash)


def test_legacy_prepare_still_requires_suite_and_merger_fields(tmp_path: Path, monkeypatch) -> None:
    scorer = _load_scorer(monkeypatch)
    _, _, _, h5_path, kwargs = _candidate_fixture(tmp_path, scorer)
    payload = torch.load(kwargs["candidate_file"], weights_only=False)
    payload["model_path"] = str(tmp_path / "libero_10")
    torch.save(payload, kwargs["candidate_file"])
    with pytest.raises(RuntimeError, match="per_suite_manifest_sha256"):
        _prepare(scorer, h5_path, kwargs)
    payload["per_suite_manifest_sha256"] = "e" * 64
    payload["source_candidate_sha256"] = "f" * 64
    torch.save(payload, kwargs["candidate_file"])
    assert "policy_checkpoint_sha256" not in _prepare(scorer, h5_path, kwargs)


@pytest.mark.parametrize("case", ["path_only", "hash_only", "bad_hash"])
def test_single_policy_options_are_paired(tmp_path: Path, monkeypatch, case: str) -> None:
    scorer = _load_scorer(monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    path = None if case == "hash_only" else checkpoint
    digest = None if case == "path_only" else ("not-a-hash" if case == "bad_hash" else "e" * 64)
    with pytest.raises(ValueError):
        scorer.verify_single_policy_checkpoint(path, digest)


def test_single_policy_verification_uses_actual_checkpoint_contents(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = _load_scorer(monkeypatch)
    generator = _load_generator(monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    digest = checkpoint_snapshot_sha256(checkpoint)
    assert scorer.verify_single_policy_checkpoint(checkpoint, digest) == (checkpoint, digest)
    assert generator.verify_policy_checkpoint(checkpoint, digest) == digest
    assert scorer.verify_single_policy_checkpoint(None, None) == (None, None)
    assert generator.verify_policy_checkpoint(checkpoint, None) is None
    (checkpoint / "model.pt").write_bytes(b"changed frozen weights")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        scorer.verify_single_policy_checkpoint(checkpoint, digest)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        generator.verify_policy_checkpoint(checkpoint, digest)


def test_policy_hash_in_score_parts_and_cache_is_optional_for_legacy(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = _load_scorer(monkeypatch)
    checkpoint, policy_hash, _, h5_path, kwargs = _candidate_fixture(tmp_path, scorer)
    prepared = _prepare(scorer, h5_path, kwargs, checkpoint, policy_hash)
    prepared["manifest_sha256"] = kwargs["manifest_sha256"]
    part = _make_part(scorer, prepared)
    assert part["policy_checkpoint_sha256"] == policy_hash
    path = tmp_path / "score.pt"
    torch.save(part, path)
    args = (
        path,
        kwargs["row"],
        kwargs["manifest_sha256"],
        "c" * 64,
        "d" * 64,
        kwargs["candidate_sha256"],
        4,
        8,
    )
    assert scorer.valid_score_part(*args, policy_checkpoint_sha256=policy_hash)
    assert not scorer.valid_score_part(*args, policy_checkpoint_sha256="e" * 64)
    del part["policy_checkpoint_sha256"]
    torch.save(part, path)
    assert scorer.valid_score_part(*args)
    assert not scorer.valid_score_part(*args, policy_checkpoint_sha256=policy_hash)


def test_single_policy_does_not_change_scoring_math(tmp_path: Path, monkeypatch) -> None:
    scorer = _load_scorer(monkeypatch)
    checkpoint, policy_hash, _, h5_path, kwargs = _candidate_fixture(tmp_path, scorer)
    prepared = _prepare(scorer, h5_path, kwargs, checkpoint, policy_hash)
    prepared["manifest_sha256"] = kwargs["manifest_sha256"]
    single = _make_part(scorer, prepared)
    prepared.pop("policy_checkpoint_sha256")
    legacy = _make_part(scorer, prepared)
    for name in (
        "cosine_per_step",
        "mse_per_step",
        "cosine_score",
        "mse_score",
        "oracle_mse_per_step",
    ):
        assert torch.equal(single[name], legacy[name])
    assert "policy_checkpoint_sha256" not in legacy


def _candidate_summary(
    path: Path, scorer, checkpoint: Path, policy_hash: str, manifest: Path, candidate_dir: Path
) -> None:
    path.write_text(
        json.dumps(
            {
                "format": scorer.CANDIDATE_SUMMARY_FORMAT,
                "status": "PASS",
                "manifest_format": scorer.FULL_MANIFEST_FORMAT,
                "manifest": str(manifest),
                "manifest_sha256": scorer.sha256_file(manifest),
                "output_dir": str(candidate_dir),
                "model_path": str(checkpoint),
                "policy_checkpoint_sha256": policy_hash,
                "counters": {"selected": 1, "generated": 1, "skipped_valid": 0, "failed": 0},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("bad_field", [None, "model_path", "policy_checkpoint_sha256"])
def test_single_policy_candidate_summary_checks_snapshot_metadata(
    tmp_path: Path, monkeypatch, bad_field: str | None
) -> None:
    scorer = _load_scorer(monkeypatch)
    checkpoint, policy_hash, manifest, _, kwargs = _candidate_fixture(tmp_path, scorer)
    path = tmp_path / "candidate_summary.json"
    _candidate_summary(
        path, scorer, checkpoint, policy_hash, manifest, kwargs["candidate_file"].parent
    )
    if bad_field:
        summary = json.loads(path.read_text())
        summary[bad_field] = str(tmp_path / "bad_model") if bad_field == "model_path" else "e" * 64
        path.write_text(json.dumps(summary))
    args = (path, manifest, kwargs["candidate_file"].parent, kwargs["manifest_sha256"], 1)
    if bad_field:
        with pytest.raises(RuntimeError):
            scorer.validate_candidate_summary(*args, checkpoint, policy_hash)
    else:
        assert scorer.validate_candidate_summary(*args, checkpoint, policy_hash)["status"] == "PASS"


def test_generator_resume_cache_requires_policy_hash_in_single_policy_mode(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = _load_scorer(monkeypatch)
    generator = _load_generator(monkeypatch)
    checkpoint, policy_hash, _, _, kwargs = _candidate_fixture(tmp_path, scorer)
    args = (
        kwargs["candidate_file"],
        kwargs["row"],
        8,
        kwargs["manifest_sha256"],
        scorer.FULL_MANIFEST_FORMAT,
        str(checkpoint),
    )
    assert generator.validate_existing_candidate(*args, policy_hash) == (True, "ok")
    assert generator.validate_existing_candidate(*args, "e" * 64) == (
        False,
        "policy_checkpoint_sha256 mismatch",
    )
    payload = torch.load(kwargs["candidate_file"], weights_only=False)
    del payload["policy_checkpoint_sha256"]
    torch.save(payload, kwargs["candidate_file"])
    assert generator.validate_existing_candidate(*args) == (True, "ok")
    assert not generator.validate_existing_candidate(*args, policy_hash)[0]


@pytest.mark.parametrize("mutate_checkpoint", [False, True])
def test_generator_main_verifies_frozen_snapshot_before_and_after_generation(
    tmp_path: Path, monkeypatch, mutate_checkpoint: bool
) -> None:
    scorer = _load_scorer(monkeypatch)
    generator = _load_generator(monkeypatch)
    checkpoint, policy_hash, manifest, _, kwargs = _candidate_fixture(tmp_path, scorer)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "generation_run"
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    modality = {"action": SimpleNamespace(modality_keys=keys)}

    class FakePolicy:
        def __init__(self, **kwargs):
            pass

        def get_modality_config(self):
            return modality

        def get_action_candidates(self, observations, initial_noises):
            if mutate_checkpoint:
                (checkpoint / "model.pt").write_bytes(b"changed during candidate generation")
            return {key: np.zeros((1, 4, 40, 1), dtype=np.float32) for key in keys}, {
                "backbone_evaluations": 1
            }

    class FakeLoader:
        def __init__(self, **kwargs):
            self.modality_configs = modality

        def __getitem__(self, index):
            return [None] * 14

    monkeypatch.setattr(generator, "Gr00tPolicy", FakePolicy)
    monkeypatch.setattr(generator, "LeRobotEpisodeLoader", FakeLoader)
    monkeypatch.setattr(generator, "prepare_observation_data", lambda **kwargs: {})
    monkeypatch.setattr(generator, "_infer_batch_size", lambda observations: 1)
    monkeypatch.setattr(
        generator, "_get_internal_noise_shape", lambda policy, batch_size: (1, 40, 7)
    )
    monkeypatch.setattr(
        generator,
        "build_explicit_initial_noises",
        lambda **kwargs: (torch.zeros(4, 1, 40, 7), [100, 101, 102, 103], None),
    )
    monkeypatch.setattr(
        generator,
        "_extract_gt_action_chunk",
        lambda **kwargs: (
            {key: np.zeros((8, 1), dtype=np.float32) for key in keys},
            np.zeros((8, 7), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(generator, "_to_saveable_cpu", lambda value: value)
    monkeypatch.setattr(generator, "set_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(GENERATOR_PATH),
            "--manifest",
            str(manifest),
            "--dataset-path",
            str(dataset),
            "--model-path",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--manifest-format",
            scorer.FULL_MANIFEST_FORMAT,
            "--policy-checkpoint-sha256",
            policy_hash,
        ],
    )
    if mutate_checkpoint:
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            generator.main()
        assert not (output / "candidate_generation_summary.json").exists()
    else:
        generator.main()
        summary = json.loads((output / "candidate_generation_summary.json").read_text())
        candidate = torch.load(
            output / "candidates" / f"{kwargs['row']['sample_id']}.pt", weights_only=False
        )
        assert summary["policy_checkpoint_sha256"] == policy_hash
        assert candidate["policy_checkpoint_sha256"] == policy_hash


def test_scorer_cached_main_propagates_policy_hash_without_wm_inference(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = _load_scorer(monkeypatch)
    checkpoint, policy_hash, manifest, h5_path, kwargs = _candidate_fixture(tmp_path, scorer)
    prepared = _prepare(scorer, h5_path, kwargs, checkpoint, policy_hash)
    prepared["manifest_sha256"] = kwargs["manifest_sha256"]
    wm_path = tmp_path / "wm_weights.pt"
    wm_path.write_bytes(b"synthetic unused WM checkpoint")
    wm_hash, gt_hash = scorer.sha256_file(wm_path), scorer.sha256_file(h5_path)
    part = _make_part(scorer, prepared, wm_hash=wm_hash, gt_hash=gt_hash)
    output = tmp_path / "scoring_run"
    parts = output / "score_parts"
    parts.mkdir(parents=True)
    torch.save(part, parts / f"{kwargs['row']['sample_id']}.pt")
    summary_path = tmp_path / "candidate_summary.json"
    _candidate_summary(
        summary_path, scorer, checkpoint, policy_hash, manifest, kwargs["candidate_file"].parent
    )
    hash_calls = []

    def check_hash(path):
        hash_calls.append(path)
        return checkpoint_snapshot_sha256(path)

    def forbid_wm(*args, **kwargs):
        raise AssertionError("Cached metadata test must not construct a WM")

    monkeypatch.setattr(scorer, "checkpoint_snapshot_sha256", check_hash)
    monkeypatch.setattr(scorer, "build_model", forbid_wm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCORER_PATH),
            "--manifest",
            str(manifest),
            "--candidate-dir",
            str(kwargs["candidate_file"].parent),
            "--candidate-summary",
            str(summary_path),
            "--gt-h5",
            str(h5_path),
            "--checkpoint",
            str(wm_path),
            "--expected-checkpoint-sha256",
            wm_hash,
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--single-policy-checkpoint",
            str(checkpoint),
            "--policy-checkpoint-sha256",
            policy_hash,
        ],
    )
    scorer.main()
    run = json.loads((output / "scoring_run_summary.json").read_text())
    validation = json.loads((output / "validation_summary.json").read_text())
    teachers = torch.load(output / "validation_teacher_records.pt", weights_only=False)
    assert run["skipped_valid"] == 1
    assert run["generated"] == 0
    assert (
        run["policy_checkpoint_sha256"]
        == validation["policy_checkpoint_sha256"]
        == teachers["policy_checkpoint_sha256"]
        == policy_hash
    )
    assert len(hash_calls) == 2


@pytest.mark.parametrize("script", ["generator", "scorer", "builder"])
def test_cli_does_not_allow_abbreviated_output_override(
    tmp_path: Path, monkeypatch, script: str
) -> None:
    if script == "generator":
        module = _load_generator(monkeypatch)
        args = [
            str(GENERATOR_PATH),
            "--manifest",
            "manifest.jsonl",
            "--dataset-path",
            "data",
            "--model-path",
            "model",
            "--output-dir",
            "safe",
        ]
        main = module.main
    elif script == "scorer":
        module = _load_scorer(monkeypatch)
        args = [str(SCORER_PATH), "--output-dir", "safe"]
        main = module.main
    else:
        module = _load_module(
            "single_policy_builder", REPO_ROOT / "scripts" / "flare" / "build_group_teacher.py"
        )
        args = [
            "build_group_teacher.py",
            "--source-dir",
            "source",
            "--score-parts-dir",
            "scores",
            "--output-dir",
            "safe",
            "--policy-provenance",
            "provenance.json",
        ]
        main = module.parse_args
    monkeypatch.setattr(sys, "argv", args + ["--output-d", str(tmp_path / "override")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
