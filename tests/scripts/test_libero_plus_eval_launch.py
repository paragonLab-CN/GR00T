# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU launcher regressions; simulator and model scripts are explicitly stubbed."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAMES = ("run_eval_libero_plus_single_model.sh", "eval_libero_plus_client.sh")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def runtime(tmp_path: Path):
    root = tmp_path / "checkout with spaces"
    scripts = root / "scripts" / "flare"
    scripts.mkdir(parents=True)
    for name in SCRIPT_NAMES:
        shutil.copy2(REPO_ROOT / "scripts" / "flare" / name, scripts / name)
    _write(
        root / "gr00t/eval/run_gr00t_server.py",
        """
        import json
        import os
        from pathlib import Path
        import signal
        import sys
        import time

        Path(os.environ["STUB_SERVER_RECORD"]).write_text(json.dumps({
            "argv": sys.argv[1:],
            "backbone": os.environ["GR00T_BACKBONE_MODEL_NAME"],
        }))
        def stop(*args):
            Path(os.environ["STUB_SERVER_STOPPED"]).write_text("stopped")
            sys.exit(0)
        signal.signal(signal.SIGTERM, stop)
        mode = os.environ.get("STUB_SERVER_MODE", "ready")
        if mode == "exit":
            print("stub checkpoint load failed", flush=True)
            sys.exit(23)
        if mode == "ready":
            print("Server ready", flush=True)
        while True:
            time.sleep(0.1)
        """,
    )
    _write(
        root / "gr00t/eval/rollout_policy.py",
        """
        import json
        import os
        from pathlib import Path
        import sys

        Path(os.environ["STUB_CLIENT_RECORD"]).write_text(json.dumps({
            "argv": sys.argv[1:],
            "config": os.environ["LIBERO_CONFIG_PATH"],
            "mujoco_gl": os.environ["MUJOCO_GL"],
        }))
        if os.environ.get("STUB_ROLLOUT_FAILURE"):
            print("stub rollout failed")
            sys.exit(31)
        argv = sys.argv[1:]
        name = argv[argv.index("--env-name") + 1]
        print("results:", (name, [False], {"episode_lengths": [2]}))
        print("success rate:", 0.0)
        """,
    )
    packages = tmp_path / "stub packages"
    _write(packages / "libero/__init__.py", "")
    _write(packages / "libero/libero/__init__.py", "")
    _write(
        packages / "libero/libero/benchmark.py",
        """
        from types import SimpleNamespace

        class Suite:
            def __init__(self, name):
                self.name = name
            def get_task(self, index):
                return SimpleNamespace(
                    name=f"{self.name}_{index}", language="pick up the block"
                )
        def get_benchmark_dict():
            names = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
            return {name: (lambda name=name: Suite(name)) for name in names}
        """,
    )
    classification = {
        name: [{"category": "camera", "difficulty_level": 1, "id": 1}] for name in SUITES
    }
    _write(
        root / "external_dependencies/LIBERO-plus/libero/libero/benchmark/task_classification.json",
        json.dumps(classification),
    )
    client_python = root / "gr00t/eval/sim/LIBERO/libero_plus_uv/.venv/bin/python"
    client_python.parent.mkdir(parents=True)
    client_python.symlink_to(sys.executable)
    server_python = tmp_path / "server runtime" / "python"
    server_python.parent.mkdir()
    server_python.symlink_to(sys.executable)
    bin_dir = tmp_path / "bin"
    uv = bin_dir / "uv"
    _write(
        uv,
        f"""
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import sys

        Path(os.environ["STUB_UV_RECORD"]).write_text(json.dumps(sys.argv[1:]))
        assert sys.argv[1:4] == ["run", "--no-sync", "python"]
        os.execv(sys.executable, [sys.executable, *sys.argv[4:]])
        """,
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    for key in (
        "GR00T_ROOT",
        "SERVER_PY",
        "SERVER_HOST",
        "SERVER_STARTUP_SECONDS",
        "CLIENT_PY",
        "LIBERO_PLUS_PY",
        "LIBERO_PLUS_CONFIG_DIR",
        "LIBERO_CONFIG_PATH",
        "GR00T_BACKBONE_MODEL_NAME",
        "TASK_FILE_SOURCE",
    ):
        env.pop(key, None)
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
        PYTHONPATH=str(packages),
        N_EPISODES="1",
        N_ENVS="1",
        MAX_EPISODE_STEPS="720",
        N_ACTION_STEPS="8",
        SEED="2026",
        DIFFUSION_SEED="2026",
        TASK_LIMIT="1",
        FULL_EVAL="0",
        KEEP_VIDEOS="0",
        RESUME="0",
        EXPECTED_TASK_COUNT="1",
        STUB_SERVER_RECORD=str(tmp_path / "server.json"),
        STUB_SERVER_STOPPED=str(tmp_path / "server.stopped"),
        STUB_CLIENT_RECORD=str(tmp_path / "client.json"),
        STUB_UV_RECORD=str(tmp_path / "uv.json"),
    )
    return root, env, server_python


def _run(runtime, overrides=None):
    root, env, _ = runtime
    env = {**env, **(overrides or {})}
    return subprocess.run(
        [
            "bash",
            str(root / "scripts/flare/run_eval_libero_plus_single_model.sh"),
            str(root / "baseline checkpoint"),
            "baseline_smoke",
            "5556",
        ],
        env=env,
        cwd=root.parent,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _argument(argv, flag):
    return argv[argv.index(flag) + 1]


def test_default_launcher_runs_client_and_aggregates_from_script_root(runtime):
    root, env, _ = runtime
    result = _run(runtime)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _json(Path(env["STUB_UV_RECORD"]))[:3] == ["run", "--no-sync", "python"]
    server = _json(Path(env["STUB_SERVER_RECORD"]))
    assert _argument(server["argv"], "--host") == "127.0.0.1"
    assert _argument(server["argv"], "--port") == "5556"
    assert _argument(server["argv"], "--embodiment-tag") == "LIBERO_PANDA"
    assert "--use-sim-policy-wrapper" in server["argv"]
    assert server["backbone"] == str(root / "checkpoints/Cosmos-Reason2-2B")
    client = _json(Path(env["STUB_CLIENT_RECORD"]))
    for flag, value in (
        ("--n-episodes", "1"),
        ("--n-envs", "1"),
        ("--max-episode-steps", "720"),
        ("--n-action-steps", "8"),
        ("--candidate-k", "1"),
        ("--diffusion-noise-base-seed", "2026"),
        ("--policy-client-host", "127.0.0.1"),
        ("--policy-client-port", "5556"),
    ):
        assert _argument(client["argv"], flag) == value
    assert client["config"] == str(root / "gr00t/eval/sim/LIBERO/libero_plus_config")
    assert client["mujoco_gl"] == "egl"
    output = root / "outputs/eval_libero_plus/baseline_smoke"
    summary = _json(output / "summary_all.json")
    assert summary["total_episodes"] == summary["num_tasks"] == 1
    assert summary["tasks"][0]["episode_successes"] == [False]
    assert summary["overall_success_rate"] == 0.0
    assert (output / "task_success_rates_all.csv").is_file()
    assert Path(env["STUB_SERVER_STOPPED"]).is_file()


def test_explicit_server_interpreter_and_backbone_are_preserved(runtime):
    root, env, server_python = runtime
    backbone = str(root.parent / "authorized backbone")
    result = _run(
        runtime,
        {
            "GR00T_ROOT": str(root),
            "SERVER_PY": str(server_python),
            "SERVER_HOST": "0.0.0.0",
            "GR00T_BACKBONE_MODEL_NAME": backbone,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(env["STUB_UV_RECORD"]).exists()
    server = _json(Path(env["STUB_SERVER_RECORD"]))
    assert server["backbone"] == backbone
    assert _argument(server["argv"], "--host") == "0.0.0.0"


@pytest.mark.parametrize("alias", ["CLIENT_PY", "LIBERO_PLUS_PY", "both"])
def test_explicit_client_runtime_aliases(runtime, alias):
    root, _, _ = runtime
    default = root / "gr00t/eval/sim/LIBERO/libero_plus_uv/.venv/bin/python"
    default.unlink()
    if alias == "both":
        overrides = {"CLIENT_PY": "/missing/python", "LIBERO_PLUS_PY": sys.executable}
    else:
        overrides = {alias: sys.executable}
    result = _run(runtime, overrides)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("alias", ["LIBERO_CONFIG_PATH", "LIBERO_PLUS_CONFIG_DIR", "both"])
def test_explicit_client_config_aliases(runtime, alias):
    root, env, _ = runtime
    config = str(root.parent / "fork config")
    if alias == "both":
        overrides = {"LIBERO_CONFIG_PATH": "/other/config", "LIBERO_PLUS_CONFIG_DIR": config}
    else:
        overrides = {alias: config}
    result = _run(runtime, overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _json(Path(env["STUB_CLIENT_RECORD"]))["config"] == config


def test_dead_server_fails_without_waiting_default_timeout(runtime):
    _, env, _ = runtime
    result = _run(runtime, {"STUB_SERVER_MODE": "exit"})
    assert result.returncode != 0
    assert "Policy server exited before becoming ready" in result.stderr
    assert "stub checkpoint load failed" in result.stderr
    assert not Path(env["STUB_CLIENT_RECORD"]).exists()


def test_startup_timeout_stops_server_and_never_runs_client(runtime):
    _, env, _ = runtime
    result = _run(runtime, {"STUB_SERVER_MODE": "waiting", "SERVER_STARTUP_SECONDS": "1"})
    assert result.returncode != 0
    assert not Path(env["STUB_CLIENT_RECORD"]).exists()
    assert Path(env["STUB_SERVER_STOPPED"]).is_file()


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_startup_timeout_fails_before_server(runtime, value):
    _, env, _ = runtime
    result = _run(runtime, {"SERVER_STARTUP_SECONDS": value})
    assert result.returncode != 0
    assert "SERVER_STARTUP_SECONDS must be a positive integer" in result.stderr
    assert not Path(env["STUB_SERVER_RECORD"]).exists()


def test_client_failure_stops_server_and_does_not_create_success_summary(runtime):
    root, env, _ = runtime
    result = _run(runtime, {"STUB_ROLLOUT_FAILURE": "1"})
    assert result.returncode != 0
    assert "stub rollout failed" in result.stdout
    assert Path(env["STUB_CLIENT_RECORD"]).is_file()
    assert Path(env["STUB_SERVER_STOPPED"]).is_file()
    assert not (root / "outputs/eval_libero_plus/baseline_smoke/summary_all.json").exists()


def test_existing_task_manifest_is_reused(runtime):
    root, _, _ = runtime
    manifest = root.parent / "frozen tasks.tsv"
    _write(manifest, "libero_sim/frozen_task\tpick up the block\tlibero_spatial\tcamera\t1\n")
    result = _run(runtime, {"TASK_FILE_SOURCE": str(manifest)})
    assert result.returncode == 0, result.stdout + result.stderr
    summary = _json(root / "outputs/eval_libero_plus/baseline_smoke/summary_all.json")
    assert summary["tasks"][0]["task"] == "frozen_task"
