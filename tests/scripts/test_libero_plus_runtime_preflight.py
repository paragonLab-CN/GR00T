# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked preflight regression tests, not evidence of GPU or benchmark evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import ssl
import subprocess
import sys
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/flare/preflight_libero_plus_runtime.py"


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("libero_runtime_preflight_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _mock_runtime(monkeypatch, module, *, torch=None, renderer=None, libraries=None, mujoco=True):
    records = []
    monkeypatch.setattr(module, "package_available", lambda name: name != "mujoco" or mujoco)
    results = {
        module.TORCH_PROBE: {
            "status": "passed",
            "torch": "test-version",
            "cuda_version": "12.8",
            "cuda_available": True,
            "gpu_model": "mock GPU",
            "cuda_matmul_passed": True,
        }
        if torch is None
        else torch,
        module.LIBRARY_PROBE: {
            "status": "passed",
            "presence_only": True,
            "libEGL_present": True,
            "libOpenGL_present": True,
        }
        if libraries is None
        else libraries,
        module.RENDER_PROBE: {
            "status": "passed",
            "offscreen_passed": True,
            "renderer": "mock renderer",
            "frame_shape": [64, 64, 3],
        }
        if renderer is None
        else renderer,
    }

    def probe(code, timeout):
        records.append((code, timeout))
        return results[code].copy()

    monkeypatch.setattr(module, "run_probe", probe)
    return records


def test_passed_inventory_never_claims_rollout_or_checks_network(module, monkeypatch):
    records = _mock_runtime(monkeypatch, module)
    monkeypatch.setattr(module, "network_probe", lambda *args: pytest.fail("unexpected network"))
    report = module.collect_report()
    assert report["status"] == "PRECHECK_PASSED"
    assert report["inventory_collected"] is True
    assert report["closed_loop_completed"] is False
    assert report["ready_to_rollout"] is False
    assert report["repo_checkpoint_assets_verified"] is False
    assert report["runtime_check"]["python"] == sys.version.split()[0]
    assert report["egl_library_check"]["presence_only"] is True
    assert report["network_check"] == {"enabled": False, "sources": []}
    assert records == [
        (module.TORCH_PROBE, 45.0),
        (module.LIBRARY_PROBE, 5.0),
        (module.RENDER_PROBE, 45.0),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"torch": {"status": "incomplete", "cuda_available": False, "cuda_matmul_passed": False}},
        {"torch": {"status": "failed", "error_type": "ImportError"}},
        {"torch": {"status": "passed", "cuda_available": False, "cuda_matmul_passed": True}},
        {"renderer": {"status": "failed", "error_type": "TimeoutExpired"}},
        {"renderer": {"status": "failed", "error_type": "ProbeProcessFailed", "exit_code": -11}},
        {"libraries": {"status": "failed", "error_type": "TimeoutExpired"}},
    ],
)
def test_gpu_missing_or_probe_failure_returns_incomplete(module, monkeypatch, capsys, kwargs):
    _mock_runtime(monkeypatch, module, **kwargs)
    assert module.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PRECHECK_INCOMPLETE"
    assert report["closed_loop_completed"] is False
    assert report["ready_to_rollout"] is False


def test_missing_mujoco_does_not_import_or_start_renderer(module, monkeypatch):
    records = _mock_runtime(monkeypatch, module, mujoco=False)
    report = module.collect_report()
    assert report["renderer_check"] == {"status": "unavailable", "offscreen_passed": False}
    assert report["status"] == "PRECHECK_INCOMPLETE"
    assert module.RENDER_PROBE not in [code for code, _ in records]


def test_library_presence_does_not_replace_failed_render(module, monkeypatch):
    _mock_runtime(monkeypatch, module, renderer={"status": "failed", "offscreen_passed": False})
    report = module.collect_report()
    assert report["egl_library_check"]["libEGL_present"] is True
    assert report["renderer_check"]["offscreen_passed"] is False
    assert report["status"] == "PRECHECK_INCOMPLETE"


def test_package_availability_does_not_import_package(module, monkeypatch):
    calls = []

    def find_spec(name):
        calls.append(name)
        return object() if name == "torch" else None

    monkeypatch.setattr(module.importlib.util, "find_spec", find_spec)
    assert module.package_available("torch") is True
    assert module.package_available("mujoco") is False
    assert calls == ["torch", "mujoco"]


def test_probe_exceptions_do_not_leak_messages(module):
    result = module.run_probe("raise RuntimeError('PRIVATE_TOKEN_123')", timeout=5.0)
    assert result == {"status": "failed", "error_type": "RuntimeError"}
    assert "PRIVATE_TOKEN_123" not in json.dumps(result)


def test_probe_keeps_measured_metadata_on_later_failure(module):
    result = module.run_probe(
        "result = {'torch': 'measured', 'cuda_available': True}\nraise RuntimeError('PRIVATE_TOKEN_123')",
        timeout=5.0,
    )
    assert result == {
        "torch": "measured",
        "cuda_available": True,
        "status": "failed",
        "error_type": "RuntimeError",
    }


def test_network_sources_are_revision_pinned(module):
    revisions = (
        "9ce19a195e423419c349abfc86fd07178b230561",
        "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
        "dd2bd61b7d9a6fef1abc52d606e983b41886a149",
    )
    for (_, url), revision in zip(module.NETWORK_SOURCES, revisions):
        assert f"/resolve/{revision}/" in url
        assert "/main/" not in url


def test_probe_timeout_is_bounded_and_redacted(module, monkeypatch):
    def run(argv, **kwargs):
        assert argv[:2] == [sys.executable, "-c"]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 0.1}
        raise subprocess.TimeoutExpired(argv, 0.1, output="PRIVATE_TOKEN_123", stderr="PRIVATE")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.run_probe("result = {}", 0.1) == {
        "status": "failed",
        "error_type": "TimeoutExpired",
    }


@pytest.mark.parametrize(
    "completed, expected",
    [
        (
            SimpleNamespace(returncode=-11, stdout="PRIVATE_TOKEN", stderr="PRIVATE"),
            "ProbeProcessFailed",
        ),
        (
            SimpleNamespace(returncode=0, stdout="PRIVATE_TOKEN", stderr="PRIVATE"),
            "InvalidProbeOutput",
        ),
        (
            SimpleNamespace(returncode=0, stdout="GR00T_PREFLIGHT_JSON:not-json", stderr=""),
            "InvalidProbeOutput",
        ),
        (
            SimpleNamespace(returncode=0, stdout="GR00T_PREFLIGHT_JSON:[]", stderr=""),
            "InvalidProbeOutput",
        ),
    ],
)
def test_probe_crashes_and_bad_stdout_are_redacted(module, monkeypatch, completed, expected):
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    result = module.run_probe("result = {}", 1)
    assert result["status"] == "failed"
    assert result["error_type"] == expected
    assert "PRIVATE" not in json.dumps(result)


def test_probe_ignores_unstructured_import_logs(module, monkeypatch):
    payload = {"status": "passed", "cuda_matmul_passed": True}
    completed = SimpleNamespace(
        returncode=0,
        stdout="PRIVATE_TOKEN\n" + module.PROBE_PREFIX + json.dumps(payload),
        stderr="PRIVATE_TOKEN",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    assert module.run_probe("result = {}", 1) == payload


def _mock_opener(module, monkeypatch, response=None, error=None):
    def build(*handlers):
        https = next(item for item in handlers if isinstance(item, urllib.request.HTTPSHandler))
        assert https._context.verify_mode == ssl.CERT_REQUIRED
        assert https._context.check_hostname is True
        assert any(isinstance(item, module.HTTPSHeadRedirect) for item in handlers)

        def open(request, *, timeout):
            assert request.get_method() == "HEAD"
            assert request.get_full_url() == module.NETWORK_SOURCES[0][1]
            assert request.get_header("Authorization") is None
            assert timeout == 10.0
            if error is not None:
                raise error
            return response

        return SimpleNamespace(open=open)

    monkeypatch.setattr(module.urllib.request, "build_opener", build)


def test_network_probe_is_verified_anonymous_head(module, monkeypatch):
    class Response:
        status = 200
        headers = SimpleNamespace(get_content_type=lambda: "application/json")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    _mock_opener(module, monkeypatch, response=Response())
    result = module.network_probe(*module.NETWORK_SOURCES[0], 10.0)
    assert result["status"] == "passed"
    assert result["http_status"] == 200
    assert result["tls_verified"] is True
    assert result["hostname"] == "huggingface.co"
    assert result["content_type"] == "application/json"
    assert "url" not in result


def test_gated_http_error_reports_code_not_auth_text(module, monkeypatch):
    error = urllib.error.HTTPError(
        module.NETWORK_SOURCES[0][1], 401, "PRIVATE_TOKEN_123", hdrs=None, fp=None
    )
    _mock_opener(module, monkeypatch, error=error)
    result = module.network_probe(*module.NETWORK_SOURCES[0], 10.0)
    assert result["status"] == "http_error"
    assert result["http_status"] == 401
    assert result["error_type"] == "HTTPError"
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(ssl.SSLCertVerificationError("PRIVATE_TOKEN")),
        TimeoutError("PRIVATE_TOKEN"),
    ],
)
def test_network_failures_do_not_claim_tls_or_leak_errors(module, monkeypatch, error):
    _mock_opener(module, monkeypatch, error=error)
    result = module.network_probe(*module.NETWORK_SOURCES[0], 10.0)
    assert result["status"] == "failed"
    assert result["tls_verified"] is False
    assert "PRIVATE" not in json.dumps(result)


def test_redirects_preserve_head_without_credentials(module):
    original = urllib.request.Request(
        module.NETWORK_SOURCES[0][1],
        headers={"Authorization": "Bearer PRIVATE_TOKEN"},
        method="HEAD",
    )
    handler = module.HTTPSHeadRedirect()
    redirected = handler.redirect_request(
        original, None, 302, "", {}, "https://cdn.example/a?signature=private"
    )
    assert redirected.get_method() == "HEAD"
    assert redirected.get_header("Authorization") is None
    with pytest.raises(urllib.error.URLError):
        handler.redirect_request(original, None, 302, "", {}, "http://cdn.example/a")


def test_explicit_network_check_probes_only_fixed_sources(module, monkeypatch):
    _mock_runtime(monkeypatch, module)
    calls = []

    def network(source, url, timeout):
        calls.append((source, url, timeout))
        return {"status": "http_error", "http_status": 401, "source": source}

    monkeypatch.setattr(module, "network_probe", network)
    report = module.collect_report(check_network=True, network_timeout=3.0)
    assert calls == [(source, url, 3.0) for source, url in module.NETWORK_SOURCES]
    assert report["network_check"]["enabled"] is True
    assert report["status"] == "PRECHECK_INCOMPLETE"


def test_output_json_is_optional_and_never_an_evaluation(module, monkeypatch, tmp_path, capsys):
    _mock_runtime(monkeypatch, module)
    output = tmp_path / "report.json"
    assert module.main(["--output", str(output)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == json.loads(output.read_text())
    assert stdout["closed_loop_completed"] is False
    assert stdout["ready_to_rollout"] is False


def test_existing_output_file_is_not_overwritten(module, monkeypatch, tmp_path, capsys):
    _mock_runtime(monkeypatch, module)
    output = tmp_path / "report.json"
    output.write_text("existing content", encoding="utf-8")
    assert module.main(["--output", str(output)]) == 2
    assert output.read_text() == "existing content"
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["output_error_type"] == "FileExistsError"


@pytest.mark.parametrize("flag", ["--network-timeout", "--probe-timeout"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_fails_without_collecting(module, monkeypatch, flag, value):
    monkeypatch.setattr(module, "collect_report", lambda *args: pytest.fail("unexpected probe"))
    with pytest.raises(SystemExit) as error:
        module.main([flag, value])
    assert error.value.code == 2


def test_standalone_help_requires_only_stdlib():
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--check-network" in result.stdout
