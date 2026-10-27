# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked public-asset staging checks; no network or model downloads."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/flare/stage_public_libero_asset.py"
URL = "https://us.aws.cdn.hf.co/xet-bridge-us/public/file?Signature=PRIVATE_TOKEN_123"
PAYLOAD = b"small mocked public asset"


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("public_libero_asset_stage_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def small_asset(module, monkeypatch):
    asset = {
        **module.ASSETS["spatial_shard1"],
        "size": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    }
    monkeypatch.setitem(module.ASSETS, "spatial_shard1", asset)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**40))
    return asset


def target_path(root, asset):
    return root / asset["path"]


def request_input(module, monkeypatch, asset="spatial_shard1", url=URL):
    monkeypatch.setattr(module.sys, "stdin", io.StringIO(json.dumps({"asset": asset, "url": url})))


def test_fixed_catalog_has_only_pinned_public_assets(module):
    assert set(module.ASSETS) == {"spatial_shard1", "spatial_shard2", "assets_zip"}
    assert module.RESERVE_BYTES == 8 * 1024**3
    expected = {
        "spatial_shard1": (
            4990519232,
            "b7c7314a989ae3b577e3265e3f25c42a018ab4a476ab99e4d0fa61266d38a878",
        ),
        "spatial_shard2": (
            1919980184,
            "ac70555aee124ef20d23473c85dbcfda42dc8d9f859384822e8aef19fd2d45dd",
        ),
        "assets_zip": (
            6395849578,
            "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf",
        ),
    }
    for name, (size, digest) in expected.items():
        asset = module.ASSETS[name]
        assert (asset["size"], asset["sha256"]) == (size, digest)
        assert not Path(asset["path"]).is_absolute()
        assert ".." not in Path(asset["path"]).parts
        assert asset["source"] in {"nvidia/GR00T-N1.7-LIBERO", "datasets/Sylvest/LIBERO-plus"}
        assert asset["revision"] == (
            "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
            if name == "assets_zip"
            else "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://us.aws.cdn.hf.co/path",
        "https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/model.safetensors",
        "https://us.aws.cdn.hf.co.evil.example/path",
        "https://evil.example/us.aws.cdn.hf.co/path",
        "https://user:password@us.aws.cdn.hf.co/path",
        "https://us.aws.cdn.hf.co:443/path",
        "https://us.aws.cdn.hf.co/path#fragment",
        "https://us.aws.cdn.hf.co/path#",
        "https://us.aws.cdn.hf.co/path\nurl = https://evil.example/file",
        'https://us.aws.cdn.hf.co/path"',
        "https://us.aws.cdn.hf.co/path\\suffix",
        "https://us.aws.cdn.hf.co/",
        "https://us.aws.cdn.hf.co/path\x7f",
        "https://us.aws.cdn.hf.co/path\u2603",
    ],
)
def test_rejects_untrusted_or_config_injecting_urls(module, url):
    with pytest.raises(module.StageError, match="InvalidURL"):
        module.read_request(io.StringIO(json.dumps({"asset": "spatial_shard1", "url": url})))


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"asset": "spatial_shard1"},
        {"asset": "spatial_shard1", "url": URL, "token": "PRIVATE"},
    ],
)
def test_rejects_wrong_request_shape(module, payload):
    with pytest.raises(module.StageError, match="InvalidRequest"):
        module.read_request(io.StringIO(json.dumps(payload)))


@pytest.mark.parametrize("name", ["cosmos", "../../file", None, [], "PRIVATE_TOKEN_123"])
def test_rejects_unknown_and_gated_asset_ids(module, name):
    with pytest.raises(module.StageError, match="UnknownAsset"):
        module.read_request(io.StringIO(json.dumps({"asset": name, "url": URL})))


def test_request_size_is_bounded(module):
    with pytest.raises(module.StageError, match="RequestTooLarge"):
        module.read_request(io.StringIO("x" * (module.MAX_REQUEST_BYTES + 1)))


def test_stdin_only_curl_invocation_is_bounded_and_tls_verified(module, monkeypatch, tmp_path):
    def run(argv, **kwargs):
        assert argv[:4] == ["curl", "--disable", "--config", "-"]
        assert URL not in argv
        assert "PRIVATE_TOKEN_123" not in " ".join(argv)
        assert "--insecure" not in argv and "-k" not in argv
        for option, value in [
            ("--proto", "=https"),
            ("--proto-redir", "=https"),
            ("--max-redirs", "0"),
            ("--continue-at", "-"),
            ("--max-filesize", "25"),
            ("--max-time", "0.5"),
            ("--connect-timeout", "0.5"),
        ]:
            assert argv[argv.index(option) + 1] == value
        assert {"--location", "--silent", "--fail"}.issubset(argv)
        assert kwargs == {
            "input": f'url = "{URL}"\n',
            "text": True,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": 0.5,
        }
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    module.fetch_partial(URL, tmp_path / "asset.part", 25, 0.5)


def test_download_is_verified_before_atomic_publication(
    module, small_asset, monkeypatch, tmp_path, capsys
):
    target = target_path(tmp_path, small_asset)

    def run(argv, **kwargs):
        part = Path(argv[argv.index("--output") + 1])
        assert part == target.with_name(target.name + ".part")
        assert part.is_file() and not target.exists()
        part.write_bytes(PAYLOAD)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    request_input(module, monkeypatch)
    assert module.main(["--root", str(tmp_path)]) == 0
    assert target.read_bytes() == PAYLOAD
    assert not target.with_name(target.name + ".part").exists()
    output = capsys.readouterr()
    reports = [json.loads(line) for line in output.out.splitlines()]
    assert [report["status"] for report in reports] == ["start", "completed"]
    assert reports[0]["resume_bytes"] == 0
    assert reports[1]["reused"] is False
    assert "PRIVATE" not in output.out + output.err


def test_partial_download_is_resumed_without_truncation(
    module, small_asset, monkeypatch, tmp_path, capsys
):
    target = target_path(tmp_path, small_asset)
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(PAYLOAD[:5])

    def fetch(url, path, size, timeout):
        assert path.read_bytes() == PAYLOAD[:5]
        with path.open("ab") as stream:
            stream.write(PAYLOAD[5:])

    monkeypatch.setattr(module, "fetch_partial", fetch)
    result = module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert result["verified_bytes"] == len(PAYLOAD)
    assert target.read_bytes() == PAYLOAD
    assert json.loads(capsys.readouterr().out)["resume_bytes"] == 5


@pytest.mark.parametrize("existing", ["target", "complete_part"])
def test_verified_complete_files_do_not_download(
    module, small_asset, monkeypatch, tmp_path, existing
):
    target = target_path(tmp_path, small_asset)
    target.parent.mkdir(parents=True)
    path = target if existing == "target" else target.with_name(target.name + ".part")
    path.write_bytes(PAYLOAD)
    monkeypatch.setattr(module, "fetch_partial", lambda *args: pytest.fail("unexpected download"))
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda path: pytest.fail("unexpected disk check")
    )
    result = module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert result["reused"] is (existing == "target")
    assert target.read_bytes() == PAYLOAD


@pytest.mark.parametrize("payload", [b"short", b"x" * len(PAYLOAD)])
def test_unverified_existing_target_is_never_overwritten(
    module, small_asset, monkeypatch, tmp_path, payload
):
    target = target_path(tmp_path, small_asset)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    monkeypatch.setattr(module, "fetch_partial", lambda *args: pytest.fail("unexpected download"))
    with pytest.raises(module.StageError, match="SizeMismatch|HashMismatch"):
        module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert target.read_bytes() == payload


@pytest.mark.parametrize("payload", [b"short", b"x" * len(PAYLOAD)])
def test_download_size_or_digest_failure_retains_partial(
    module, small_asset, monkeypatch, tmp_path, payload
):
    target = target_path(tmp_path, small_asset)
    monkeypatch.setattr(module, "fetch_partial", lambda url, path, *args: path.write_bytes(payload))
    with pytest.raises(module.StageError, match="SizeMismatch|HashMismatch"):
        module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert not target.exists()
    assert target.with_name(target.name + ".part").read_bytes() == payload


@pytest.mark.parametrize("place", ["root", "parent", "target", "part"])
def test_symlinks_are_refused_without_touching_destination(
    module, small_asset, monkeypatch, tmp_path, place
):
    root = tmp_path / "root"
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "marker"
    marker.write_bytes(b"user data")
    target = target_path(root, small_asset)
    if place == "root":
        root.symlink_to(victim, target_is_directory=True)
    elif place == "parent":
        root.mkdir()
        (root / "checkpoints").symlink_to(victim, target_is_directory=True)
    else:
        target.parent.mkdir(parents=True)
        path = target if place == "target" else target.with_name(target.name + ".part")
        path.symlink_to(marker)
    monkeypatch.setattr(module, "fetch_partial", lambda *args: pytest.fail("unexpected download"))
    with pytest.raises(module.StageError, match="UnsafePath"):
        module.stage_asset("spatial_shard1", URL, root, 5.0)
    assert marker.read_bytes() == b"user data"


@pytest.mark.parametrize("kind", ["directory", "fifo", "hardlink"])
def test_nonregular_or_aliased_partial_is_refused(module, small_asset, monkeypatch, tmp_path, kind):
    target = target_path(tmp_path, small_asset)
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    if kind == "directory":
        part.mkdir()
    elif kind == "fifo":
        os.mkfifo(part)
    else:
        victim = tmp_path / "user_file"
        victim.write_bytes(b"user data")
        os.link(victim, part)
    monkeypatch.setattr(module, "fetch_partial", lambda *args: pytest.fail("unexpected download"))
    with pytest.raises(module.StageError, match="UnsafePath"):
        module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    if kind == "hardlink":
        assert victim.read_bytes() == b"user data"


def test_disk_reserve_uses_remaining_bytes_and_does_not_create_partial(
    module, small_asset, monkeypatch, tmp_path
):
    target = target_path(tmp_path, small_asset)
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=module.RESERVE_BYTES + len(PAYLOAD) - 1),
    )
    monkeypatch.setattr(module, "fetch_partial", lambda *args: pytest.fail("unexpected download"))
    with pytest.raises(module.StageError, match="InsufficientDiskSpace"):
        module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_resumed_asset_accepts_exact_remaining_plus_reserve(
    module, small_asset, monkeypatch, tmp_path
):
    target = target_path(tmp_path, small_asset)
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(PAYLOAD[:5])
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=module.RESERVE_BYTES + len(PAYLOAD) - 5),
    )
    monkeypatch.setattr(module, "fetch_partial", lambda url, path, *args: path.write_bytes(PAYLOAD))
    module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert target.read_bytes() == PAYLOAD


def test_oversized_partial_is_not_replaced(module, small_asset, monkeypatch, tmp_path):
    target = target_path(tmp_path, small_asset)
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(PAYLOAD + b"extra")
    monkeypatch.setattr(module, "fetch_partial", lambda *args: pytest.fail("unexpected download"))
    with pytest.raises(module.StageError, match="PartialTooLarge"):
        module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert part.read_bytes() == PAYLOAD + b"extra"


def test_concurrent_target_creation_cannot_be_clobbered(module, small_asset, monkeypatch, tmp_path):
    target = target_path(tmp_path, small_asset)

    def fetch(url, path, *args):
        path.write_bytes(PAYLOAD)
        target.write_bytes(b"user-created target")

    monkeypatch.setattr(module, "fetch_partial", fetch)
    with pytest.raises(FileExistsError):
        module.stage_asset("spatial_shard1", URL, tmp_path, 5.0)
    assert target.read_bytes() == b"user-created target"
    assert target.with_name(target.name + ".part").read_bytes() == PAYLOAD


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "-inf", "PRIVATE_TOKEN_123"])
def test_invalid_timeout_is_sanitized_before_reading_stdin(
    module, monkeypatch, tmp_path, capsys, timeout
):
    monkeypatch.setattr(
        module.sys, "stdin", SimpleNamespace(read=lambda *args: pytest.fail("stdin read"))
    )
    assert module.main(["--root", str(tmp_path), "--timeout", timeout]) == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "error"
    assert "PRIVATE" not in output.out + output.err


@pytest.mark.parametrize("failure", ["curl", "timeout", "oserror"])
def test_child_failures_hide_urls_and_stderr(
    module, small_asset, monkeypatch, tmp_path, capsys, failure
):
    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 0.5, output=URL, stderr="PRIVATE_TOKEN_123")
        if failure == "oserror":
            raise OSError(URL)
        return SimpleNamespace(returncode=22, stdout=URL, stderr="PRIVATE_TOKEN_123")

    monkeypatch.setattr(module.subprocess, "run", run)
    request_input(module, monkeypatch)
    assert module.main(["--root", str(tmp_path), "--timeout", "0.5"]) == 2
    output = capsys.readouterr()
    report = json.loads(output.out.splitlines()[-1])
    assert report == (
        {"status": "error", "error_type": "CurlFailed", "exit_code": 22}
        if failure == "curl"
        else {
            "status": "error",
            "error_type": "TimeoutExpired" if failure == "timeout" else "OSError",
        }
    )
    assert "PRIVATE" not in output.out + output.err
    assert URL not in output.out + output.err
    assert not target_path(tmp_path, small_asset).exists()


def test_cli_never_echoes_accidentally_supplied_url(module, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        module.sys, "stdin", SimpleNamespace(read=lambda *args: pytest.fail("stdin read"))
    )
    assert module.main(["--root", str(tmp_path), "--url", URL]) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"status": "error", "error_type": "InvalidArguments"}
    assert "PRIVATE" not in output.out + output.err


def test_malformed_stdin_json_is_sanitized(module, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(module.sys, "stdin", io.StringIO(URL))
    assert module.main(["--root", str(tmp_path)]) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"status": "error", "error_type": "JSONDecodeError"}
    assert "PRIVATE" not in output.out + output.err
