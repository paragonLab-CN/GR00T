#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage one pinned public LIBERO asset from an official CDN URL supplied on stdin."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from urllib.parse import urlsplit


CDN_HOST = "us.aws.cdn.hf.co"
RESERVE_BYTES = 8 * 1024**3
MAX_REQUEST_BYTES = 65536
ASSETS = {
    "spatial_shard1": {
        "path": "checkpoints/GR00T-N1.7-LIBERO/libero_spatial/model-00001-of-00002.safetensors",
        "size": 4990519232,
        "sha256": "b7c7314a989ae3b577e3265e3f25c42a018ab4a476ab99e4d0fa61266d38a878",
        "source": "nvidia/GR00T-N1.7-LIBERO",
        "revision": "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
    },
    "spatial_shard2": {
        "path": "checkpoints/GR00T-N1.7-LIBERO/libero_spatial/model-00002-of-00002.safetensors",
        "size": 1919980184,
        "sha256": "ac70555aee124ef20d23473c85dbcfda42dc8d9f859384822e8aef19fd2d45dd",
        "source": "nvidia/GR00T-N1.7-LIBERO",
        "revision": "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
    },
    "assets_zip": {
        "path": "downloads/LIBERO-plus/assets.zip",
        "size": 6395849578,
        "sha256": "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf",
        "source": "datasets/Sylvest/LIBERO-plus",
        "revision": "dd2bd61b7d9a6fef1abc52d606e983b41886a149",
    },
}


class StageError(Exception):
    def __init__(self, error_type: str, exit_code: int | None = None):
        self.error_type = error_type
        self.exit_code = exit_code
        super().__init__(error_type)


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's usual error can echo an accidentally supplied signed URL.
        raise StageError("InvalidArguments")


def emit(status: str, **fields) -> None:
    print(json.dumps({"status": status, **fields}, sort_keys=True), flush=True)


def read_request(stream) -> tuple[str, str]:
    raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise StageError("RequestTooLarge")
    request = json.loads(raw)
    if not isinstance(request, dict) or set(request) != {"asset", "url"}:
        raise StageError("InvalidRequest")
    asset_id, url = request["asset"], request["url"]
    if not isinstance(asset_id, str) or asset_id not in ASSETS:
        raise StageError("UnknownAsset")
    if not isinstance(url, str):
        raise StageError("InvalidURL")
    if any(ord(char) <= 32 or ord(char) >= 127 or char in '\\"#' for char in url):
        raise StageError("InvalidURL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != CDN_HOST
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or parsed.fragment
    ):
        raise StageError("InvalidURL")
    return asset_id, url


def ensure_directory(path: Path) -> None:
    if ".." in path.parts:
        raise StageError("UnsafePath")
    for current in reversed((path, *path.parents)):
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise StageError("UnsafePath")


def regular_stat(path: Path, *, single_link=False):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or (single_link and info.st_nlink != 1):
        raise StageError("UnsafePath")
    return info


def same_file(path: Path, stream) -> None:
    info = regular_stat(path)
    opened = os.fstat(stream.fileno())
    if info is None or (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
        raise StageError("FileChanged")


def verify_stream(stream, asset: dict) -> None:
    before = os.fstat(stream.fileno())
    if before.st_size != asset["size"]:
        raise StageError("SizeMismatch")
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise StageError("FileChanged")
    if digest.hexdigest() != asset["sha256"]:
        raise StageError("HashMismatch")


def verify_target(path: Path, asset: dict) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        same_file(path, stream)
        verify_stream(stream, asset)
        same_file(path, stream)


@contextmanager
def open_partial(path: Path):
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        regular_stat(path, single_link=True)
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        regular_stat(path, single_link=True)
        same_file(path, stream)
        yield stream


def fetch_partial(url: str, path: Path, size: int, timeout: float) -> None:
    # Disable curlrc and all redirects: this is already the resolved official CDN URL.
    command = [
        "curl",
        "--disable",
        "--config",
        "-",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--location",
        "--max-redirs",
        "0",
        "--fail",
        "--silent",
        "--continue-at",
        "-",
        "--max-filesize",
        str(size),
        "--max-time",
        str(timeout),
        "--connect-timeout",
        str(min(timeout, 30.0)),
        "--output",
        str(path),
    ]
    completed = subprocess.run(
        command,
        input=f'url = "{url}"\n',
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise StageError("CurlFailed", completed.returncode)


def promote_partial(path: Path, target: Path, stream) -> None:
    same_file(path, stream)
    # Same-directory hard-link publication is atomic and never clobbers another target.
    os.link(path, target, follow_symlinks=False)
    same_file(path, stream)
    path.unlink()


def stage_asset(asset_id: str, url: str, root: Path, timeout: float) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise StageError("InvalidTimeout")
    asset = ASSETS[asset_id]
    if ".." in root.parts:
        raise StageError("UnsafePath")
    target = root.absolute() / asset["path"]
    ensure_directory(target.parent)
    if regular_stat(target) is not None:
        verify_target(target, asset)
        return {"asset": asset_id, "verified_bytes": asset["size"], "reused": True}

    part = target.with_name(target.name + ".part")
    info = regular_stat(part, single_link=True)
    resume_bytes = info.st_size if info is not None else 0
    if resume_bytes > asset["size"]:
        raise StageError("PartialTooLarge")
    remaining = asset["size"] - resume_bytes
    if remaining and shutil.disk_usage(target.parent).free < remaining + RESERVE_BYTES:
        raise StageError("InsufficientDiskSpace")

    with open_partial(part) as stream:
        resume_bytes = os.fstat(stream.fileno()).st_size
        if resume_bytes > asset["size"]:
            raise StageError("PartialTooLarge")
        remaining = asset["size"] - resume_bytes
        if remaining:
            if shutil.disk_usage(target.parent).free < remaining + RESERVE_BYTES:
                raise StageError("InsufficientDiskSpace")
            emit("start", asset=asset_id, expected_bytes=asset["size"], resume_bytes=resume_bytes)
            fetch_partial(url, part, asset["size"], timeout)
        same_file(part, stream)
        verify_stream(stream, asset)
        stream.flush()
        os.fsync(stream.fileno())
        promote_partial(part, target, stream)
    return {"asset": asset_id, "verified_bytes": asset["size"], "reused": False}


def main(argv=None) -> int:
    try:
        parser = SafeArgumentParser(description=__doc__, allow_abbrev=False)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--timeout", type=float, default=900.0)
        args = parser.parse_args(argv)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise StageError("InvalidTimeout")
        asset_id, url = read_request(sys.stdin)
        result = stage_asset(asset_id, url, args.root, args.timeout)
        emit("completed", **result)
        return 0
    except StageError as error:
        fields = {"error_type": error.error_type}
        if error.exit_code is not None:
            fields["exit_code"] = error.exit_code
        emit("error", **fields)
    except Exception as error:
        emit("error", error_type=type(error).__name__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
