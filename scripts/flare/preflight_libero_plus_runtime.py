#!/usr/bin/env python3
"""Bounded GPU-worker inventory, not model inference or closed-loop evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import ssl
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request


PROBE_PREFIX = "GR00T_PREFLIGHT_JSON:"
NETWORK_SOURCES = (
    (
        "cosmos_config",
        "https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/9ce19a195e423419c349abfc86fd07178b230561/config.json",
    ),
    (
        "gr00t_libero_spatial_config",
        "https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/resolve/2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/libero_spatial/config.json",
    ),
    (
        "libero_plus_assets",
        "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/dd2bd61b7d9a6fef1abc52d606e983b41886a149/assets.zip",
    ),
)
TORCH_PROBE = """
import torch
result = {
    "torch": str(torch.__version__), "cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()), "gpu_model": None,
    "cuda_matmul_passed": False, "status": "incomplete",
}
if result["cuda_available"]:
    result["gpu_model"] = torch.cuda.get_device_name(0)
    try:
        value = torch.ones((2, 2), device="cuda:0")
        product = value @ value
        torch.cuda.synchronize()
        result["cuda_matmul_passed"] = bool(product.sum().item() == 8.0)
        result["status"] = "passed" if result["cuda_matmul_passed"] else "failed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__)
"""
LIBRARY_PROBE = """
import ctypes.util
result = {
    "status": "passed", "presence_only": True,
    "libEGL_present": ctypes.util.find_library("EGL") is not None,
    "libOpenGL_present": ctypes.util.find_library("OpenGL") is not None,
}
"""
RENDER_PROBE = """
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import mujoco
model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
<light pos="0 0 3"/><geom type="plane" size="1 1 .1"/>
<geom type="box" size=".1 .1 .1" pos="0 0 .1" rgba="1 .2 .1 1"/>
</worldbody></mujoco>''')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=64, width=64)
try:
    renderer.update_scene(data)
    frame = renderer.render()
    passed = bool(frame.shape == (64, 64, 3) and int(frame.max()) > 0)
    result = {
        "status": "passed" if passed else "failed", "offscreen_passed": passed,
        "renderer": "mujoco_egl_synthetic_box", "frame_shape": list(frame.shape),
    }
finally:
    renderer.close()
"""


def run_probe(code: str, timeout: float) -> dict:
    wrapped = (
        "import json\ntry:\n"
        + textwrap.indent(textwrap.dedent(code).strip(), "    ")
        + "\nexcept Exception as error:\n"
        + "    result = locals().get('result', {})\n"
        + "    result.update(status='failed', error_type=type(error).__name__)\n"
        + f"print({PROBE_PREFIX!r} + json.dumps(result))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", wrapped], capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        return {"status": "failed", "error_type": type(error).__name__}
    if completed.returncode != 0:
        return {
            "status": "failed",
            "error_type": "ProbeProcessFailed",
            "exit_code": completed.returncode,
        }
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(PROBE_PREFIX):
            try:
                result = json.loads(line[len(PROBE_PREFIX) :])
                if isinstance(result, dict) and "status" in result:
                    return result
            except (TypeError, ValueError):
                break
    return {"status": "failed", "error_type": "InvalidProbeOutput"}


class HTTPSHeadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.URLError("HTTPS redirect required")
        # Preserve HEAD and do not forward credentials or signed redirect URLs into logs.
        return urllib.request.Request(newurl, method="HEAD")


def network_probe(source: str, url: str, timeout: float) -> dict:
    result = {
        "source": source,
        "hostname": urllib.parse.urlsplit(url).hostname,
        "method": "HEAD",
        "tls_verification_required": True,
        "tls_verified": False,
        "http_status": None,
        "content_type": None,
    }
    try:
        context = ssl.create_default_context()
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context), HTTPSHeadRedirect()
        )
        with opener.open(urllib.request.Request(url, method="HEAD"), timeout=timeout) as response:
            result.update(
                status="passed",
                http_status=response.status,
                tls_verified=True,
                content_type=response.headers.get_content_type(),
            )
    except urllib.error.HTTPError as error:
        result.update(
            status="http_error", http_status=error.code, tls_verified=True, error_type="HTTPError"
        )
    except (urllib.error.URLError, OSError, ValueError) as error:
        result.update(status="failed", error_type=type(error).__name__)
    return result


def package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def collect_report(check_network=False, network_timeout=10.0, probe_timeout=45.0) -> dict:
    packages = {
        name: package_available(name)
        for name in ("torch", "mujoco", "libero", "transformers", "zmq", "huggingface_hub")
    }
    runtime = {
        "python": sys.version.split()[0],
        "torch": None,
        "cuda_version": None,
        "cuda_available": False,
        "gpu_model": None,
        "cuda_matmul_passed": False,
        **run_probe(TORCH_PROBE, probe_timeout),
    }
    libraries = {"presence_only": True, **run_probe(LIBRARY_PROBE, min(probe_timeout, 5.0))}
    renderer = {"status": "unavailable", "offscreen_passed": False}
    if packages["mujoco"]:
        renderer.update(run_probe(RENDER_PROBE, probe_timeout))
    network = (
        [network_probe(source, url, network_timeout) for source, url in NETWORK_SOURCES]
        if check_network
        else []
    )
    passed = (
        runtime.get("status") == "passed"
        and runtime["cuda_available"]
        and runtime["cuda_matmul_passed"]
        and renderer.get("offscreen_passed", False)
        and libraries.get("status") == "passed"
        and all(check.get("status") == "passed" for check in network)
    )
    return {
        "status": "PRECHECK_PASSED" if passed else "PRECHECK_INCOMPLETE",
        "inventory_collected": True,
        "closed_loop_completed": False,
        "ready_to_rollout": False,
        "repo_checkpoint_assets_verified": False,
        "runtime_check": runtime,
        "package_available": packages,
        "egl_library_check": libraries,
        "renderer_check": renderer,
        "network_check": {"enabled": bool(check_network), "sources": network},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-network", action="store_true")
    parser.add_argument("--network-timeout", type=float, default=10.0)
    parser.add_argument("--probe-timeout", type=float, default=45.0)
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.network_timeout, args.probe_timeout)
    ):
        parser.error("timeouts must be finite positive seconds")
    report = collect_report(args.check_network, args.network_timeout, args.probe_timeout)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except OSError as error:
            report.update(status="PRECHECK_INCOMPLETE", output_error_type=type(error).__name__)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PRECHECK_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
