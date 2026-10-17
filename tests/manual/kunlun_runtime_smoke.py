#!/usr/bin/env python3
"""Run the measured Kunlun P800 runtime and CUDA-boxing smoke checks.

This is a manual hardware test, not a pytest test. It intentionally checks the
small runtime contract before broader operator surveys are attempted.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch_fl  # noqa: E402


DEVICE = "flagos:1"


def check(name: str, fn) -> None:
    try:
        value = fn()
    except BaseException as exc:
        print(f"{name}: FAIL ({type(exc).__name__}: {exc})", flush=True)
        raise
    if isinstance(value, torch.Tensor):
        # Tensor formatting performs reductions, including FP64 min on this
        # vendor stack. Log metadata without reading uninitialized storage or
        # requiring operators outside the runtime allocation check.
        value = (
            f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
        )
    print(f"{name}: PASS ({value})", flush=True)


def main() -> None:
    check("device_count", torch_fl.flagos.device_count)
    check("set_device", lambda: torch_fl.flagos.set_device(1))
    check("current_device", torch_fl.flagos.current_device)

    check("allocation", lambda: torch.empty((8, 8), device=DEVICE))
    source = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    device_value = source.to(DEVICE)
    check("host_to_device", lambda: bool(torch.equal(device_value.cpu(), source)))
    check("device_to_host", lambda: bool(torch.equal(device_value.to("cpu"), source)))

    result = torch.mm(device_value, device_value)
    expected = torch.mm(source, source)
    check("boxed_mm", lambda: bool(torch.allclose(result.cpu(), expected)))

    check("stream_create", torch_fl.flagos.Stream)
    check("synchronize", torch_fl.flagos.synchronize)
    check("empty_cache", torch_fl.flagos.empty_cache)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
