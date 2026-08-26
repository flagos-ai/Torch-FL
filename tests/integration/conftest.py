# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

pytest_plugins = ("profiler_support", "amp_support")


def _ensure_backend_config() -> None:
    """Ensure MetaX backend config is set before importing torch_fl (if not already specified).

    Only forces the hand-written mxcc config (backends_metax.conf) for the
    legacy source-build path. In boxing mode (FLAGOS_METAX_BOXING=1) the mxcc
    backend is NOT compiled, so we must leave the choice to torch_fl's own
    _select_backend_config(), which picks backends_cuda.conf (pure boxing) or
    backends_metax_flaggems.conf (FLAGOS_USE_FLAGGEMS=1). Setting metax.conf here
    would route ops to the unregistered `metax` backend -> "backend not registered".
    """
    if os.environ.get("FLAGOS_BACKEND_CONFIG"):
        return
    if os.environ.get("FLAGOS_METAX_BOXING", "0") == "1":
        return
    accel = os.environ.get("ACCELERATOR", "").lower()
    use_metax = accel in ("metax", "maca") or Path("/dev/mxcd").exists()
    if use_metax:
        cfg = _REPO_ROOT / "torch_fl" / "configs" / "backends_metax.conf"
        if cfg.is_file():
            os.environ["FLAGOS_BACKEND_CONFIG"] = str(cfg)


_ensure_backend_config()


def pytest_addoption(parser):
    parser.addoption("--model", default="Qwen/Qwen3-0.6B", help="Path to Qwen3 model")
    parser.addoption(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for inference tests",
    )
    parser.addoption("--steps", type=int, default=10, help="Training steps")
    parser.addoption(
        "--batch-size", type=int, default=2, help="Batch size for training"
    )
    parser.addoption(
        "--seq-len", type=int, default=1024, help="Sequence length for training"
    )
    parser.addoption(
        "--lr", type=float, default=1e-5, help="Learning rate for training"
    )


def pytest_configure(config):
    # `main_ops` is CI's selection hook (`-m main_ops`). It was registered only in
    # tests/integration/ops/conftest.py, so tests using it directly under
    # tests/integration/ raised PytestUnknownMarkWarning -- and would hard-error
    # under --strict-markers. Registering it here covers the whole directory.
    config.addinivalue_line(
        "markers", "anyplatform: runs on every supported hardware platform"
    )
    config.addinivalue_line(
        "markers",
        "main_ops: representative test in the CI smoke subset "
        "(select with -m main_ops)",
    )
    config.addinivalue_line("markers", "ascend: requires Ascend NPU hardware")
    config.addinivalue_line("markers", "musa: requires Moore Threads MUSA hardware")
    config.addinivalue_line("markers", "gcu: requires Enflame GCU hardware")
    config.addinivalue_line(
        "markers", "profiler: shared public torch.profiler contract"
    )
    config.addinivalue_line(
        "markers", "profiler_device: requires device activity from the profiler"
    )
    config.addinivalue_line(
        "markers", "profiler_kernel: requires profiler kernel activities"
    )
    config.addinivalue_line(
        "markers", "profiler_runtime: requires profiler runtime activities"
    )
    config.addinivalue_line(
        "markers", "profiler_memcpy: requires profiler memcpy activities"
    )
    config.addinivalue_line(
        "markers", "profiler_memset: requires profiler memset activities"
    )
    config.addinivalue_line(
        "markers", "profiler_flow: requires CPU-to-device profiler flows"
    )
    config.addinivalue_line(
        "markers", "profiler_linkage: requires profiler device-time linkage"
    )
    config.addinivalue_line(
        "markers", "profiler_metadata: requires profiler device metadata"
    )
    config.addinivalue_line(
        "markers",
        "math_bits: shared contract for PyTorch's lazy Conjugate/Negative bits",
    )
    config.addinivalue_line("markers", "amp: shared public torch.amp contract")
    config.addinivalue_line(
        "markers", "amp_device: requires AMP compute on the flagos device"
    )
    config.addinivalue_line(
        "markers", "amp_grad_scaler: requires the GradScaler unscale/update routes"
    )

    import torch_fl

    if not torch_fl.flagos.is_available():
        pytest.exit("flagos device is not available.")

    # Initialize flagos device to ensure ACL runtime is properly set up
    torch_fl.flagos.init()
