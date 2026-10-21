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

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest_plugins = ("profiler_support", "amp_support")


# PPU selects its vendor libtorch before Python loads torch. A test module or
# string subprocess that imports torch first therefore fails before any test can
# skip itself. Check both boundaries here so shared tests retain one import order.
_TEST_ROOT = Path(__file__).resolve().parent
_SOURCE_RULES: ModuleType | None = None


def _tracked_runtime_imports(code) -> list[str]:
    """Imports in source order, including both names in ``import torch_fl, torch``."""
    import dis

    imports: list[str] = []
    for instruction in dis.get_instructions(code):
        if instruction.opname == "IMPORT_NAME" and instruction.argval in (
            "torch",
            "torch_fl",
        ):
            imports.append(instruction.argval)
        elif (
            instruction.opname in ("IMPORT_FROM", "IMPORT_STAR")
            and isinstance(instruction.argval, str)
            and instruction.argval in ("torch", "torch_fl")
        ):
            imports.append(instruction.argval)
    return imports


def _imports_torch_before_torch_fl(imports: list[str]) -> bool:
    return "torch" in imports and (
        "torch_fl" not in imports or imports.index("torch") < imports.index("torch_fl")
    )


def _check_module_import_order(path: str) -> None:
    target = Path(path).resolve()
    try:
        target.relative_to(_TEST_ROOT)
    except ValueError:
        return
    if _SOURCE_RULES is not None:
        _SOURCE_RULES.check_source(path)


def _check_static_imports() -> None:
    """Load the checker as a normal module so every test process can rerun it."""
    global _SOURCE_RULES
    existing = sys.modules.get("integration_rules")
    if existing is not None:
        _SOURCE_RULES = existing
        existing.check_source_files()
        return
    spec = importlib.util.spec_from_file_location(
        "integration_rules", _TEST_ROOT / "integration_rules.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_source_files()
    _SOURCE_RULES = module


def _check_child_import_order(code) -> None:
    if not _imports_torch_before_torch_fl(_tracked_runtime_imports(code)):
        return
    try:
        _SOURCE_RULES.checked_source(code)
    except AssertionError:
        return
    raise AssertionError(
        "a test subprocess imported torch before torch_fl; PPU selects its "
        "vendor libtorch during that import"
    )


_original_module_from_spec = importlib.util.module_from_spec


def _module_from_spec(spec):
    module = _original_module_from_spec(spec)
    if spec.origin and spec.origin.endswith(".py"):
        _check_module_import_order(spec.origin)
    return module


importlib.util.module_from_spec = _module_from_spec

# Backend config selection moved fully into torch_fl._select_backend_config(),
# which reads the build record (the recorded accelerator + the lib/flagos_platform
# marker) and picks backends_<platform>.conf. conftest no longer pre-empts it: a MetaX
# wheel is boxing-only, so torch_fl already resolves backends_metax.conf, and
# hard-coding the file here only duplicated -- and could contradict -- that.


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
    _check_static_imports()

    # Fresh interpreters use the same initializer as the collecting process.
    # Keep its directory on every integration path, including one hard-coded
    # file, so ``python -c`` and new backend helper files share device selection.
    from device_init import device_init_path, prepare_child_imports

    prepare_child_imports()
    current = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [device_init_path(), current] if current else [device_init_path()]
    )
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
    config.addinivalue_line("markers", "mps: requires Apple Silicon (Metal) hardware")
    config.addinivalue_line("markers", "tsingmicro: requires TsingMicro TX8 hardware")
    # `dcu` is registered in ops/conftest.py too, where the per-platform skip
    # table reads it. A DCU-only contract at the top level of tests/integration/
    # (test_comm_native_bridge.py) needs it registered here as well: ops/conftest
    # is loaded only when something under ops/ is collected.
    config.addinivalue_line("markers", "dcu: requires Hygon DCU hardware")
    # The multi-device contract files sit at the top level of tests/integration/,
    # where nothing sweeps by default: every manifest names what it runs, so a
    # file added there is picked up by no CI job (issue #391). They are selected
    # by this marker instead of by filename, so one identical manifest step works
    # on every platform and the 2+-device guard stays inside the test.
    config.addinivalue_line(
        "markers",
        "multi_device: asserts a multi-device (2+ devices) contract "
        "(select with -m multi_device)",
    )
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
