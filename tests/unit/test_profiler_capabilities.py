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

"""Test profiler transfer detection and its CUDA fixture."""

import importlib.util
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INTEGRATION_DIR = _REPO_ROOT / "tests" / "integration"
_PROFILER_SUPPORT_PATH = _INTEGRATION_DIR / "profiler_support.py"


def _load_profiler_support():
    """Import profiler_support.py without the integration pytest plugin.

    The module imports `platform_support` by bare name, which the integration
    conftest puts on `sys.path`; do the same here instead of loading the whole
    plugin machinery.
    """
    integration_dir = str(_INTEGRATION_DIR)
    if integration_dir not in sys.path:
        sys.path.insert(0, integration_dir)
    spec = importlib.util.spec_from_file_location(
        "profiler_support_under_test", _PROFILER_SUPPORT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiler_support = _load_profiler_support()


def _copy_event(name="Memcpy"):
    return {
        "cat": "gpu_memcpy",
        "ph": "X",
        "name": name,
        "dur": 1,
        "args": {"bytes": 64},
    }


def _fake_transfer_profiler(monkeypatch, traces):
    captures, syncs, transfers = [], [], []
    traces = iter(traces)

    class Capture:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps({"traceEvents": next(traces)}))

    def profile(*, activities):
        captures.append(activities)
        return Capture()

    class Host:
        def to(self, device):
            transfers.append(device)
            return self

        def cpu(self):
            transfers.append("cpu")
            return self

    torch = SimpleNamespace(
        profiler=SimpleNamespace(
            profile=profile,
            ProfilerActivity=SimpleNamespace(CPU="CPU", PrivateUse1="PrivateUse1"),
        ),
        flagos=SimpleNamespace(synchronize=lambda: syncs.append(True)),
        randn=lambda *args: Host(),
    )
    monkeypatch.setattr(profiler_support, "_torch_module", lambda: torch)
    return captures, syncs, transfers


@pytest.mark.parametrize("name", ["Memcpy", "Memcpy HtoD", "a renamed copy"])
def test_transfer_detector_counts_activity_category_not_direction_names(
    monkeypatch, name
):
    captures, syncs, _ = _fake_transfer_profiler(monkeypatch, [[_copy_event(name)]])
    runs = []
    assert profiler_support.device_memcpy_count(lambda: runs.append(True)) == 1
    assert captures == [["CPU", "PrivateUse1"]]
    assert len(syncs) == 2 and len(runs) == 2


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"cat": "cpu_op", "ph": "X", "name": "aten::mul"}],
        [{"cat": "privateuse1_runtime", "ph": "X"}],
    ],
)
def test_missing_device_trace_cannot_prove_zero_transfers(monkeypatch, events):
    _fake_transfer_profiler(monkeypatch, [events])
    with pytest.raises(AssertionError, match="no device activities"):
        profiler_support.device_memcpy_count(lambda: None)


def test_zero_transfer_result_requires_a_working_host_transfer_control(monkeypatch):
    kernel = {"cat": "kernel", "ph": "X", "name": "mul_kernel", "dur": 1}
    _, _, transfers = _fake_transfer_profiler(
        monkeypatch, [[_copy_event(), _copy_event()], [kernel]]
    )
    measure = profiler_support.device_copy_profiler.__wrapped__()
    assert transfers == ["flagos:0", "cpu", "flagos:0", "cpu"]
    assert measure(lambda: None) == 0


@pytest.mark.parametrize(
    "events", [[], [_copy_event()], [{"cat": "kernel", "ph": "X", "name": "kernel"}]]
)
def test_invisible_or_partial_positive_control_rejects_measurement(monkeypatch, events):
    _fake_transfer_profiler(monkeypatch, [events])
    with pytest.raises(AssertionError):
        profiler_support.device_copy_profiler.__wrapped__()


@pytest.mark.parametrize("cuda_torch", [False, True])
def test_native_copy_gate_requires_cuda_linked_torch(tmp_path, cuda_torch):
    tree = ast.parse(
        (_INTEGRATION_DIR / "ops/test_dtype_cast_performance.py").read_text()
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "native_cuda_copy"
    )
    function.decorator_list = []
    namespace = {
        "Path": Path,
        "torch": SimpleNamespace(__file__=str(tmp_path / "torch/__init__.py")),
    }
    if cuda_torch:
        lib = tmp_path / "torch/lib/libtorch_cuda.so"
        lib.parent.mkdir(parents=True)
        lib.touch()
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "native copy gate", "exec"
        ),
        namespace,
    )
    if cuda_torch:
        namespace["native_cuda_copy"]()
    else:
        with pytest.raises(AssertionError, match="require CUDA-linked PyTorch"):
            namespace["native_cuda_copy"]()
