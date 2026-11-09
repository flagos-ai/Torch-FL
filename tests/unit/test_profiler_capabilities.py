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

"""Unit coverage for the profiler capability table's `memset` row.

`tests/integration/profiler_support.py::capabilities_for_platform` decides
whether the cross-backend profiler suite asserts on `gpu_memset`. The row is
a platform set intersected with the build record carrying boxing kernels, so
it is pure data: it can be checked without an accelerator and without the
integration conftest.

The DCU entry is covered here because it is otherwise only observable as a
green PCIe-free integration lane. The `hygon-boxing` job failed
`test_profiler_parity.py::test_category_coverage` with "missing categories
torch-cuda emits: ['gpu_memset']" precisely because the row answered False on
DCU, which also meant `append_boxing_path_probe()` never appended the
`linalg_lu_factor` call that makes rocBLAS zero its workspace. Dropping the
platform back out of the set would re-open the same failure silently, and the
memset cases would report as skipped rather than failing.

Run: pytest tests/unit/test_profiler_capabilities.py -v
"""

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


@pytest.mark.parametrize(
    "boxing,cuda_torch", [(False, False), (True, False), (False, True)]
)
def test_native_copy_gate_uses_the_wheel_build_not_only_the_torch_package(
    monkeypatch, tmp_path, boxing, cuda_torch
):
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
    monkeypatch.setitem(
        sys.modules,
        "torch_fl",
        SimpleNamespace(
            _env=SimpleNamespace(
                build_kernels=lambda: ["boxing"] if boxing else ["flaggems"]
            )
        ),
    )
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
    if boxing or cuda_torch:
        namespace["native_cuda_copy"]()
    else:
        with pytest.raises(AssertionError, match="require a boxing wheel"):
            namespace["native_cuda_copy"]()


# Platforms the row may claim, and platforms it must never claim. Kept as two
# lists rather than one parametrization so a new platform has to be placed
# deliberately instead of inheriting whichever list is closer.
_MEMSET_PLATFORMS = ["cuda", "metax", "dcu"]
_NEVER_MEMSET_PLATFORMS = [
    "ascend",
    "gcu",
    "iluvatar",
    "kunlunxin",
    "mps",
    "musa",
    "ppu",
    "tsingmicro",
]


def _set_boxing_built(monkeypatch, built):
    monkeypatch.setattr(profiler_support, "_boxing_built", lambda: built)


@pytest.mark.parametrize("platform", _MEMSET_PLATFORMS)
def test_memset_row_follows_the_boxing_build_record(monkeypatch, platform):
    """A boxing wheel claims memset; the same platform without boxing does not.

    The memset the row asserts is the vendor blas zeroing its gemm workspace,
    and that library is reachable only through the generated boxing kernels.
    """
    _set_boxing_built(monkeypatch, True)
    assert profiler_support.capabilities_for_platform(platform).memset is True

    _set_boxing_built(monkeypatch, False)
    assert profiler_support.capabilities_for_platform(platform).memset is False


def test_dcu_boxing_wheel_claims_memset(monkeypatch):
    """The DCU row added for the hygon-boxing lane, stated on its own.

    Tested separately from the parametrization above so a failure names the
    platform this row was actually added for.
    """
    _set_boxing_built(monkeypatch, True)
    capabilities = profiler_support.capabilities_for_platform("dcu")
    assert capabilities.platform == "dcu"
    assert capabilities.memset is True


def test_dcu_vendor_wheel_leaves_memset_off(monkeypatch):
    """`backends_hygon.conf` routes linalg_lu_factor_ex `none`, so no fill.

    The default DCU wheel reaches cpu_fallback for the probe op and makes no
    rocBLAS call, so claiming the capability there would turn a skip into a
    failure on a record the wheel cannot produce.
    """
    _set_boxing_built(monkeypatch, False)
    assert profiler_support.capabilities_for_platform("dcu").memset is False


@pytest.mark.parametrize("platform", _NEVER_MEMSET_PLATFORMS)
def test_memset_row_stays_off_where_the_tracer_emits_no_fill(monkeypatch, platform):
    """Boxing kernels alone are not the signal; the vendor blas call is."""
    _set_boxing_built(monkeypatch, True)
    assert profiler_support.capabilities_for_platform(platform).memset is False


def test_boxing_path_probe_is_a_noop_without_the_capability(monkeypatch):
    """The probe is gated on the same row, so it costs nothing when off.

    `append_boxing_path_probe` imports torch only after the gate, which is what
    lets this check run on a machine with no torch installed.
    """
    _set_boxing_built(monkeypatch, False)
    assert profiler_support.append_boxing_path_probe(platform="dcu") is None
    assert profiler_support.append_boxing_path_probe(platform="cuda") is None
