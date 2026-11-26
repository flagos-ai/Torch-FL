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

"""Unit coverage for the CUDA shim's autotune surface.

FlagGems' autotuner benchmarks every candidate config before it picks one, and
`LibEntry` resolves that benchmark through triton's replay protocol by default.
Replay means `triton.testing.do_bench_cudagraph`, which captures a
`torch.cuda.CUDAGraph` -- a dummy base class in the CPU torch wheel the CUDA shim
exists for, so the capture raises inside `LibEntry`'s per-config try/except. Each
candidate is then logged as "failed to compile" and timed at `inf`, which floods
the terminal and, silently, leaves autotuning with nothing to select from.

These tests need no GPU: they drive the two decisions -- is graph capture
available, and which benchmarker reaches triton -- with the capability stubbed.
"""

import pytest
import torch
import torch_fl
import triton

from torch_fl.accelerator.nvidia import _cuda_compat


class _FakeGraphBase:
    """Stands in for a real `torch._C._CUDAGraph` binding."""


class _FakeDummyGraphBase:
    """Stands in for `torch._utils._dummy_type`'s placeholder."""


# `type()` stamps `__module__` from the module that made the class, and that is
# the whole difference the probe reads: the placeholder is built in torch._utils.
_Binding = type("_CUDAGraph", (_FakeGraphBase,), {"__module__": "torch._C"})
_Placeholder = type(
    "_CUDAGraph", (_FakeDummyGraphBase,), {"__module__": "torch._utils"}
)


def test_device_index_accepts_cuda_device_context():
    """Third-party probes pass ``torch.cuda.device`` context objects."""
    device = torch.cuda.device(3)
    assert _cuda_compat._device_index(device) == 3


def test_graph_capture_is_read_from_the_binding_not_the_attribute(monkeypatch):
    monkeypatch.setattr(torch._C, "_CUDAGraph", _Binding, raising=False)
    assert _cuda_compat.cuda_graph_supported() is True

    monkeypatch.setattr(torch._C, "_CUDAGraph", _Placeholder, raising=False)
    assert _cuda_compat.cuda_graph_supported() is False


def test_this_build_agrees_with_torch_about_graph_capture():
    """The probe must not disagree with torch's own placeholder detection.

    `hasattr(torch._C, "_CUDAGraph")` is True on the CPU wheel too -- torch injects
    the placeholder into `torch._C.__dict__` as soon as `torch.cuda.streams` is
    imported -- so it cannot be the question. A real binding is not a placeholder.
    """
    import torch.cuda.graphs  # noqa: F401

    placeholder = torch._C._CUDAGraph.__module__ == "torch._utils"
    assert _cuda_compat.cuda_graph_supported() is (not placeholder)


@pytest.fixture
def sentinel_benchmarker(monkeypatch):
    """Watch whether `_patch_triton_do_bench` rewrites the graph benchmarker."""

    def sentinel(*args, **kwargs):  # pragma: no cover - identity only
        raise AssertionError("the original graph benchmarker was called")

    monkeypatch.setattr(triton.testing, "do_bench_cudagraph", sentinel, raising=False)
    # restore do_bench as well: the patch under test assigns it directly.
    monkeypatch.setattr(triton.testing, "do_bench", triton.testing.do_bench)
    return sentinel


def test_wall_clock_benchmarker_replaces_the_graph_one_without_graphs(
    monkeypatch, sentinel_benchmarker
):
    monkeypatch.setattr(_cuda_compat, "cuda_graph_supported", lambda: False)
    _cuda_compat._patch_triton_do_bench()

    assert triton.testing.do_bench_cudagraph is not sentinel_benchmarker
    # The replacement has to answer the call the resolver makes, which passes the
    # replay protocol's own keyword arguments.
    result = triton.testing.do_bench_cudagraph(
        lambda: None, rep=10, n_retries=10, quantiles=[0.5]
    )
    assert isinstance(result, float)


def test_graph_benchmarker_is_kept_where_torch_can_capture(
    monkeypatch, sentinel_benchmarker
):
    monkeypatch.setattr(_cuda_compat, "cuda_graph_supported", lambda: True)
    _cuda_compat._patch_triton_do_bench()

    assert triton.testing.do_bench_cudagraph is sentinel_benchmarker


def test_stream_shim_answers_under_the_name_the_vendor_streams_use(monkeypatch):
    """`flagos.Stream` reads `.handle` off its delegate; `.cuda_stream` is not it."""
    monkeypatch.delattr(torch_fl._C, "_get_nvidia_current_raw_stream", raising=False)
    monkeypatch.setattr(_cuda_compat, "_real_stream", lambda index: None)
    shim = _cuda_compat._StreamShim(0)
    assert shim.handle == shim.cuda_stream == 0


def test_cpu_stream_lookup_does_not_reenter_the_shim(monkeypatch):
    monkeypatch.delattr(torch._C, "_cuda_getCurrentStream", raising=False)

    def recurse(*args):
        raise AssertionError("CPU CUDA lookup must not reenter flagos.current_stream")

    monkeypatch.setattr(torch.flagos, "current_stream", recurse)
    assert _cuda_compat._real_stream(0) is None


def test_native_current_and_default_stream_snapshots(monkeypatch):
    current = [101]
    monkeypatch.setattr(
        torch_fl._C,
        "_get_nvidia_current_raw_stream",
        lambda index: current[0] + index,
        raising=False,
    )
    monkeypatch.setattr(
        torch_fl._C,
        "_get_nvidia_default_raw_stream",
        lambda index: 55 + index,
        raising=False,
    )
    first = _cuda_compat._StreamShim(1)
    current[0] = 202
    second = _cuda_compat._StreamShim(1)
    default = _cuda_compat._StreamShim(1, use_default=True)
    assert (first.cuda_stream, second.cuda_stream, default.cuda_stream) == (
        102,
        203,
        56,
    )
    assert first.stream_id == first.cuda_stream


def test_cuda_runtime_is_read_from_the_shim_not_the_missing_symbol(monkeypatch):
    """The vendor branches must not open on a build the shim has wired up.

    `torch._C._cuda_getCurrentStream` reports whether torch was *compiled* with
    CUDA, not whether a CUDA runtime is reachable, so its absence on the CPU wheel
    sent `flagos.Stream` into the Ascend path even with the shim active.
    """
    from torch_fl import flagos as flagos_module

    monkeypatch.delattr(torch._C, "_cuda_getCurrentStream", raising=False)
    monkeypatch.setattr(_cuda_compat, "_patched", True)
    assert flagos_module._has_cuda_runtime() is True

    monkeypatch.setattr(_cuda_compat, "_patched", False)
    assert flagos_module._has_cuda_runtime() is False


def test_cuda_shim_replaces_cpu_wheel_stream_capture_dummy_even_with_stream_base(
    monkeypatch,
):
    """Issue #661: ``torch.cuda.streams`` defines a dummy ``_CudaStreamBase`` on
    ``torch+cpu`` wheels, so ``hasattr(torch._C, '_CudaStreamBase')`` is True.
    ``patch_torch_cuda_for_flagos`` must still replace the dummy stream-capture
    placeholder across ``torch.cuda``, ``torch.cuda.graphs``, ``torch._C``, and
    ``torch.flagos``, returning ``False`` before context initialization and
    querying ``cudaStreamIsCapturing`` / ``cuStreamIsCapturing`` once initialized.
    """
    import ctypes
    import types
    import torch.cuda.graphs

    dummy_stream_base = type("_CudaStreamBase", (), {"__module__": "torch._utils"})

    def _dummy_err(cls, *args, **kwargs):
        raise RuntimeError(f"Tried to instantiate dummy base class {cls.__name__}")

    dummy_capture = type(
        "_cuda_isCurrentStreamCapturing",
        (),
        {"__module__": "torch._utils", "__new__": _dummy_err},
    )

    for attr in (
        "is_available",
        "device_count",
        "current_device",
        "set_device",
        "get_device_properties",
        "get_device_name",
        "get_device_capability",
        "synchronize",
        "_lazy_init",
        "_initialized",
        "_exchange_device",
        "_maybe_exchange_device",
        "current_stream",
        "default_stream",
        "set_stream",
        "Event",
        "Stream",
        "memory_allocated",
        "memory_reserved",
        "memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
        "empty_cache",
        "reset_peak_memory_stats",
        "reset_max_memory_allocated",
        "is_current_stream_capturing",
        "manual_seed",
        "manual_seed_all",
        "default_generators",
    ):
        if hasattr(torch.cuda, attr):
            monkeypatch.setattr(torch.cuda, attr, getattr(torch.cuda, attr))
    monkeypatch.setattr(
        torch.cuda.device, "__init__", torch.cuda.device.__init__, raising=False
    )
    monkeypatch.setattr(
        torch.cuda.graphs,
        "is_current_stream_capturing",
        lambda: torch.cuda.graphs._cuda_isCurrentStreamCapturing(),
    )
    monkeypatch.setattr(
        torch.cuda.graphs, "_cuda_isCurrentStreamCapturing", dummy_capture
    )
    monkeypatch.setattr(torch._C, "_CudaStreamBase", dummy_stream_base, raising=False)
    monkeypatch.setattr(torch._C, "_CUDAGraph", _Placeholder, raising=False)
    monkeypatch.setattr(
        torch._C, "_cuda_isCurrentStreamCapturing", dummy_capture, raising=False
    )
    if hasattr(torch._C, "_cuda_getCurrentRawStream"):
        monkeypatch.setattr(
            torch._C, "_cuda_getCurrentRawStream", torch._C._cuda_getCurrentRawStream
        )
    if hasattr(torch._C, "_cuda_synchronize"):
        monkeypatch.setattr(torch._C, "_cuda_synchronize", torch._C._cuda_synchronize)

    initialized = [False]
    selected_dev = [0]
    fake_flagos = types.SimpleNamespace(
        is_initialized=lambda: initialized[0],
        current_device=lambda: selected_dev[0],
        set_device=lambda idx: selected_dev.__setitem__(0, idx),
        Event=object,
        Stream=object,
        memory_allocated=lambda idx=0: 0,
        memory_reserved=lambda idx=0: 0,
        memory_stats=lambda idx=0: {},
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda idx=0: None,
        manual_seed=lambda seed: None,
        manual_seed_all=lambda seed: None,
        _C=types.SimpleNamespace(
            _get_native_current_raw_stream=lambda idx: 0xCAFE0 + idx
        ),
    )
    monkeypatch.setattr(torch, "flagos", fake_flagos, raising=False)
    monkeypatch.setattr(_cuda_compat, "_patched", False)
    monkeypatch.setattr(_cuda_compat, "is_nvidia_cuda_available", lambda: True)
    monkeypatch.setattr(_cuda_compat, "_patch_triton_do_bench", lambda: None)

    cudart_calls = []
    driver_calls = []
    capture_status = [0]

    class _FakeCudart:
        def cudaStreamIsCapturing(self, stream_ptr, out_status):
            cudart_calls.append(stream_ptr.value)
            ctypes.cast(out_status, ctypes.POINTER(ctypes.c_int))[0] = capture_status[0]
            return 0

    class _FakeDriver:
        def cuDeviceGetCount(self, out_count):
            ctypes.cast(out_count, ctypes.POINTER(ctypes.c_int))[0] = 4
            return 0

        def cuStreamIsCapturing(self, stream_ptr, out_status):
            driver_calls.append(stream_ptr.value)
            ctypes.cast(out_status, ctypes.POINTER(ctypes.c_int))[0] = capture_status[0]
            return 0

    cudart_holder = [_FakeCudart()]
    monkeypatch.setattr(_cuda_compat, "_load_cudart", lambda: cudart_holder[0])
    monkeypatch.setattr(_cuda_compat, "_load_libcuda", lambda: _FakeDriver())

    assert hasattr(torch._C, "_CudaStreamBase")
    assert _cuda_compat.patch_torch_cuda_for_flagos() is True

    # Before context initialization: returns False without querying CUDA APIs.
    assert torch.cuda.is_current_stream_capturing() is False
    assert torch.cuda.graphs.is_current_stream_capturing() is False
    assert torch._C._cuda_isCurrentStreamCapturing() is False
    assert fake_flagos.is_current_stream_capturing() is False
    assert cudart_calls == []
    assert driver_calls == []

    # After context initialization: queries active FlagOS stream via cudart.
    initialized[0] = True
    torch.cuda.set_device(2)
    assert torch.cuda.is_current_stream_capturing() is False
    assert cudart_calls == [0xCAFE2]

    capture_status[0] = 1
    assert torch.cuda.is_current_stream_capturing() is True
    assert torch.cuda.graphs.is_current_stream_capturing() is True

    # Driver fallback when libcudart is absent.
    cudart_holder[0] = None
    assert fake_flagos.is_current_stream_capturing() is True
    assert driver_calls == [0xCAFE2]
