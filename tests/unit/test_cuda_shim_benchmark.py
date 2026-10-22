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
import triton

from torch_fl.accelerator.cuda import _cuda_compat


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


def test_stream_shim_answers_under_the_name_the_vendor_streams_use():
    """`flagos.Stream` reads `.handle` off its delegate; `.cuda_stream` is not it."""
    shim = _cuda_compat._StreamShim(0)
    assert shim.handle == shim.cuda_stream == 0


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
