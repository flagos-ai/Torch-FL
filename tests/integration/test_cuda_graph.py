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

"""CUDA/HIP graph capture that records flagos-device kernels.

``torch.cuda.CUDAGraph`` / ``torch.cuda.graph`` is the public entry point on
every platform that has a graph implementation: on a wheel linking a vendor
libtorch it is the genuine binding, and on the CPU torch wheel torch_fl rebinds
the dummy to a ctypes-driven implementation. Either way the flagos device
submits through the captured stream, so the recorded graph must reproduce the
eager result and must read the static inputs' current values on each replay.

The capture holds one ``mm``, chosen because ``mm`` is routed on every platform
this file runs on. Its output is allocated inside the graph rather than passed
through ``out=``: not every platform routes ``mm.out``, and a routed-to-``none``
overload raises on a boxing wheel instead of falling back.
"""

import pytest
import torch_fl  # noqa: F401  -- must precede torch: preloads the vendor libtorch
import torch

DEVICE = torch.device("flagos:0")
SIZE = 64


def _graph_supported() -> bool:
    """Whether this wheel and device expose a usable CUDA/HIP graph.

    Constructing the graph is the honest probe: a dummy ``torch.cuda`` class
    raises, and the ctypes implementation raises when no HIP runtime is loaded.
    """
    if torch_fl.flagos.device_count() == 0:
        return False
    try:
        torch.cuda.CUDAGraph()
    except Exception:  # noqa: BLE001 - any failure means "no graph here"
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _graph_supported(), reason="no CUDA/HIP graph implementation for flagos"
)


def _random() -> torch.Tensor:
    return torch.randn(SIZE, SIZE, device=DEVICE)


def _capture_mm(lhs, rhs):
    """Warm up outside capture, then record ``lhs @ rhs`` into a graph."""
    for _ in range(3):
        torch.mm(lhs, rhs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.mm(lhs, rhs)
    torch.cuda.synchronize()
    return graph, captured


def _eager_mm(lhs, rhs) -> torch.Tensor:
    """The eager result on the host, computed while no capture is active."""
    return torch.mm(lhs, rhs).cpu()


def test_replay_reproduces_the_eager_result():
    lhs, rhs = _random(), _random()
    graph, captured = _capture_mm(lhs, rhs)

    expected = _eager_mm(lhs, rhs)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.allclose(captured.cpu(), expected, rtol=1e-3, atol=1e-3)


def test_replay_reads_inputs_written_after_capture():
    lhs, rhs = _random(), _random()
    graph, captured = _capture_mm(lhs, rhs)

    lhs.copy_(_random())
    expected = _eager_mm(lhs, rhs)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.allclose(captured.cpu(), expected, rtol=1e-3, atol=1e-3)


def test_replay_is_repeatable():
    lhs, rhs = _random(), _random()
    graph, _captured = _capture_mm(lhs, rhs)

    graph.replay()
    torch.cuda.synchronize()
    first = _captured.cpu()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_captured.cpu(), first)
