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

"""Which graph implementation the flagos device exposes, and under which name.

``torch_fl.flagos.cuda_graph`` selects one of two implementations at import
time:

* the genuine ``torch.cuda.CUDAGraph`` pybind class, when the wheel links a
  vendor libtorch (MetaX maca). The flagos device submits through that same
  binding's capture stream, so nothing has to be reimplemented.
* a ctypes-driven HIP implementation, when the CPU torch wheel's
  ``torch.cuda.CUDAGraph`` is the dummy base class (Hygon DCU). The dummy is
  also rebound in ``torch.cuda`` so upper-layer code can name one API.

Neither branch needs the hardware: the selector reads the class's defining
module, and the alias is a rebinding on the ``torch.cuda`` namespace.
"""

import inspect
import sys
import types

import pytest
import torch_fl  # noqa: F401  -- must precede torch: relinks the vendor libtorch
import torch

import torch_fl.flagos as flagos
from torch_fl.flagos import cuda_graph


class _GenuineBinding:
    """A real pybind class, which torch defines in ``torch._C``."""

    __module__ = "torch._C"


class _DummyBinding:
    """The CPU wheel's placeholder, which torch defines in ``torch._utils``."""

    __module__ = "torch._utils"


@pytest.fixture
def importable_graphs_module(monkeypatch):
    """Make ``import torch.cuda.graphs`` succeed even without a CUDA runtime.

    The selector probes that module before reading the class; on a wheel whose
    ``torch.cuda`` package cannot provide it, every case would take the
    exception branch and prove nothing about the class check.
    """
    monkeypatch.setitem(
        sys.modules, "torch.cuda.graphs", types.ModuleType("torch.cuda.graphs")
    )


def test_selects_the_genuine_binding(monkeypatch, importable_graphs_module):
    monkeypatch.setattr(torch._C, "_CUDAGraph", _GenuineBinding, raising=False)

    assert cuda_graph._cuda_graph_binding_available() is True


def test_rejects_the_cpu_wheels_dummy(monkeypatch, importable_graphs_module):
    monkeypatch.setattr(torch._C, "_CUDAGraph", _DummyBinding, raising=False)

    assert cuda_graph._cuda_graph_binding_available() is False


def test_rejects_a_torch_without_a_CUDAGraph(monkeypatch, importable_graphs_module):
    monkeypatch.delattr(torch._C, "_CUDAGraph", raising=False)

    assert cuda_graph._cuda_graph_binding_available() is False


def test_rejects_a_torch_without_the_graphs_module(monkeypatch):
    # None in sys.modules is how the import system records "this module does
    # not exist"; `import torch.cuda.graphs` then raises ImportError.
    monkeypatch.setitem(sys.modules, "torch.cuda.graphs", None)

    assert cuda_graph._cuda_graph_binding_available() is False


def test_alias_rebinds_the_dummy_to_the_flagos_implementation(
    monkeypatch, importable_graphs_module
):
    dummy = object()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", dummy, raising=False)
    monkeypatch.setattr(torch.cuda, "graph", dummy, raising=False)
    monkeypatch.setattr(torch._C, "_CUDAGraph", _DummyBinding, raising=False)

    flagos._alias_torch_cuda_graph()

    assert torch.cuda.CUDAGraph is cuda_graph.CUDAGraph
    assert torch.cuda.graph is cuda_graph.graph


def test_alias_leaves_a_genuine_binding_untouched(
    monkeypatch, importable_graphs_module
):
    sentinel = object()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", sentinel, raising=False)
    monkeypatch.setattr(torch.cuda, "graph", sentinel, raising=False)
    monkeypatch.setattr(torch._C, "_CUDAGraph", _GenuineBinding, raising=False)

    flagos._alias_torch_cuda_graph()

    assert torch.cuda.CUDAGraph is sentinel
    assert torch.cuda.graph is sentinel


def test_one_api_names_the_selected_implementation():
    """``torch.cuda`` and ``torch_fl.flagos`` expose the same graph objects.

    This is the guarantee the alias exists for: a model can be written against
    ``torch.cuda.CUDAGraph`` and run on either platform.
    """
    assert torch.cuda.CUDAGraph is cuda_graph.CUDAGraph
    assert torch.cuda.graph is cuda_graph.graph


def test_graph_accepts_capture_error_mode_on_either_backend():
    """Both implementations take the keyword ``torch.cuda.graph`` documents."""
    assert "capture_error_mode" in inspect.signature(torch.cuda.graph).parameters
