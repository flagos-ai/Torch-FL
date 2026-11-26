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

"""Shims that must stay out of the way while torch.compile traces.

Both cases surfaced while compiling Qwen-Image-2.1 on MetaX:

* the ``zero_``/``fill_`` data-pointer shim fired on a FakeTensor, which has no
  storage (inductor calls ``zero_`` while lowering, ``ir_node_to_tensor``);
* ``install_raw_stream_getter()`` was re-run by a recompile and captured its own
  getter as the "native" one, so every stream query recursed into itself.
"""

import pytest
import torch_fl  # noqa: F401  -- must precede torch: relinks the vendor libtorch
import torch

from torch_fl import _build_config
from torch_fl.flagos import cuda_graph

needs_metax = pytest.mark.skipif(
    getattr(_build_config, "ACCELERATOR", "") != "metax",
    reason="MetaX installs the tensor-op shims in this test",
)


@needs_metax
def test_zero_and_fill_accept_a_fake_tensor():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        tensor = torch.empty(4, device="cuda:0")
        tensor.zero_()
        tensor.fill_(3)


def test_raw_stream_getter_can_be_installed_twice(monkeypatch):
    """A recompile re-installs the getter; it must not capture itself."""
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", object(), raising=False)
    monkeypatch.setattr(cuda_graph, "_capture_raw_stream", None)
    monkeypatch.setattr(cuda_graph, "_native_raw_stream", None)

    cuda_graph.install_raw_stream_getter()
    native = cuda_graph._native_raw_stream
    cuda_graph.install_raw_stream_getter()

    assert cuda_graph._native_raw_stream is native
    assert cuda_graph._native_raw_stream is not cuda_graph._raw_stream_getter
