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

"""Native views must preserve storage under the default non-boxing NVIDIA build."""

import contextlib

import pytest
import torch
import torch_fl  # noqa: F401

# The repository's `cuda` mark requires boxing; this regression exercises the
# default NVIDIA build, where there is deliberately no boxing kernel set.
pytestmark = [
    pytest.mark.anyplatform,
    pytest.mark.nvidia,
]

CASES = [
    ("transpose", lambda x: x.transpose(-1, -2)),
    ("unsqueeze", lambda x: x.unsqueeze(-1)),
    ("select", lambda x: x.select(-1, -1)),
    ("t", lambda x: x.t()),
    ("detach", lambda x: x.detach()),
    ("unsafe_view", lambda x: torch.ops.aten._unsafe_view(x, list(x.shape))),
]


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("name,operation", CASES, ids=[c[0] for c in CASES])
def test_views_preserve_offset_strides_and_storage(name, operation, inference):
    # Construct the non-contiguous input through an already-native primitive,
    # so the view under test cannot accidentally create its own valid input.
    host = torch.arange(40, dtype=torch.float32)
    device = host.to("flagos:0")
    ref = host.as_strided((3, 4), (7, 1), 2)
    value = device.as_strided((3, 4), (7, 1), 2)
    mode = torch.inference_mode() if inference else contextlib.nullcontext()
    with mode:
        expected = operation(ref)
        actual = operation(value)
    assert actual.device == value.device
    assert actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("shape", [(), (0,), (0, 4), (3, 0)])
@pytest.mark.parametrize("name", ["t", "unsqueeze", "detach", "unsafe_view"])
def test_scalar_and_empty_views(shape, name):
    operation = dict(CASES)[name]
    host = torch.empty(shape)
    value = host.to("flagos:0")
    actual, expected = operation(value), operation(host)
    assert actual.device == value.device
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "shape,operation,match",
    [
        ((2, 3, 4), lambda x: x.t(), "expects a tensor"),
        ((), lambda x: x.select(0, 0), "select"),
        ((3, 4), lambda x: x.select(1, 4), "out of range"),
        ((3, 4), lambda x: x.unsqueeze(4), "Dimension out of range"),
        ((3, 4), lambda x: x.transpose(0, 3), "Dimension out of range"),
    ],
)
def test_invalid_metadata_matches_cpu_errors(shape, operation, match):
    host = torch.empty(shape)
    value = host.to("flagos:0")
    with pytest.raises((RuntimeError, IndexError), match=match):
        operation(host)
    with pytest.raises((RuntimeError, IndexError), match=match):
        operation(value)


def test_unsafe_view_rejects_materialization():
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    value = host.to("flagos:0")
    for tensor in (host, value):
        noncontiguous = tensor.as_strided((4, 3), (1, 4))
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            torch.ops.aten._unsafe_view(noncontiguous, [12])


def test_detach_preserves_storage_and_shared_version_counter():
    value = torch.arange(12, dtype=torch.float32).to("flagos:0").requires_grad_(True)
    detached = value.detach()
    assert not detached.requires_grad
    assert detached.grad_fn is None
    assert detached.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    version = value._version
    torch.autograd.graph.increment_version(value)
    assert value._version == detached._version == version + 1


@pytest.mark.parametrize("name", ["transpose", "unsqueeze", "select", "t"])
def test_view_backward_records_autograd(name):
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_(True)
    value = host.detach().to("flagos:0").requires_grad_(True)
    operation = dict(CASES)[name]
    expected, actual = operation(host), operation(value)
    assert actual.requires_grad
    assert actual.grad_fn is not None
    expected.backward(torch.ones_like(expected))
    actual.backward(torch.ones(expected.shape).to("flagos:0"))
    torch.testing.assert_close(value.grad.cpu(), host.grad)
