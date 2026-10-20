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

"""Apple Silicon (Metal) copies from strided views into dense tensors."""

import math

import pytest
import torch
import torch_fl  # noqa: F401

from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]


def _device_view(host: torch.Tensor) -> torch.Tensor:
    return host.to(DEVICE)


def _assert_same(out: torch.Tensor, host: torch.Tensor) -> None:
    assert out.device.type == "flagos"
    assert out.shape == host.shape
    assert out.is_contiguous()
    assert out.dtype == host.dtype
    assert torch.equal(out.cpu(), host)


VIEWS = [
    pytest.param((4, 6), lambda t: t.t(), id="two-dimensional-transpose"),
    pytest.param((5, 6), lambda t: t[1:, 2:].t(), id="offset-transpose"),
    pytest.param(
        (4, 5, 6),
        lambda t: t[1:3, 1:4, 1:6:2].permute(2, 0, 1),
        id="three-dimensional-narrow",
    ),
    pytest.param((4, 5, 6), lambda t: t[:, :, ::2], id="strided-columns"),
    pytest.param((4, 5, 6), lambda t: t[::2, ::2, 1::2], id="strided-axes"),
    pytest.param(
        (4, 5, 6), lambda t: t.permute(2, 0, 1), id="three-dimensional-permute"
    ),
    pytest.param(
        (4, 5, 6),
        lambda t: t.permute(1, 0, 2)[:, 1:],
        id="offset-permute",
    ),
    pytest.param(
        (4, 5, 6),
        lambda t: t.flip(0).permute(1, 2, 0),
        id="negative-row-stride",
    ),
    pytest.param(
        (4, 5, 6),
        lambda t: t.flip(0).flip(1).permute(2, 0, 1),
        id="negative-axes",
    ),
    pytest.param(
        (4, 5, 6),
        lambda t: t.flip(0).flip(2).permute(2, 0, 1),
        id="negative-edges",
    ),
    pytest.param(
        (4, 5, 6),
        lambda t: t[::2].flip(1).permute(0, 2, 1),
        id="reverse-permute",
    ),
]


@pytest.mark.parametrize(("shape", "make_view"), VIEWS)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.int8,
        torch.uint8,
        torch.int64,
        torch.bool,
    ],
)
def test_contiguous_matches_cpu(shape, make_view, dtype):
    host = torch.arange(math.prod(shape)).reshape(shape)
    host = host % 2 == 0 if dtype == torch.bool else host.to(dtype)
    host_view = make_view(host).contiguous()
    device_view = make_view(host.to(DEVICE))
    assert not device_view.is_contiguous()

    for out in (device_view.contiguous(), device_view.clone()):
        _assert_same(out, host_view)

    in_place = host_view.to(DEVICE)
    in_place.copy_(device_view)
    _assert_same(in_place, host_view)

    dense = torch.empty(
        host_view.shape,
        dtype=host_view.dtype,
        device=DEVICE,
    )
    dense.copy_(device_view)
    _assert_same(dense, host_view)


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda v: v[1:, :], id="storage-offset"),
        pytest.param(lambda v: v[:, 1::2].t(), id="offset-transpose"),
    ],
)
def test_storage_offset_preserves_the_view(transform):
    host = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    source = transform(host.to(DEVICE))
    assert source.storage_offset() > 0
    _assert_same(source.contiguous(), transform(host).contiguous())


def test_rank_zero_bool_and_empty_views():
    parent = torch.arange(9).reshape(3, 3)
    scalar_host = parent[1, 1]
    scalar = parent.to(DEVICE)[1, 1]
    assert scalar.contiguous().cpu().item() == scalar_host.item()
    assert scalar.clone().cpu().item() == scalar_host.item()
    target = scalar_host.to(DEVICE)
    target.copy_(scalar)
    assert target.cpu().item() == scalar_host.item()

    vector_host = torch.arange(9).reshape(9)
    singleton_host = vector_host[1:8:3]
    singleton = vector_host.to(DEVICE)[1:8:3]
    _assert_same(singleton.contiguous(), singleton_host)
    _assert_same(singleton.clone(), singleton_host)
    target = singleton_host.to(DEVICE)
    target.copy_(singleton)
    assert torch.equal(target.cpu(), singleton_host)

    boolean = (torch.arange(12) % 2 == 0).reshape(3, 4).to(DEVICE)[:, ::2]
    assert not boolean.is_contiguous()
    _assert_same(
        boolean.contiguous(),
        (torch.arange(12) % 2 == 0).reshape(3, 4)[:, ::2].contiguous(),
    )

    for host in (
        torch.empty(0, 5),
        torch.empty(3, 0, 4).permute(2, 0, 1),
    ):
        source = _device_view(host)
        _assert_same(source.contiguous(), host.contiguous())
        _assert_same(source.clone(), host.contiguous())


@pytest.mark.parametrize(
    "storage_recipe",
    [
        pytest.param(lambda t: t[:2], id="larger-than-view"),
        pytest.param(lambda t: t[:2, :, 1:], id="interior-slice"),
        pytest.param(lambda t: t[:2].flip(2), id="reversed-view"),
    ],
)
def test_dense_views_read_their_own_storage_and_offset(storage_recipe):
    host = torch.arange(120, dtype=torch.float32).reshape(4, 5, 6)
    view = storage_recipe(host)
    source = storage_recipe(host.to(DEVICE))
    assert source.stride() == view.stride()
    _assert_same(source.clone(), view)


@pytest.mark.parametrize(
    "make_target",
    [
        pytest.param(lambda t: t[0].t(), id="transpose"),
        pytest.param(lambda t: t[:, 1::2], id="strided-columns"),
        pytest.param(lambda t: t.permute(2, 0, 1), id="permute"),
    ],
)
def test_copy_to_strided_target_preserves_bytes_outside_the_view(make_target):
    """The fallback must update exactly the target view's storage bytes."""
    storage = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    host_target = make_target(storage)
    assert not host_target.is_contiguous()
    device_storage = storage.to(DEVICE)
    target = make_target(device_storage)
    source_host = torch.arange(storage.numel(), dtype=torch.float32).reshape(
        storage.shape
    )
    source_view = make_target(source_host)
    source = source_view.to(DEVICE)
    _assert_same(source, source_view.contiguous())
    target.copy_(source)
    assert torch.equal(target.cpu(), source_view)
    assert torch.equal(device_storage.cpu(), storage)


def test_copy_to_strided_target_with_an_interior_storage_offset():
    host_storage = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    target = host_storage.to(DEVICE)[1:5, 1:4].t()
    source = torch.arange(host_storage.numel(), dtype=torch.float32).reshape(
        host_storage.shape
    )
    target.copy_(source.to(DEVICE)[1:5, 1:4].t())
    expected = source[1:5, 1:4].t().clone()
    assert torch.equal(target.cpu(), expected)
