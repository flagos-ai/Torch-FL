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

"""Iluvatar ``add.Tensor``: the FlagGems route and its complex escape.

``backends_iluvatar.conf`` routes ``add.Tensor`` to ``flaggems``. It used to be
``none``, which cost issue #667's model 545 calls per forward through the
out-of-place -> in-place lowering and then a host crossing (measured
2757.9 -> 2369.4 ms when the route moved).

FlagGems' complex ``add`` is wrong on this chip -- it returns zeros for
complex128 -- so the op cannot simply be handed to it. ``FlagGemsRejectsDtype``
has a USE_ILUVATAR arm that sends a complex operand to the kIluvatar slot, where
``csrc/aten/register.cc`` publishes an out-of-place host round trip. Both halves
need pinning, because each is silent when it breaks: the flaggems half returns
the right shape and dtype with wrong values, and a complex call that loses the
escape returns zeros instead of raising.

What the conf *states* is pinned by ``tests/unit/test_gen_vendor_confs.py``,
which reads the table without a device; every case here is a value assertion,
so the file is collected unchanged on every lane. A conf that does not route
``add.Tensor`` to FlagGems answers through its own kernel or the fallback, and
these cases hold either way.
"""

import pytest
import torch
import torch_fl  # noqa: F401  (registers the flagos backend)

DEVICE = "flagos:0"

pytestmark = [pytest.mark.anyplatform]

#: Dtypes the FlagGems route serves. bfloat16 and float32 are what the model
#: adds; float16 and int64 pin the two neighbours a broadcast call reaches.
FLAGGEMS_DTYPES = [torch.bfloat16, torch.float32, torch.float16, torch.int64]

#: Dtypes that must leave the route. FlagGems returns zeros for complex128 and
#: its complex64 path is the same kernel, so both are asserted.
ESCAPE_DTYPES = [torch.complex64, torch.complex128]


def _make(shape, dtype):
    if dtype in ESCAPE_DTYPES:
        real = torch.randn(shape, device=DEVICE, dtype=torch.float32)
        imag = torch.randn(shape, device=DEVICE, dtype=torch.float32)
        return torch.complex(real, imag).to(dtype)
    if dtype is torch.int64:
        return torch.randint(-50, 50, shape, device=DEVICE, dtype=dtype)
    return torch.randn(shape, device=DEVICE, dtype=dtype)


@pytest.mark.parametrize("dtype", FLAGGEMS_DTYPES)
def test_add_matches_cpu(dtype):
    torch.manual_seed(1)
    left = _make((1, 267, 2048), dtype)
    right = _make((1, 267, 2048), dtype)
    actual = left + right
    torch.accelerator.synchronize()
    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), left.cpu() + right.cpu())


@pytest.mark.parametrize("dtype", FLAGGEMS_DTYPES)
def test_add_with_alpha_matches_cpu(dtype):
    """HF's residual adds pass alpha; the operand is not optional."""
    torch.manual_seed(2)
    left = _make((4, 64), dtype)
    right = _make((4, 64), dtype)
    actual = torch.add(left, right, alpha=2)
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        actual.cpu(), torch.add(left.cpu(), right.cpu(), alpha=2)
    )


@pytest.mark.parametrize("dtype", FLAGGEMS_DTYPES)
def test_add_broadcast_matches_cpu(dtype):
    torch.manual_seed(3)
    left = _make((1, 267, 2048), dtype)
    right = _make((1, 267, 1), dtype)
    actual = left + right
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual.cpu(), left.cpu() + right.cpu())


@pytest.mark.parametrize("dtype", ESCAPE_DTYPES)
def test_complex_add_leaves_the_flaggems_route(dtype):
    """The measured FlagGems defect is zeros; the escape must answer instead."""
    torch.manual_seed(4)
    left = _make((1, 16, 64), dtype)
    right = _make((1, 16, 64), dtype)
    actual = left + right
    torch.accelerator.synchronize()
    expected = left.cpu() + right.cpu()
    torch.testing.assert_close(actual.cpu(), expected)
    assert not torch.equal(actual.cpu(), torch.zeros_like(expected))


@pytest.mark.parametrize("dtype", ESCAPE_DTYPES)
def test_complex_add_with_alpha_leaves_the_flaggems_route(dtype):
    torch.manual_seed(5)
    left = _make((8, 32), dtype)
    right = _make((8, 32), dtype)
    actual = torch.add(left, right, alpha=3)
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        actual.cpu(), torch.add(left.cpu(), right.cpu(), alpha=3)
    )


@pytest.mark.parametrize("dtype", FLAGGEMS_DTYPES)
def test_inplace_add_still_matches_cpu(dtype):
    """add_ keeps its own native route; the out-of-place change must not move it."""
    torch.manual_seed(6)
    left = _make((16, 24), dtype)
    right = _make((16, 24), dtype)
    target = left.clone()
    target.add_(right)
    torch.accelerator.synchronize()
    torch.testing.assert_close(target.cpu(), left.cpu() + right.cpu())
