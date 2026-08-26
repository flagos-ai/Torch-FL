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

"""Shared contract for PyTorch's lazy math bits on the flagos device.

`torch.conj` and `torch._neg_view` are metadata-only: they set the Conjugate /
Negative bit on the returned tensor and leave storage untouched. Any operator
that reads storage directly -- every raw-memcpy fast path in copy_ops.cc and
contiguous_ops.cc -- therefore has to materialize those bits first, or it
silently returns the unconjugated / unnegated values (#209).

The bits are a platform-neutral property of PyTorch's dispatcher, not of any
vendor runtime, so every backend that reaches those shared copy paths is held to
the same contract here. Complex support is uneven across vendors, so the complex
cases probe the device once and skip where the dtype is unavailable; the
Negative-bit cases use float32 and run everywhere.
"""

import pytest
import torch
import torch_fl  # noqa: F401

pytestmark = pytest.mark.math_bits

# Values are exactly representable, so every assertion below is bit-exact
# (rtol=0, atol=0) and cannot mask a partially-materialized result.
_COMPLEX_INPUT = [-0.4 - 0.6j, 0.25 + 0.5j]
_COMPLEX_CONJ = [-0.4 + 0.6j, 0.25 - 0.5j]
_REAL_INPUT = [-0.4, 0.25]
_REAL_NEG = [0.4, -0.25]


def _complex_supported() -> bool:
    """Report whether the active backend can hold a complex tensor at all."""
    try:
        torch.tensor([1 + 1j], dtype=torch.complex128, device="flagos:0")
    except Exception:
        return False
    return True


requires_complex = pytest.mark.skipif(
    not _complex_supported(),
    reason="backend does not support complex dtypes on the flagos device",
)


@pytest.fixture
def conj_tensor():
    """A lazily-conjugated flagos tensor: Conjugate bit set, storage untouched."""
    torch.flagos.set_device(0)
    base = torch.tensor(_COMPLEX_INPUT, dtype=torch.complex128, device="flagos:0")
    lazy = torch.conj(base)
    assert lazy.is_conj(), "torch.conj must stay lazy for this contract to apply"
    return lazy


@pytest.fixture
def neg_tensor():
    """A lazily-negated flagos tensor: Negative bit set, storage untouched."""
    torch.flagos.set_device(0)
    base = torch.tensor(_REAL_INPUT, dtype=torch.float32, device="flagos:0")
    lazy = torch._neg_view(base)
    assert lazy.is_neg(), "torch._neg_view must stay lazy for this contract to apply"
    return lazy


def _expected_conj():
    return torch.tensor(_COMPLEX_CONJ, dtype=torch.complex128)


def _expected_neg():
    return torch.tensor(_REAL_NEG, dtype=torch.float32)


@pytest.mark.anyplatform
@requires_complex
def test_resolve_conj_materializes_values(conj_tensor):
    resolved = torch.resolve_conj(conj_tensor)

    assert not resolved.is_conj()
    torch.testing.assert_close(resolved.cpu(), _expected_conj(), rtol=0, atol=0)


@pytest.mark.anyplatform
@requires_complex
def test_clone_materializes_conjugate_bit(conj_tensor):
    cloned = conj_tensor.clone()

    assert not cloned.is_conj()
    torch.testing.assert_close(cloned.cpu(), _expected_conj(), rtol=0, atol=0)


@pytest.mark.anyplatform
@requires_complex
def test_contiguous_materializes_conjugate_bit(conj_tensor):
    contig = conj_tensor.contiguous()

    torch.testing.assert_close(contig.cpu(), _expected_conj(), rtol=0, atol=0)


@pytest.mark.anyplatform
@requires_complex
def test_device_to_host_copy_materializes_conjugate_bit(conj_tensor):
    torch.testing.assert_close(conj_tensor.cpu(), _expected_conj(), rtol=0, atol=0)


@pytest.mark.anyplatform
@requires_complex
def test_device_to_device_copy_materializes_conjugate_bit(conj_tensor):
    dst = torch.empty_like(conj_tensor)

    dst.copy_(conj_tensor)

    assert not dst.is_conj()
    torch.testing.assert_close(dst.cpu(), _expected_conj(), rtol=0, atol=0)


@pytest.mark.anyplatform
@requires_complex
def test_explicit_host_copy_materializes_conjugate_bit(conj_tensor):
    dst = torch.empty(len(_COMPLEX_INPUT), dtype=torch.complex128)

    dst.copy_(conj_tensor)

    assert not dst.is_conj()
    torch.testing.assert_close(dst, _expected_conj(), rtol=0, atol=0)


@pytest.mark.anyplatform
@requires_complex
def test_consumer_op_sees_materialized_conjugate(conj_tensor):
    # A plain elementwise consumer: covers the path where an operator resolves
    # the bit through the shared copy helpers rather than resolving it itself.
    result = conj_tensor + 0

    torch.testing.assert_close(result.cpu(), _expected_conj(), rtol=0, atol=0)


@pytest.mark.anyplatform
def test_clone_materializes_negative_bit(neg_tensor):
    cloned = neg_tensor.clone()

    assert not cloned.is_neg()
    torch.testing.assert_close(cloned.cpu(), _expected_neg(), rtol=0, atol=0)


@pytest.mark.anyplatform
def test_device_to_host_copy_materializes_negative_bit(neg_tensor):
    torch.testing.assert_close(neg_tensor.cpu(), _expected_neg(), rtol=0, atol=0)


@pytest.mark.anyplatform
def test_device_to_device_copy_materializes_negative_bit(neg_tensor):
    dst = torch.empty_like(neg_tensor)

    dst.copy_(neg_tensor)

    assert not dst.is_neg()
    torch.testing.assert_close(dst.cpu(), _expected_neg(), rtol=0, atol=0)


@pytest.mark.anyplatform
def test_resolve_neg_materializes_values(neg_tensor):
    resolved = torch.resolve_neg(neg_tensor)

    assert not resolved.is_neg()
    torch.testing.assert_close(resolved.cpu(), _expected_neg(), rtol=0, atol=0)


@pytest.mark.anyplatform
def test_plain_tensor_copy_is_unaffected():
    # The fast paths stay in place for tensors without math bits; this guards
    # against the materialization branch swallowing the common case.
    src = torch.arange(16, dtype=torch.float32, device="flagos:0").reshape(4, 4)
    dst = torch.empty_like(src)

    dst.copy_(src)

    assert not dst.is_conj() and not dst.is_neg()
    torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)
