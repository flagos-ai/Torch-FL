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

"""
clamp dispatch tests

Guards the ``optional<Tensor>`` boxing in the generated in-place kernels.

``clamp_.Tensor`` takes ``optional<Tensor> min`` / ``optional<Tensor> max``.
``DeviceBoxingGuard`` only rewrites the tensors it is handed, so an unboxed
``min``/``max`` reaches a CUDA ``self`` still carrying a flagos device -- which
crashes rather than failing cleanly. ``scripts/codegen/codegen_ops.py`` materializes each
optional into a holder (``min_t``/``max_t``) before the guard; these tests hold
that in place across regenerations.

Usage:
    pytest tests/integration/ops/test_clamp_dispatch.py -v
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _bounds(shape, lo, hi, device):
    """min/max tensors broadcastable against ``shape``."""
    return (
        torch.full(shape, lo, device=device),
        torch.full(shape, hi, device=device),
    )


class TestClampScalarBounds:
    """clamp / clamp_ with Scalar min & max (the ``clamp.Scalar`` overloads)."""

    @pytest.mark.anyplatform
    def test_clamp_scalar_both(self):
        torch.manual_seed(0)
        a = torch.randn(64, 64, device=DEVICE)
        out = torch.clamp(a, -0.5, 0.5)
        ref = torch.clamp(a.cpu(), -0.5, 0.5)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_clamp_scalar_one_sided(self, bound):
        """Exercises the ``None`` branch of the optional holder."""
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        kwargs = {bound: 0.25}
        out = torch.clamp(a, **kwargs)
        ref = torch.clamp(a.cpu(), **kwargs)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    def test_clamp_inplace_scalar(self):
        torch.manual_seed(2)
        a = torch.randn(64, 64, device=DEVICE)
        ref = torch.clamp(a.cpu(), -1.0, 1.0)
        ret = a.clamp_(-1.0, 1.0)
        assert ret.data_ptr() == a.data_ptr(), "clamp_ must mutate in place"
        torch.testing.assert_close(a.cpu(), ref, rtol=0, atol=0)


class TestClampTensorBounds:
    """clamp / clamp_ with Tensor min & max -- the optional<Tensor> boxing path."""

    @pytest.mark.anyplatform
    def test_clamp_tensor_both(self):
        torch.manual_seed(3)
        a = torch.randn(64, 64, device=DEVICE)
        lo, hi = _bounds((64, 64), -0.5, 0.5, DEVICE)
        out = torch.clamp(a, lo, hi)
        ref = torch.clamp(a.cpu(), lo.cpu(), hi.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_clamp_tensor_one_sided(self, bound):
        torch.manual_seed(4)
        a = torch.randn(32, 32, device=DEVICE)
        b = torch.full((32, 32), 0.25, device=DEVICE)
        out = torch.clamp(a, **{bound: b})
        ref = torch.clamp(a.cpu(), **{bound: b.cpu()})
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    def test_clamp_inplace_tensor_both(self):
        """The regression itself: unboxed min/max used to core dump here."""
        torch.manual_seed(5)
        a = torch.randn(64, 64, device=DEVICE)
        lo, hi = _bounds((64, 64), -0.5, 0.5, DEVICE)
        ref = torch.clamp(a.cpu(), lo.cpu(), hi.cpu())
        ret = a.clamp_(lo, hi)
        assert ret.data_ptr() == a.data_ptr(), "clamp_ must mutate in place"
        torch.testing.assert_close(a.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_clamp_inplace_tensor_one_sided(self, bound):
        """One optional set, the other empty -- both holder branches at once."""
        torch.manual_seed(6)
        a = torch.randn(32, 32, device=DEVICE)
        b = torch.full((32, 32), 0.25, device=DEVICE)
        ref = torch.clamp(a.cpu(), **{bound: b.cpu()})
        a.clamp_(**{bound: b})
        torch.testing.assert_close(a.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    def test_clamp_tensor_broadcast(self):
        """Row-vector bounds broadcast against a 2-D input."""
        torch.manual_seed(7)
        a = torch.randn(16, 8, device=DEVICE)
        lo = torch.linspace(-1.0, 0.0, 8, device=DEVICE)
        hi = torch.linspace(0.0, 1.0, 8, device=DEVICE)
        out = torch.clamp(a, lo, hi)
        ref = torch.clamp(a.cpu(), lo.cpu(), hi.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    def test_clamp_min_max_ops(self):
        """clamp_min / clamp_max, the single-bound siblings."""
        torch.manual_seed(8)
        a = torch.randn(32, 32, device=DEVICE)
        b = torch.full((32, 32), 0.1, device=DEVICE)

        torch.testing.assert_close(
            torch.clamp_min(a, b).cpu(),
            torch.clamp_min(a.cpu(), b.cpu()),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.clamp_max(a, b).cpu(),
            torch.clamp_max(a.cpu(), b.cpu()),
            rtol=0,
            atol=0,
        )


class TestClampDtypes:
    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
    def test_clamp_inplace_tensor_dtype(self, dtype):
        torch.manual_seed(9)
        if dtype.is_floating_point:
            a = torch.randn(32, 32, device=DEVICE, dtype=dtype)
            lo, hi = -0.5, 0.5
        else:
            a = torch.randint(-10, 10, (32, 32), device=DEVICE, dtype=dtype)
            lo, hi = -3, 3
        lo_t = torch.full((32, 32), lo, device=DEVICE, dtype=dtype)
        hi_t = torch.full((32, 32), hi, device=DEVICE, dtype=dtype)

        ref = torch.clamp(a.cpu(), lo_t.cpu(), hi_t.cpu())
        a.clamp_(lo_t, hi_t)
        assert a.dtype == dtype
        torch.testing.assert_close(a.cpu(), ref, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
