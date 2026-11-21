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

"""Compare representative flagos operator results with CPU references."""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

# ---------------------------------------------------------------------------
# Correctness: newly-registered ops match CPU reference
# ---------------------------------------------------------------------------


class TestUnaryElementwiseNewOps:
    """functional_pure unary ops that were NOT in the original 71-op conf."""

    # (name, callable, needs_positive_input)
    UNARY = [
        ("tanh", torch.tanh, False),
        ("sigmoid", torch.sigmoid, False),
        ("exp", torch.exp, False),
        ("expm1", torch.expm1, False),
        ("log", torch.log, True),
        ("log2", torch.log2, True),
        ("log1p", torch.log1p, True),
        ("sqrt", torch.sqrt, True),
        ("erf", torch.erf, False),
        ("erfc", torch.erfc, False),
        ("floor", torch.floor, False),
        ("ceil", torch.ceil, False),
        ("round", torch.round, False),
        ("trunc", torch.trunc, False),
        ("sign", torch.sign, False),
        ("relu", torch.relu, False),
        ("tan", torch.tan, False),
        ("sinh", torch.sinh, False),
        ("cosh", torch.cosh, False),
        ("atan", torch.atan, False),
        ("asin", torch.asin, False),
        ("acosh", torch.acosh, True),
        ("reciprocal", torch.reciprocal, True),
        ("frac", torch.frac, False),
    ]

    @pytest.mark.parametrize("name,fn,positive", UNARY, ids=[u[0] for u in UNARY])
    def test_unary_matches_cpu(self, name, fn, positive):
        torch.manual_seed(0)
        base = torch.rand(32, 32) if positive else (torch.rand(32, 32) * 2 - 1)
        if positive:
            base = base + 1.0  # keep input in [1, 2): valid domain for log/sqrt/acosh
        ref = fn(base)
        out = fn(base.to(DEVICE))
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


class TestBinaryElementwiseNewOps:
    """functional_pure binary ops not in the original conf."""

    def test_div_tensor(self):
        torch.manual_seed(1)
        a, b = torch.randn(16, 16), torch.rand(16, 16) + 0.5
        ref = a / b
        out = a.to(DEVICE) / b.to(DEVICE)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_maximum_minimum(self):
        torch.manual_seed(2)
        a, b = torch.randn(64), torch.randn(64)
        for fn in (torch.maximum, torch.minimum):
            ref = fn(a, b)
            out = fn(a.to(DEVICE), b.to(DEVICE))
            torch.testing.assert_close(out.cpu(), ref)

    def test_atan2(self):
        torch.manual_seed(3)
        a, b = torch.randn(32), torch.randn(32)
        ref = torch.atan2(a, b)
        out = torch.atan2(a.to(DEVICE), b.to(DEVICE))
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_addmm(self):
        torch.manual_seed(4)
        m, mat1, mat2 = torch.randn(8, 8), torch.randn(8, 16), torch.randn(16, 8)
        ref = torch.addmm(m, mat1, mat2)
        out = torch.addmm(m.to(DEVICE), mat1.to(DEVICE), mat2.to(DEVICE))
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


class TestReductionNewOps:
    """reductions not in the original conf."""

    def test_prod(self):
        torch.manual_seed(5)
        a = torch.rand(8, 8) + 0.5
        torch.testing.assert_close(
            torch.prod(a.to(DEVICE)).cpu(), torch.prod(a), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("fn", [torch.amax, torch.amin])
    def test_amax_amin(self, fn):
        torch.manual_seed(6)
        a = torch.randn(16, 16)
        ref = fn(a, dim=1)
        out = fn(a.to(DEVICE), dim=1)
        torch.testing.assert_close(out.cpu(), ref)

    def test_cumprod(self):
        torch.manual_seed(7)
        a = torch.rand(4, 8) + 0.5
        ref = torch.cumprod(a, dim=1)
        out = torch.cumprod(a.to(DEVICE), dim=1)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


class TestShapeNewOps:
    """tril/triu/flip and similar structure-preserving ops."""

    @pytest.mark.parametrize("fn", [torch.tril, torch.triu])
    def test_tri(self, fn):
        torch.manual_seed(8)
        a = torch.randn(16, 16)
        torch.testing.assert_close(fn(a.to(DEVICE)).cpu(), fn(a))

    def test_flip(self):
        torch.manual_seed(9)
        a = torch.randn(4, 5)
        ref = torch.flip(a, [0, 1])
        out = torch.flip(a.to(DEVICE), [0, 1])
        torch.testing.assert_close(out.cpu(), ref)


class TestFactoryNewOps:
    """
    factory ops. Two sub-kinds (see gen_factory):
      fill-style    (ones/full) -> at::empty + fill_, exact values
      compute-style (randn/eye/linspace) -> built on CUDA device, then unboxed
    For random ops we can only check shape/dtype/device + statistical sanity,
    not exact values (no cross-device seed parity). eye/linspace are
    deterministic so we check values.
    """

    def test_ones_full_exact(self):
        o = torch.ones(3, 4, device=DEVICE)
        assert o.shape == (3, 4) and o.device.type == "flagos"
        torch.testing.assert_close(o.cpu(), torch.ones(3, 4))
        f = torch.full((2, 5), 3.5, device=DEVICE)
        torch.testing.assert_close(f.cpu(), torch.full((2, 5), 3.5))

    def test_eye_deterministic(self):
        e = torch.eye(5, device=DEVICE)
        assert e.device.type == "flagos"
        torch.testing.assert_close(e.cpu(), torch.eye(5))

    def test_linspace_deterministic(self):
        ls = torch.linspace(0, 1, 11, device=DEVICE)
        assert ls.device.type == "flagos"
        torch.testing.assert_close(
            ls.cpu(), torch.linspace(0, 1, 11), rtol=1e-4, atol=1e-4
        )

    def test_randn_shape_and_stats(self):
        """randn is compute-style: must return the requested shape (not 0-dim)
        with plausible values -- this is the regression guard for the
        gen_factory 0-dim/garbage bug."""
        torch.manual_seed(0)
        r = torch.randn(4096, device=DEVICE)
        assert r.shape == (4096,), f"randn returned wrong shape {r.shape}"
        assert r.device.type == "flagos"
        rc = r.cpu()
        # standard normal: mean ~0, std ~1. Loose bounds to avoid flakiness.
        assert abs(rc.mean().item()) < 0.15, rc.mean().item()
        assert 0.8 < rc.std().item() < 1.2, rc.std().item()
        # not all-zero / not constant
        assert rc.abs().sum().item() > 0
        assert rc.min().item() < rc.max().item()

    def test_rand_shape_and_range(self):
        torch.manual_seed(0)
        r = torch.rand(2048, device=DEVICE)
        assert r.shape == (2048,)
        rc = r.cpu()
        assert rc.min().item() >= 0.0 and rc.max().item() <= 1.0
        assert rc.min().item() < rc.max().item()  # not constant


class TestForeachNewOps:
    """_foreach_* ops beyond the ones in the original conf."""

    def test_foreach_add_list(self):
        torch.manual_seed(0)
        a = [torch.randn(8, device=DEVICE) for _ in range(3)]
        b = [torch.ones(8, device=DEVICE) for _ in range(3)]
        a_ref = [t.cpu().clone() for t in a]
        out = torch._foreach_add(a, b)
        for o, r in zip(out, a_ref):
            torch.testing.assert_close(o.cpu(), r + 1.0)

    def test_foreach_sqrt(self):
        torch.manual_seed(1)
        a = [torch.rand(8, device=DEVICE) + 0.5 for _ in range(3)]
        refs = [torch.sqrt(t.cpu()) for t in a]
        out = torch._foreach_sqrt(a)
        for o, r in zip(out, refs):
            torch.testing.assert_close(o.cpu(), r, rtol=1e-3, atol=1e-3)
