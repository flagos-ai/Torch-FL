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

"""Unit coverage for the FlagGems ``vector_norm`` composite installed on DCU.

These run on any host. They cover the *decision* -- whether a call is handed to
FlagGems' own implementation or to the composite that avoids ``dim_compress``'s
transposed copy -- because that decision is pure Python and is where a mistake
would be silent: a gate that is too wide changes numerics for shapes nobody
measured, and a gate that is too narrow just loses the speedup.

The on-hardware numbers are in ``docs/reference/operator-support.md``; the
composite's equivalence against the implementation it replaces is measured in
``tests/manual/`` on a DCU host.
"""

import torch
import pytest

from torch_fl import flagos


# Shapes the wrapper has an opinion about. ``(1, N, 1, 32, 32)`` is the rank-5
# layout of the VAE's activation tensors: the reduced axis is strided, so
# FlagGems' ``dim_compress`` would gather it into a fresh buffer.
VAE_LIKE = (1, 144, 1, 32, 32)
SENTINEL = -12345.0


def _fake_original(x, ord=2, dim=None, keepdim=False, dtype=None):
    """Stands in for ``flag_gems.ops.vector_norm.vector_norm``."""
    return torch.full((1,), SENTINEL, dtype=x.dtype)


@pytest.fixture
def wrapped():
    return flagos._flaggems_vector_norm_wrapper(_fake_original)


def _delegated(out):
    return out.shape == (1,) and bool((out == SENTINEL).all())


class TestVectorNormPermutes:
    """``dim_compress`` only stays a view when the permute is the identity."""

    def test_innermost_dim_needs_no_permute(self):
        x = torch.randn(8, 16)
        assert flagos._vector_norm_permutes(x, [1]) is False

    def test_transposed_input_needs_no_permute(self):
        # ``.t()`` leaves the reduced axis at stride 1, so dim_compress is a view.
        x = torch.randn(8, 16).t()
        assert flagos._vector_norm_permutes(x, [1]) is False

    def test_strided_dim_needs_a_permute(self):
        x = torch.randn(*VAE_LIKE)
        assert flagos._vector_norm_permutes(x, [1]) is True

    def test_full_reduction_needs_no_permute(self):
        # Every dim is reduced, so the sorted order is the identity.
        x = torch.randn(*VAE_LIKE)
        assert flagos._vector_norm_permutes(x, list(range(x.ndim))) is False

    def test_last_dim_needs_no_permute(self):
        x = torch.randn(2, 64, 32, 32)
        assert flagos._vector_norm_permutes(x, [3]) is False


class TestCompositeGate:
    """Every case the composite does not cover must reach ``original``."""

    @pytest.mark.parametrize(
        "args",
        [
            # ord: only L2 is decomposed.
            (2, [1], True, None),
            (2.0, [1], True, None),
        ],
    )
    def test_l2_on_a_strided_dim_takes_the_composite(self, wrapped, args):
        x = torch.randn(*VAE_LIKE)
        assert not _delegated(wrapped(x, *args))

    @pytest.mark.parametrize(
        "args",
        [
            # innermost axis: FlagGems never permuted here, keep its kernel
            (2, [4], True, None),
            # a full or multi-dim reduction
            (2, None, True, None),
            (2, [0, 1], True, None),
            # other orders
            (1, [1], True, None),
            (3, [1], True, None),
            (float("inf"), [1], True, None),
            # an explicit dtype means FlagGems casts before it reduces, so the
            # composite's x*x would be evaluated in the wrong dtype
            (2, [1], True, torch.float32),
        ],
    )
    def test_everything_else_reaches_the_original(self, wrapped, args):
        x = torch.randn(*VAE_LIKE)
        assert _delegated(wrapped(x, *args))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_inputs_reach_the_original(self, wrapped, dtype):
        # x*x would round -- and for fp16 above 256, overflow -- before
        # FlagGems' fp32 accumulator sees it.
        x = torch.randn(*VAE_LIKE, dtype=dtype)
        assert _delegated(wrapped(x, 2, [1], True, None))

    def test_negative_dim_is_resolved(self, wrapped):
        x = torch.randn(*VAE_LIKE)
        assert not _delegated(wrapped(x, 2, -2, True, None))

    def test_bare_int_dim_is_accepted(self, wrapped):
        x = torch.randn(*VAE_LIKE)
        assert not _delegated(wrapped(x, 2, 1, True, None))

    def test_float64_is_accepted(self, wrapped):
        x = torch.randn(*VAE_LIKE, dtype=torch.float64)
        assert not _delegated(wrapped(x, 2, [1], True, None))


class TestCompositeResult:
    """The composite has to agree with the reduction it replaces."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_matches_torch_reference(self, wrapped, dtype, keepdim):
        x = torch.randn(*VAE_LIKE, dtype=dtype)
        got = wrapped(x, 2, [1], keepdim, None)
        ref = torch.linalg.vector_norm(x, 2, [1], keepdim)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_keepdim_false_drops_the_reduced_axis(self, wrapped):
        x = torch.randn(*VAE_LIKE)
        assert wrapped(x, 2, [1], False, None).shape == (1, 1, 32, 32)

    def test_nonzero_dim_keeps_the_other_axes(self, wrapped):
        x = torch.randn(2, 4, 8)
        got = wrapped(x, 2, [1], False, None)
        assert got.shape == (2, 8)
        assert torch.allclose(
            got, torch.linalg.vector_norm(x, 2, [1], False), rtol=1e-5
        )


class TestWrapperShape:
    """The wrapper has to look enough like the function it replaces."""

    def test_marks_itself(self, wrapped):
        assert wrapped._torch_fl_composite_l2 is True

    def test_keeps_the_original_reachable(self, wrapped):
        assert wrapped.__wrapped__ is _fake_original
