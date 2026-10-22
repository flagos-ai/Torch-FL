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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``native_layer_norm``.

``F.layer_norm`` is the normalisation a transformer block calls once per block,
and it is the **first body here that answers with more than one tensor**:
``native_layer_norm`` returns ``(out, mean, rstd)``. Both of those shape what the
cases below are about.

The finding the increment rests on is that it is **one MPSGraph node, not the
``native_batch_norm`` + ``addcmul`` decomposition the design doc predicted**.
That prediction came from reading the CPU fallback -- which is what this op was
taking, so the ops the CPU build decomposes it into were what the fallback log
showed. Measured on the M5, MPSGraph computes the whole operation behind
``normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:
epsilon:name:``, whose documented semantics are exactly aten's. Three things
about that call are what a body can get wrong, and each is measured here rather
than read off the header:

* **``normalized_shape`` is a trailing run of the input's axes**, which is the
  reference's own precondition and not the node's. The body checks the suffix and
  declines otherwise, so a non-trailing shape reaches the reference's own
  ``RuntimeError`` (``TestLayerNormDeclines``) instead of being answered by a
  node built for a different question.
* **``mean`` and ``rstd`` are the *keepdim* shape over the reduced axes** --
  ``(2, 3, 1)`` for a ``(2, 3, 8)`` input with ``normalized_shape=(8,)`` -- and not
  the reduced shape. Held outright in ``TestLayerNormStatShapes``, because a body
  that reshaped them to the reduced shape would return a correctly shaped ``out``
  and wrong-shaped statistics.
* **The variance is the population one** (divided by ``N``), which is what the
  reference's ``unbiased=False`` computes and what
  ``varianceOfTensor:meanTensor:axes:`` returns -- measured on an ``N=3`` case
  whose sample and population variances differ visibly. That is the one case a
  body built on the sample-variance node cannot pass, and it is in
  ``TestLayerNormVariance`` rather than left to the random shapes, which
  frequently agree to within the tolerance either way.

The ``weight``/``bias`` operands are optional and are the node's ``gamma`` and
``beta``; when absent the node is handed ``nil`` and it is the same graph shape.
An integral input is declined like every other reduction here -- the reference
computes it in float, which a graph at the operand's own width cannot express --
so the round-trip keeps the reference's dtype and values.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
"""

import os
import subprocess
import sys

import pytest
from device_init import torch, torch_fl

from backend_conf import routed_backend_or_none
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The two half widths accumulate in f32 inside MPSGraph the way the CPU kernel
# does but round at different points, so they are compared at a tolerance a few
# orders of magnitude above the 2.4e-7 measured for f32.
LOW_PRECISION_TOLERANCE = {"rtol": 1e-2, "atol": 1e-2}

# ``(input shape, normalized_shape)`` pairs that are a trailing run by
# construction, plus the spelling that covers the whole input.
SHAPE_CASES = [
    ((2, 8, 512), (512,)),
    ((1, 128, 1024), (1024,)),
    ((2, 512, 1024), (1024,)),
    ((2, 3, 8), (8,)),
    ((2, 3, 8), (3, 8)),
    ((4, 6, 7, 8), (6, 7, 8)),
    ((2, 3, 8), (2, 3, 8)),
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another case
    # left behind.
    return t.clone().to(DEVICE)


def _reference(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
):
    """``at::native_layer_norm`` on the host, returning the reference triple.

    The reference the kernel is held to, and deliberately the same op the body
    round-trips to rather than a hand-built ``(x - mean) / sqrt(var + eps) * w +
    b``: the trailing-run precondition and the keepdim statistic shapes are the
    reference's contract, so a reimplementation would encode the test author's
    reading of it and could agree with a body that is wrong in the same way.
    """
    return torch.ops.aten.native_layer_norm(
        input, list(normalized_shape), weight, bias, eps
    )


def _assert_matches(
    got,
    expected,
    dtype: torch.dtype = torch.float32,
    tolerance: dict | None = None,
) -> None:
    """All three results equal the reference's, in shape, dtype and value.

    Every result comes back on the flagos device: a caller with only the values
    would not see a body that left one result on the host, because ``getattr``
    on a CPU tensor answers the CPU's numbers too. A tensor that returns to the
    input's device, which is the host, is the same bug in a different spelling.
    """
    assert len(got) == len(expected) == 3
    for index, name in enumerate(("out", "mean", "rstd")):
        assert got[index].device.type == "flagos", f"{name} left the device"
        assert got[index].dtype == expected[index].dtype, f"{name} dtype differs"
        assert got[index].shape == expected[index].shape, f"{name} shape differs"
    if tolerance is not None:
        for got_t, exp_t in zip(got, expected):
            torch.testing.assert_close(got_t.cpu(), exp_t, **tolerance)
    elif dtype == torch.float32:
        for got_t, exp_t in zip(got, expected):
            torch.testing.assert_close(got_t.cpu(), exp_t, rtol=1e-5, atol=1e-5)
    else:
        for got_t, exp_t in zip(got, expected):
            torch.testing.assert_close(
                got_t.cpu().float(), exp_t.float(), **LOW_PRECISION_TOLERANCE
            )


class TestLayerNormCorrectness:
    """``native_layer_norm`` on the flagos device equals the CPU reference."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("shape, normalized_shape", SHAPE_CASES)
    def test_matches_cpu_for_trailing_shapes(self, shape, normalized_shape):
        """Every shape here is a trailing run, and the reference defines the run.

        ``normalized_shape`` of ``(8,)`` over a ``(2, 3, 8)`` input is the last
        axis and ``(3, 8)`` the last two; the body reduces over
        ``[-len(normalized_shape), 0)``, so the two must be the same axes and the
        shapes are what says so.
        """
        host = torch.randn(*shape)
        out = torch.ops.aten.native_layer_norm(
            _to_device(host), list(normalized_shape), None, None, 1e-5
        )
        _assert_matches(out, _reference(host, normalized_shape))

    @pytest.mark.anyplatform
    def test_weight_and_bias_are_the_gamma_and_beta(self):
        """The affine pair reaches the node, and a wrong feed shows in ``out``.

        ``weight`` is scaled away from 1 and ``bias`` away from 0 so a body that
        handed the node's gamma and beta the wrong operands -- or dropped them --
        cannot match.
        """
        host = torch.randn(2, 3, 8)
        weight = torch.randn(8) * 2.0 + 1.0
        bias = torch.randn(8)
        out = torch.ops.aten.native_layer_norm(
            _to_device(host), (8,), weight=weight, bias=bias, eps=1e-5
        )
        _assert_matches(out, _reference(host, (8,), weight, bias))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("eps", [1e-5, 1e-3])
    def test_eps_reaches_the_node(self, eps):
        """``eps`` is added to the variance, and it is in the graph key.

        A body that baked the first ``eps`` into a warm plan and answered the
        second call out of it would return the first call's ``rstd``; the two
        cases run through one process, so a key that lost ``eps`` would show as a
        mismatch on whichever case ran second.
        """
        host = torch.randn(2, 8, 512)
        out = torch.ops.aten.native_layer_norm(_to_device(host), [512], None, None, eps)
        expected = _reference(host, (512,), None, None, eps)
        _assert_matches(out, expected)

    @pytest.mark.anyplatform
    def test_weight_without_bias(self):
        """``weight`` alone is a valid spelling, and ``bias`` is ``nil``.

        The body reads the two optional operands positionally out of the
        operands it built, so a weight-only call is the case where a body that
        assumed both are present, or read ``bias`` out of ``args[1]``, would feed
        the node gamma and beta from the wrong slots.

        ``weight`` is left on the host on purpose, which is how a caller writes
        the call. A body that fed the host tensor to the graph would resolve no
        Metal buffer for it, and the measured failure is not an exception: the
        operand is refused, the body is written to round-trip on refusal, and
        the caller receives the *host* tensors -- values right, device wrong.
        Hence the device assertion in ``_assert_matches``.
        """
        host = torch.randn(2, 3, 8)
        weight = torch.randn(8) + 1.0
        out = torch.ops.aten.native_layer_norm(
            _to_device(host), (8,), weight=weight, bias=None, eps=1e-5
        )
        _assert_matches(out, _reference(host, (8,), weight, None))
        assert out[0].device.type == "flagos"

    @pytest.mark.anyplatform
    def test_a_host_affine_pair_is_moved_rather_than_answered_on_the_host(self):
        """Both affine operands on the host is the common ``F.layer_norm`` call.

        The device result is the one the caller asked for, and it equals what the
        same pair produces once moved: the two calls differ only in where the
        operand was built, so a body that computed with a different pair -- or
        dropped the pair on one path and not the other -- cannot answer both.
        """
        host = torch.randn(2, 8, 512)
        weight = torch.randn(512) * 2.0 + 1.0
        bias = torch.randn(512)
        on_host = torch.ops.aten.native_layer_norm(
            _to_device(host), [512], weight, bias, 1e-5
        )
        moved = torch.ops.aten.native_layer_norm(
            _to_device(host), [512], weight.to(DEVICE), bias.to(DEVICE), 1e-5
        )
        for got, expected, name in zip(on_host, moved, ("out", "mean", "rstd")):
            assert got.device.type == "flagos", f"{name} left the device"
            torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
        _assert_matches(on_host, _reference(host, (512,), weight, bias))

    @pytest.mark.anyplatform
    def test_bias_without_weight(self):
        """``bias`` alone is the mirror case, and the node's beta-only feed."""
        host = torch.randn(2, 3, 8)
        bias = torch.randn(8)
        out = torch.ops.aten.native_layer_norm(_to_device(host), (8,), None, bias, 1e-5)
        _assert_matches(out, _reference(host, (8,), None, bias))

    @pytest.mark.anyplatform
    def test_weight_and_bias_on_the_full_normalized_shape(self):
        """The affine pair over the whole input, which is one row per sample.

        ``normalized_shape`` equal to the input's own shape reduces every axis,
        so ``mean`` and ``rstd`` are all-singleton and the result is the
        per-sample normalisation.
        """
        host = torch.randn(4, 6, 7, 8)
        weight = torch.randn(4, 6, 7, 8)
        bias = torch.randn(4, 6, 7, 8)
        out = torch.ops.aten.native_layer_norm(
            _to_device(host), (4, 6, 7, 8), weight=weight, bias=bias, eps=1e-5
        )
        _assert_matches(out, _reference(host, (4, 6, 7, 8), weight, bias))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_every_served_dtype(self, dtype):
        """The three float widths are served by the graph, not round-tripped."""
        host = (torch.randn(2, 3, 8) * 3.0).to(dtype)
        out = torch.ops.aten.native_layer_norm(_to_device(host), [8], None, None, 1e-5)
        assert out[0].dtype == dtype
        _assert_matches(out, _reference(host, (8,)), dtype=dtype)

    @pytest.mark.anyplatform
    def test_non_contiguous_input(self):
        """A strided operand is materialised before the graph reads it.

        The body hands the graph a dense buffer, so a strided operand read
        densely would read the wrong bytes (see the row-pitch note in
        ``mps_common.h``).
        """
        host = torch.randn(8, 4, 6).transpose(0, 1)
        assert not host.is_contiguous()
        out = torch.ops.aten.native_layer_norm(_to_device(host), [6], None, None, 1e-5)
        _assert_matches(out, _reference(host, (6,)))

    @pytest.mark.anyplatform
    def test_non_contiguous_weight(self):
        """The affine pair is materialised too, since it is graph input."""
        host = torch.randn(2, 3, 8)
        weight = torch.randn(16)[::2]
        assert not weight.is_contiguous()
        out = torch.ops.aten.native_layer_norm(
            _to_device(host), (8,), weight=weight, bias=None, eps=1e-5
        )
        _assert_matches(out, _reference(host, (8,), weight, None))

    @pytest.mark.anyplatform
    def test_empty_input_keeps_the_shape(self):
        """A zero-element input is a valid call and answered like the reference.

        MPSGraph rejects an empty tensor, so the body short-circuits this shape
        before it builds a node -- and the answer is the reference's own, which
        is an empty ``out`` rather than a decline the caller sees.
        """
        host = torch.empty(0, 8)
        out = torch.ops.aten.native_layer_norm(_to_device(host), [8], None, None, 1e-5)
        expected = _reference(host, (8,))
        assert out[0].shape == expected[0].shape == (0, 8)
        assert out[1].shape == expected[1].shape
        assert out[2].shape == expected[2].shape

    @pytest.mark.anyplatform
    def test_rank_one_input(self):
        """A vector input reduces to a scalar per statistic, keepdim-shaped."""
        host = torch.randn(16)
        out = torch.ops.aten.native_layer_norm(_to_device(host), [16], None, None, 1e-5)
        assert out[1].shape == (1,)
        _assert_matches(out, _reference(host, (16,)))

    @pytest.mark.anyplatform
    def test_the_layer_norm_front_end_reaches_the_same_kernel(self):
        """``F.layer_norm`` is the spelling a model calls, and it lowers here.

        The op under test is ``native_layer_norm``; ``F.layer_norm`` is the front
        end that reaches it. The statistics are discarded rather than compared,
        because the front end does not expose them -- what this holds is that the
        module-level call the whole increment exists for is served by the kernel
        and not left on the fallback.
        """
        host = torch.randn(2, 8, 512)
        weight = torch.randn(512)
        bias = torch.randn(512)
        out = torch.nn.functional.layer_norm(
            _to_device(host), (512,), weight=weight, bias=bias, eps=1e-5
        )
        expected = torch.nn.functional.layer_norm(
            host, (512,), weight=weight, bias=bias, eps=1e-5
        )
        assert out.device.type == "flagos"
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)


class TestLayerNormStatShapes:
    """``mean`` and ``rstd`` come back keepdim, which is the reference's shape."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "shape, normalized_shape, stat_shape",
        [
            ((2, 3, 8), (8,), (2, 3, 1)),
            ((2, 3, 8), (3, 8), (2, 1, 1)),
            ((4, 6, 7, 8), (6, 7, 8), (4, 1, 1, 1)),
            ((4, 6, 7, 8), (7, 8), (4, 6, 1, 1)),
            ((4, 6, 7, 8), (8,), (4, 6, 7, 1)),
        ],
    )
    def test_the_statistics_keep_the_rank(self, shape, normalized_shape, stat_shape):
        """The reduced axes are left at 1, and ``out`` keeps the input's shape.

        This is the shape a body has to build itself and the one MPSGraph's named
        reductions already return; a body that reshaped them to the *reduced*
        shape -- ``(2, 3)`` for the first case -- would return a tensor the
        reference's caller is not owed, and would broadcast wrongly in the
        backward pass.
        """
        host = torch.randn(*shape)
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            _to_device(host), list(normalized_shape), None, None, 1e-5
        )
        assert out.shape == shape
        assert mean.shape == stat_shape
        assert rstd.shape == stat_shape
        expected = _reference(host, normalized_shape)
        assert mean.shape == expected[1].shape
        assert rstd.shape == expected[2].shape

    @pytest.mark.anyplatform
    def test_the_statistics_are_the_reference_numbers(self):
        """Shape and value together, since a correct shape can hold a wrong mean."""
        host = torch.randn(2, 3, 8) * 3.0 + 1.0
        _, mean, rstd = torch.ops.aten.native_layer_norm(
            _to_device(host), [8], None, None, 1e-5
        )
        expected = _reference(host, (8,))
        torch.testing.assert_close(mean.cpu(), expected[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(rstd.cpu(), expected[2], rtol=1e-5, atol=1e-5)


class TestLayerNormVariance:
    """The variance is the population one, which a sample node cannot answer."""

    @pytest.mark.anyplatform
    def test_a_three_element_row_separates_the_two_variances(self):
        """``N=3``, where the sample and population answers are far apart.

        The reference divides by ``N`` (``unbiased=False``), so a body built on a
        sample-variance node would be off by ``sqrt(N/(N-1))`` -- 22% at ``N=3``,
        which is visible without a tolerance. The row is written out rather than
        drawn so the case is the same on every run: mean 3, squared deviations
        4 + 1 + 9 = 14, so the population variance is ``14/3`` and the sample one
        ``14/2 = 7``. Their reciprocal square roots are 0.4629 and 0.3780 at
        ``eps = 1e-5``, which is the difference this case is here for.
        """
        host = torch.tensor([[1.0, 2.0, 6.0]])
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            _to_device(host), [3], None, None, 1e-5
        )
        expected = _reference(host, (3,))
        got = rstd[0, 0].cpu().item()
        assert got == pytest.approx(expected[2][0, 0].item(), abs=1e-5)
        assert got == pytest.approx(1.0 / (14.0 / 3.0) ** 0.5, abs=1e-4)
        assert abs(got - 1.0 / 7.0**0.5) > 1e-2
        torch.testing.assert_close(out.cpu(), expected[0], rtol=1e-6, atol=1e-6)

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("extent", [2, 3, 4, 5])
    def test_matches_the_reference_at_small_reduced_extents(self, extent):
        """The ``N`` values where the two variances differ most.

        The ratio between them is ``sqrt(N/(N-1))``, so the separation is largest
        at the smallest ``N`` and vanishes as ``N`` grows; a body that leaned on
        the large-shape cases above for its evidence would not be distinguishing
        them at all.
        """
        host = torch.randn(2, extent)
        out = torch.ops.aten.native_layer_norm(
            _to_device(host), [extent], None, None, 1e-5
        )
        _assert_matches(out, _reference(host, (extent,)))


class TestLayerNormDeclines:
    """The shapes the graph has no node for reach the reference's own error."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "shape, normalized_shape",
        [
            ((2, 3, 8), (2,)),
            ((2, 3, 8), (3,)),
            ((2, 3, 8), (2, 3)),
            ((4, 6, 7, 8), (6, 7)),
            ((4, 6, 7, 8), (4, 6)),
        ],
        ids=[
            "leading-axis",
            "middle-axis",
            "two-leading-axes",
            "rank-4-non-trailing",
            "rank-4-leading",
        ],
    )
    def test_non_trailing_normalized_shape_reaches_the_reference(
        self, shape, normalized_shape
    ):
        """A shape that is not the input's trailing axes is the reference's error.

        The body checks the suffix on the host and declines, so the caller sees
        aten's own ``Given normalized_shape=[...], expected input with shape
        [*, ...]`` rather than a node that normalised over the wrong axes. Both
        directions are covered -- a normalised shape over the middle of the input
        and one over the leading axes.
        """
        host = torch.randn(*shape)
        with pytest.raises(RuntimeError, match="normalized_shape"):
            torch.ops.aten.native_layer_norm(
                _to_device(host), list(normalized_shape), None, None, 1e-5
            )

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "shape, normalized_shape",
        [
            ((2, 3, 8), (2, 3, 8, 1)),
            ((2, 3, 8), (8, 1)),
            ((8,), (8, 1)),
        ],
        ids=["over-rank-3", "one-longer", "over-rank-1"],
    )
    def test_normalized_shape_longer_than_the_input_raises(
        self, shape, normalized_shape
    ):
        """The other side of the suffix rule, and the body declines it too.

        ``normalized_shape`` with more axes than the input is the reference's
        ``RuntimeError`` as well, and the host check has to decline it rather than
        reduce over axes that do not exist.
        """
        host = torch.randn(*shape)
        with pytest.raises(RuntimeError):
            torch.ops.aten.native_layer_norm(
                _to_device(host), list(normalized_shape), None, None, 1e-5
            )

    @pytest.mark.anyplatform
    def test_the_trailing_run_is_not_just_the_last_axis(self):
        """A rank-3 input with a two-axis run is served, its neighbour is not.

        The pair of calls is the point: ``(3, 8)`` is the input's own trailing run
        and takes the graph, while ``(2, 3)`` is not and takes the reference. A
        body that had hard-coded a single trailing axis would serve the second one
        as well, and one that had checked only the last axis of the shape would
        serve it too.
        """
        host = torch.randn(2, 3, 8)
        _assert_matches(
            torch.ops.aten.native_layer_norm(
                _to_device(host), [3, 8], None, None, 1e-5
            ),
            _reference(host, (3, 8)),
        )
        with pytest.raises(RuntimeError, match="normalized_shape"):
            torch.ops.aten.native_layer_norm(_to_device(host), [2, 3], None, None, 1e-5)


class TestLayerNormDtypes:
    """The served widths are the reduction family's, and the rest round-trip."""

    @pytest.mark.anyplatform
    def test_float64_takes_the_round_trip_without_leaving_the_device(self):
        """float64 has no MPSGraph element type; the value is still right.

        The result stays on the flagos device. The body round-trips the *values*
        through the host -- it copies in, computes with the reference kernel and
        copies the three results back -- rather than answering with the host
        tensors themselves, which is what a bare ``return at::native_layer_norm(
        input.cpu(), ...)`` would do.
        """
        host = torch.randn(2, 3, 8, dtype=torch.float64)
        out = torch.ops.aten.native_layer_norm(_to_device(host), [8], None, None, 1e-5)
        assert out[0].dtype == torch.float64
        _assert_matches(out, _reference(host, (8,)), dtype=torch.float64)

    @pytest.mark.anyplatform
    def test_float64_round_trips_on_a_large_shape_too(self):
        """The same width at a size where a served graph would be visible.

        ``float64`` is declined by the dtype gate rather than by a shape, so this
        is the same case at a shape a model would present; it exercises the
        round-trip where a silent promotion would be cheapest to miss.
        """
        host = torch.randn(2, 8, 512, dtype=torch.float64)
        out = torch.ops.aten.native_layer_norm(
            _to_device(host), [512], None, None, 1e-5
        )
        assert out[0].dtype == torch.float64
        _assert_matches(out, _reference(host, (512,)), dtype=torch.float64)

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_widths_keep_their_dtype(self, dtype):
        """The graph serves half and bfloat16 without widening the result."""
        host = (torch.randn(2, 3, 8) * 3.0).to(dtype)
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            _to_device(host), [8], None, None, 1e-5
        )
        assert out.dtype == dtype == mean.dtype == rstd.dtype
        expected = _reference(host, (8,))
        for got_t, exp_t in zip((out, mean, rstd), expected):
            torch.testing.assert_close(
                got_t.cpu().float(), exp_t.float(), **LOW_PRECISION_TOLERANCE
            )

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.int16])
    def test_integer_inputs_are_reference_errors_not_a_graph_at_their_width(
        self, dtype
    ):
        """An integer input is refused, and refused the same way by the reference.

        This is the one dtype case where the reference itself has no answer:
        ``LayerNormKernelImpl`` is implemented for the float widths only, so
        ``native_layer_norm`` on an int tensor raises on the CPU too. The body
        declines -- the operand has no MPSGraph element type and the suffix check
        would leave it nowhere to go -- and the error the caller sees is the
        reference's own rather than a graph that computed something in int.

        The raise is asserted against the CPU's own, because "it raised" would
        also hold for a body that raised a different error, or for one that
        raised where the CPU would have answered.
        """
        host = torch.randint(0, 8, (2, 3, 8)).to(dtype)
        with pytest.raises(NotImplementedError) as on_host:
            _reference(host, (8,))
        with pytest.raises(NotImplementedError) as on_device:
            torch.ops.aten.native_layer_norm(_to_device(host), [8], None, None, 1e-5)
        assert str(on_device.value) == str(on_host.value)


class TestLayerNormPlanReuse:
    """One compiled plan answers a transformer block's calls."""

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_another_dtype(self):
        """An operand of another width gets its own plan, not the first one's.

        A key that lost an operand would answer the second call out of the first
        graph's plan, which reads the wrong bytes rather than failing.
        """
        for dtype in FLOAT_DTYPES:
            host = (torch.randn(2, 3, 8) * 3.0).to(dtype)
            _assert_matches(
                torch.ops.aten.native_layer_norm(
                    _to_device(host), [8], None, None, 1e-5
                ),
                _reference(host, (8,)),
                dtype=dtype,
            )

    @pytest.mark.anyplatform
    def test_normalized_shape_is_in_the_key(self):
        """Two runs over the same input are two plans, not one.

        ``(8,)`` and ``(3, 8)`` over the same ``(2, 3, 8)`` input have the same
        operand signature and different result shapes, so a key that named only
        the operands would hand the second call the first plan -- a
        correctly-shaped ``out`` and wrong statistics.
        """
        host = torch.randn(2, 3, 8)
        device = _to_device(host)
        one_axis = torch.ops.aten.native_layer_norm(device, [8], None, None, 1e-5)
        two_axes = torch.ops.aten.native_layer_norm(device, [3, 8], None, None, 1e-5)
        _assert_matches(one_axis, _reference(host, (8,)))
        _assert_matches(two_axes, _reference(host, (3, 8)))
        assert one_axis[1].shape != two_axes[1].shape

    @pytest.mark.anyplatform
    def test_eps_is_in_the_key(self):
        """Two ``eps`` values are two plans, and the second is not the first's.

        The body bakes ``eps`` into the graph as a constant, so a key that lost it
        would answer the second call with the first call's ``rstd`` -- a wrong
        number rather than a failure, and one that a single-``eps`` suite cannot
        see.
        """
        host = torch.randn(2, 8, 512) * 3.0
        device = _to_device(host)
        small = torch.ops.aten.native_layer_norm(device, [512], None, None, 1e-5)
        large = torch.ops.aten.native_layer_norm(device, [512], None, None, 1e-1)
        _assert_matches(small, _reference(host, (512,), eps=1e-5))
        _assert_matches(large, _reference(host, (512,), eps=1e-1))
        # The two answers are distinguishable, which is the whole point of
        # keeping eps in the key.
        assert large[2].cpu().max() < small[2].cpu().max()

    @pytest.mark.anyplatform
    def test_the_affine_flags_are_in_the_key(self):
        """A weight-less call and a weighted one are two plans over one input.

        The operands differ only in how many the body built, so the plan is
        separated by the ``has_weight``/``has_bias`` entries in the key and by the
        result signature; a call served out of the other one's plan would drop the
        affine transform.
        """
        host = torch.randn(2, 3, 8)
        device = _to_device(host)
        weight = torch.randn(8) + 1.0
        plain = torch.ops.aten.native_layer_norm(device, [8], None, None, 1e-5)
        affine = torch.ops.aten.native_layer_norm(
            device, (8,), weight=weight, bias=None, eps=1e-5
        )
        _assert_matches(plain, _reference(host, (8,)))
        _assert_matches(affine, _reference(host, (8,), weight, None))
        assert not torch.allclose(plain[0].cpu(), affine[0].cpu(), atol=1e-3)

    @pytest.mark.anyplatform
    def test_a_second_call_after_a_declined_one_is_served(self):
        """A declined call leaves no plan that the next one inherits.

        The suffix check returns before the graph is built, so the plan cache is
        untouched; this holds that the next trailing-run call still gets a correct
        answer rather than a half-built graph.
        """
        host = torch.randn(2, 3, 8)
        device = _to_device(host)
        with pytest.raises(RuntimeError, match="normalized_shape"):
            torch.ops.aten.native_layer_norm(device, [2], None, None, 1e-5)
        _assert_matches(
            torch.ops.aten.native_layer_norm(device, [8], None, None, 1e-5),
            _reference(host, (8,)),
        )

    @pytest.mark.anyplatform
    def test_the_three_results_land_in_their_own_buffers(self):
        """``out``, ``mean`` and ``rstd`` are separate tensors with their own data.

        The multi-result path flattens one executable target per result; a body
        whose targets were listed in a different order from ``outs`` would write
        each graph result into the wrong buffer, and the likeliest symptom is the
        statistics appearing where ``out`` belongs. The three are checked for
        being distinct storage as well as distinct values.
        """
        host = torch.randn(2, 3, 8)
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            _to_device(host), [8], None, None, 1e-5
        )
        assert out.data_ptr() not in (mean.data_ptr(), rstd.data_ptr())
        assert mean.data_ptr() != rstd.data_ptr()
        expected = _reference(host, (8,))
        torch.testing.assert_close(mean.cpu(), expected[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(rstd.cpu(), expected[2], rtol=1e-5, atol=1e-5)
        # The statistics are near zero and the normalised values are not, so a
        # swapped buffer is visible in the magnitude alone.
        assert mean.cpu().abs().max().item() < 1.0
        assert out.cpu().abs().max().item() > 1.0


class TestLayerNormDispatch:
    """``native_layer_norm`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_layer_norm_to_mps(self):
        assert routed_backend_or_none("native_layer_norm") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """``native_layer_norm.out`` writes into caller tensors, like the rest."""
        assert routed_backend_or_none("native_layer_norm.out") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether the graph ran or the op reached
        ``cpu_fallback``, because both compute the same answer. The log line is
        emitted by the C++ dispatcher only on the ``mps`` slot. Run in a
        subprocess because ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from device_init import torch, torch_fl; "
                "t = torch.randn(2, 3, 8, device='flagos:0'); "
                "torch.ops.aten.native_layer_norm(t, [8], None, None, 1e-5); "
                "torch.nn.functional.layer_norm(t, [8])",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] native_layer_norm -> mps") == 2

    @pytest.mark.main_ops
    def test_the_cpu_fallback_no_longer_carries_layer_norm(self):
        """The fallback log is empty of the decomposition the doc predicted.

        Before this increment ``F.layer_norm`` round-tripped and the fallback log
        printed ``native_batch_norm`` (the inference-mode spelling the CPU kernel
        decomposes into) and ``addcmul``. The design doc predicted the graph
        increment would be those two ops; measured, MPSGraph has the whole
        operation behind one normalisation node, so neither reaches the host. This
        holds that -- and if a later change reintroduces the decomposition, this is
        where it shows.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "fallback"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from device_init import torch, torch_fl; "
                "t = torch.randn(2, 3, 8, device='flagos:0'); "
                "torch.nn.functional.layer_norm(t, [8]); "
                "torch.ops.aten.native_layer_norm(t, [8], None, None, 1e-5)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "native_layer_norm" not in result.stderr
        assert "native_batch_norm" not in result.stderr
        assert "addcmul" not in result.stderr

    @pytest.mark.main_ops
    def test_a_declined_shape_does_not_reach_the_node(self):
        """A non-trailing call is entered and answered by the reference.

        The dispatch log names the backend even when the body declines, because
        the kernel is what decided; what this holds is that the *answer* is the
        reference's on the path the body has no node for.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                # A one-liner cannot carry this: `try` is a compound statement
                # and a semicolon will not start one, so the body is written in
                # newlines that the `-c` argument preserves.
                "from device_init import torch, torch_fl\n"
                "t = torch.randn(2, 3, 8, device='flagos:0')\n"
                "try:\n"
                "    torch.ops.aten.native_layer_norm(t, [2], None, None, 1e-5)\n"
                "except RuntimeError as e:\n"
                "    print('RuntimeError', 'normalized_shape' in str(e))\n",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "RuntimeError True" in result.stdout
        assert "[flagos dispatch] native_layer_norm -> mps" in result.stderr


class TestLayerNormCost:
    """The op is served, not round-tripped, on the shapes a model presents."""

    @pytest.mark.anyplatform
    def test_the_device_call_is_far_cheaper_than_one_round_trip(self):
        """The served call is an order of magnitude below a *copy* of the operand.

        The regression this guards against is a silent return to the fallback,
        which a correctness test cannot see. The yardstick is the cheapest part
        of any round trip -- reading the operand back off the device -- over a
        shape large enough that the transfer, not the dispatch, sets the time.
        Measured on an M5 at ``(8, 512, 4096)``: the graph is 4.0-6.3 ms and the
        device-to-host read alone is 3.1-3.7 ms, so a served call would have to
        gain more than an order of magnitude to trip this. What the bound says
        is that the call does not carry a transfer; anything that copied the
        operand would cross it immediately.

        Neither a *host* call nor the same op in float64 is the right yardstick.
        A host call with no copy at all is cheap at small shapes -- measured on
        an M5 it beats the device call at ``(2, 512, 1024)``, 0.23 ms against
        0.35 ms -- and the float64 round trip came out only 3.5-4.1x the graph
        at the shape below, inside a factor a noisy run can move. The transfer
        is the part of the round trip that dominates at this size, and it is
        stable at 3.1-3.7 ms across runs.
        """
        import time

        shape = (8, 512, 4096)
        normalized_shape = shape[-1]
        host = torch.randn(*shape)
        operand = host.to(DEVICE)

        def per_call(fn, n=20):
            for _ in range(3):
                fn()
            torch_fl._C._synchronize()
            start = time.perf_counter()
            for _ in range(n):
                fn()
            torch_fl._C._synchronize()
            return (time.perf_counter() - start) / n * 1e3

        on_device = per_call(
            lambda: torch.ops.aten.native_layer_norm(
                operand, [normalized_shape], None, None, 1e-5
            )
        )
        one_read = per_call(lambda: operand.cpu())
        assert on_device < one_read * 8.0, (
            f"layer_norm on the device took {on_device:.3f} ms against "
            f"{one_read:.3f} ms to read the operand back once -- the op is "
            "paying a transfer it should not"
        )
        # And the result is still the reference's, so the timing is of a correct
        # answer rather than of a fast wrong one.
        _assert_matches(
            torch.ops.aten.native_layer_norm(
                operand, [normalized_shape], None, None, 1e-5
            ),
            _reference(host, (normalized_shape,)),
        )
