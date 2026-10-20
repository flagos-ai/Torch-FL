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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``native_batch_norm``.

``native_batch_norm`` is the normalisation a convolutional block calls, and the
second body here that answers with more than one tensor --
``native_batch_norm(input, weight?, bias?, running_mean?, running_var?,
training, momentum, eps)`` returns ``(out, save_mean, save_invstd)``.

**It is two different computations behind one schema, and the body branches on
``training`` rather than on a tensor.** The graph covers the *inference* path
only, where the statistics are the running buffers the caller passes and the
whole operation is one ``normalizationWithTensor:meanTensor:varianceTensor:
gammaTensor:betaTensor:epsilon:name:`` node. Training is declined and takes the
CPU round-trip, for a reason a forward-output comparison cannot see: it updates
``running_mean``/``running_var`` **in place** with an exponential moving average
of the *unbiased* batch variance, and no graph node writes a caller's buffer. A
body that served training while dropping the update would leave a model's running
statistics frozen and still return a matching forward output.

Three contract details the inference path has to get right, each measured against
the reference rather than read off the header:

* **``save_mean`` and ``save_invstd`` in inference are *empty*** -- shape ``(0,)``
  on the input's device and dtype -- and not the running statistics. Measured:
  ``native_batch_norm(x, w, b, rm, rv, training=False, ...)`` on the reference
  answers ``(out, tensor([]), tensor([]))``, while the same call with
  ``training=True`` answers two ``[C]`` tensors. The two results are the *batch*
  statistics the backward pass consumes, and an inference call has none. This is
  the case ``TestBatchNormStatShapes`` holds outright, because the intuitive
  reading -- the running statistics, which are right there -- produces a
  correctly-valued ``out`` and wrong-shaped statistics.
* **The channel axis is 1** for every rank the reference accepts in inference
  (NCHW and NCDHW alike), so ``weight``/``bias`` are ``[C]`` and the node
  broadcasts them along the rest. The mean and variance operands are reshaped to
  the ``[1, C, 1, ...]`` channel view inside the graph, because the node's type
  inference requires its statistics to be broadcast *compatible* with the input
  and a trailing ``[C]`` operand is not -- the node rejects it with a process
  abort, not a declined build.
* **``momentum`` is not an operand in inference.** It only weights the running
  update, so it is read on neither path: the graph is a function of ``eps`` alone
  among the host scalars, and ``momentum`` is what the round-trip carries.

An integral input is declined like every other reduction here, and in this case
the reference itself has no answer: ``batch_norm`` is not implemented for
``Long`` on the CPU either, so the round-trip is what raises the reference's own
message.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

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

# ``input shape`` for the inference cases. The channel axis is 1 in every one of
# them, which is the axis the body normalises over; the ranks span a rank-2
# tensor (no spatial axes), NCHW and NCDHW.
SHAPE_CASES = [
    (4, 7),
    (2, 3, 8),
    (2, 3, 4, 5),
    (1, 5, 6, 6),
    (2, 3, 4, 5, 6),
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another case
    # left behind.
    return t.clone().to(DEVICE)


def _reference(
    input: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    running_mean: torch.Tensor | None = None,
    running_var: torch.Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    """``at::native_batch_norm`` on the host, returning the reference triple.

    The reference the kernel is held to, and deliberately the same op the body
    round-trips to rather than a hand-built ``(x - mean) / sqrt(var + eps) * w +
    b``: the empty inference statistics, the training-mode running update and the
    inference-mode decline are the reference's contract, so a reimplementation
    would encode the test author's reading of it and could agree with a body that
    is wrong in the same way.
    """
    return torch.ops.aten.native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )


def _running_pair(channels: int, dtype: torch.dtype = torch.float32):
    """A running mean and a strictly positive running variance.

    The variance is shifted off zero so ``1 / sqrt(var + eps)`` is finite and a
    body that fed the node the *variance* where the *standard deviation* belongs
    -- or the reciprocal of one for the other -- cannot agree by accident.
    """
    return torch.randn(channels).to(dtype), (torch.rand(channels) + 0.5).to(dtype)


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

    The inference statistics are empty, so the value half of this is a no-op for
    them -- ``assert_close`` on two empty tensors passes -- and the *shape* half
    is what holds that contract; a body that returned the running statistics
    would fail on the shape assertion rather than on a value comparison.
    """
    assert len(got) == len(expected) == 3
    for index, name in enumerate(("out", "save_mean", "save_invstd")):
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


class TestBatchNormCorrectness:
    """``native_batch_norm`` in inference equals the CPU reference on device."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("shape", SHAPE_CASES)
    def test_matches_cpu_for_inference_shapes(self, shape):
        """Every rank the reference accepts normalises over axis 1.

        The channel axis is 1 in each case, which is the axis the body reads and
        the axis the node's ``[C]`` gamma and beta broadcast along.
        """
        channels = shape[1]
        host = torch.randn(*shape)
        weight = torch.randn(channels) * 2.0 + 1.0
        bias = torch.randn(channels)
        running_mean, running_var = _running_pair(channels)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        _assert_matches(
            out, _reference(host, weight, bias, running_mean, running_var, False)
        )

    @pytest.mark.anyplatform
    def test_weight_and_bias_are_the_gamma_and_beta(self):
        """The affine pair reaches the node, and a wrong feed shows in ``out``.

        ``weight`` is scaled away from 1 and ``bias`` away from 0 so a body that
        handed the node's gamma and beta the wrong operands -- or dropped them --
        cannot match.
        """
        host = torch.randn(2, 3, 4, 5)
        weight = torch.randn(3) * 2.0 + 1.0
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        _assert_matches(
            out, _reference(host, weight, bias, running_mean, running_var, False)
        )

    @pytest.mark.anyplatform
    def test_weight_without_bias(self):
        """``weight`` alone is a valid spelling and the node gets a zero beta.

        The body fills the missing affine operand with an all-zeros ``[C]`` so the
        operand *list* is fixed, and a body that instead dropped the feed -- or
        read the wrong slot -- would apply the caller's ``weight`` as a bias.
        """
        host = torch.randn(2, 3, 4, 5)
        weight = torch.randn(3) + 1.0
        running_mean, running_var = _running_pair(3)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            None,
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        _assert_matches(
            out, _reference(host, weight, None, running_mean, running_var, False)
        )

    @pytest.mark.anyplatform
    def test_bias_without_weight(self):
        """``bias`` alone is the mirror case, with an all-ones gamma."""
        host = torch.randn(2, 3, 4, 5)
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            None,
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        _assert_matches(
            out, _reference(host, None, bias, running_mean, running_var, False)
        )

    @pytest.mark.anyplatform
    def test_neither_affine_operand_is_a_valid_spelling(self):
        """Both absent is the plain normalisation, and both substitutes are used."""
        host = torch.randn(2, 5, 4, 4)
        running_mean, running_var = _running_pair(5)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            None,
            None,
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        _assert_matches(
            out, _reference(host, None, None, running_mean, running_var, False)
        )

    @pytest.mark.anyplatform
    def test_a_host_affine_pair_is_moved_rather_than_answered_on_the_host(self):
        """Both affine operands on the host is a common ``F.batch_norm`` call.

        The device result is the one the caller asked for, and it equals what the
        same pair produces once moved: the two calls differ only in where the
        operand was built, so a body that computed with a different pair -- or
        dropped the pair on one path and not the other -- cannot answer both.
        """
        host = torch.randn(2, 3, 4, 5)
        weight = torch.randn(3) * 2.0 + 1.0
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        on_host = torch.ops.aten.native_batch_norm(
            _to_device(host),
            weight,
            bias,
            running_mean,
            running_var,
            False,
            0.1,
            1e-5,
        )
        moved = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        for got, expected, name in zip(
            on_host, moved, ("out", "save_mean", "save_invstd")
        ):
            assert got.device.type == "flagos", f"{name} left the device"
            torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
        _assert_matches(
            on_host, _reference(host, weight, bias, running_mean, running_var, False)
        )

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_every_served_dtype(self, dtype):
        """The three float widths are served by the graph, not round-tripped."""
        host = (torch.randn(2, 3, 4, 5) * 3.0).to(dtype)
        weight = (torch.randn(3) + 1.0).to(dtype)
        bias = torch.randn(3).to(dtype)
        running_mean, running_var = _running_pair(3, dtype)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        assert out[0].dtype == dtype
        _assert_matches(
            out,
            _reference(host, weight, bias, running_mean, running_var, False),
            dtype=dtype,
        )

    @pytest.mark.anyplatform
    def test_non_contiguous_input(self):
        """A strided operand is materialised before the graph reads it.

        The body hands the graph a dense buffer, so a strided operand read
        densely would read the wrong bytes (see the row-pitch note in
        ``mps_common.h``).
        """
        host = torch.randn(2, 6, 4, 5)[:, ::2]
        assert not host.is_contiguous()
        weight = torch.randn(3) + 1.0
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        _assert_matches(
            out, _reference(host, weight, bias, running_mean, running_var, False)
        )

    @pytest.mark.anyplatform
    def test_empty_input_keeps_the_shape(self):
        """A zero-element input is a valid call and answered like the reference.

        MPSGraph rejects an empty tensor, so the body short-circuits this shape
        before it builds a node -- and the answer is the reference's own, which
        is an empty ``out`` rather than a decline the caller sees.
        """
        host = torch.empty(2, 3, 0, 5)
        weight = torch.randn(3)
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        out = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        expected = _reference(host, weight, bias, running_mean, running_var, False)
        assert out[0].shape == expected[0].shape == (2, 3, 0, 5)
        assert out[0].device.type == "flagos"
        assert out[1].shape == expected[1].shape
        assert out[2].shape == expected[2].shape

    @pytest.mark.anyplatform
    def test_the_batch_norm_front_end_reaches_the_same_kernel(self):
        """``F.batch_norm`` is the spelling a model calls, and it lowers here.

        The op under test is ``native_batch_norm``; ``F.batch_norm`` is the front
        end that reaches it in inference. The statistics are discarded rather than
        compared, because the front end does not expose them -- what this holds is
        that the module-level call the increment exists for is served by the
        kernel and not left on the fallback.
        """
        host = torch.randn(2, 3, 4, 5)
        weight = torch.randn(3)
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        out = torch.nn.functional.batch_norm(
            _to_device(host),
            _to_device(running_mean),
            _to_device(running_var),
            _to_device(weight),
            _to_device(bias),
            training=False,
            momentum=0.1,
            eps=1e-5,
        )
        expected = torch.nn.functional.batch_norm(
            host, running_mean, running_var, weight, bias, False, 0.1, 1e-5
        )
        assert out.device.type == "flagos"
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)


class TestBatchNormStatShapes:
    """The inference statistics are empty, which is the reference's own shape."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("shape", [(2, 3, 4, 5), (2, 3, 4, 5, 6), (4, 7)])
    def test_inference_statistics_are_empty(self, shape):
        """``save_mean`` and ``save_invstd`` are ``(0,)``, not the running pair.

        This is the case the increment's body got wrong first. The two results
        are the *batch* statistics the backward pass consumes; an inference call
        has none, so the reference returns empty tensors. The intuitive reading --
        the running statistics, which are right there in the arguments -- produces
        a correctly-valued ``out`` and a ``[C]`` tensor where the reference has a
        ``(0,)`` one, which a value comparison does not see.
        """
        channels = shape[1]
        host = torch.randn(*shape)
        weight = torch.randn(channels)
        bias = torch.randn(channels)
        running_mean, running_var = _running_pair(channels)
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        expected = _reference(host, weight, bias, running_mean, running_var, False)
        assert expected[1].shape == (0,)
        assert expected[2].shape == (0,)
        assert save_mean.shape == (0,)
        assert save_invstd.shape == (0,)
        assert save_mean.device.type == "flagos"
        assert save_invstd.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_training_statistics_are_the_channel_width(self):
        """The mirror case: training answers ``[C]`` batch statistics.

        Held next to the inference case so the two shapes are visibly different
        contracts rather than one shape the body could have hard-coded, and so the
        fact that training is declined -- not unserved -- is exercised too.
        """
        channels = 3
        host = torch.randn(2, channels, 4, 5)
        weight = torch.randn(channels)
        bias = torch.randn(channels)
        running_mean, running_var = _running_pair(channels)
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            True,
            0.1,
            1e-5,
        )
        torch_fl._C._synchronize()
        expected = _reference(
            host,
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
        )
        assert save_mean.shape == expected[1].shape == (channels,)
        assert save_invstd.shape == expected[2].shape == (channels,)
        assert save_mean.device.type == "flagos"
        torch.testing.assert_close(save_mean.cpu(), expected[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(save_invstd.cpu(), expected[2], rtol=1e-5, atol=1e-5)


class TestBatchNormTraining:
    """Training is declined and still updates the running buffers in place."""

    @pytest.mark.anyplatform
    def test_training_matches_the_reference_and_updates_the_running_buffers(self):
        """The declined training path is correct *including* the in-place update.

        This is the case the graph cannot answer and the reason the body declines
        rather than serving the inference node: training weights the running
        update by the *unbiased* batch variance and writes it into the caller's
        buffers, and no graph node writes a caller's buffer. A body that served
        the node would return a matching ``out`` and leave the running statistics
        frozen -- so this holds the buffers, on the *device* copies the caller
        passed, against what the reference computes.
        """
        channels = 3
        host = torch.randn(4, channels, 5, 6)
        weight = torch.randn(channels)
        bias = torch.randn(channels)
        running_mean, running_var = _running_pair(channels)
        # The device buffers are named so the in-place update is observable, and
        # the host pair is the reference's own starting point.
        device_mean = _to_device(running_mean)
        device_var = _to_device(running_var)
        host_mean = running_mean.clone()
        host_var = running_var.clone()
        out, _, _ = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            device_mean,
            device_var,
            True,
            0.1,
            1e-5,
        )
        torch_fl._C._synchronize()
        expected = _reference(host, weight, bias, host_mean, host_var, True)
        torch.testing.assert_close(out.cpu(), expected[0], rtol=1e-5, atol=1e-5)
        assert device_mean.device.type == "flagos"
        assert device_var.device.type == "flagos"
        # The update happened, and it is the reference's own numbers -- a body
        # that left the buffer alone would match on the first assertion and fail
        # here.
        assert not torch.allclose(device_mean.cpu(), running_mean)
        torch.testing.assert_close(device_mean.cpu(), host_mean, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(device_var.cpu(), host_var, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_the_running_update_is_the_unbiased_one(self):
        """The update weights the *unbiased* variance, which the node cannot do.

        ``momentum`` scales a correction of ``count / (count - 1)`` that a single
        normalisation node does not apply, so a body that reused the inference
        node for training would be off by that factor in the running variance.
        The case is small enough that the factor is visible, and the hold is
        against the variance the reference leaves in its own buffer rather than
        against a recomputation, because the update is an in-place write to the
        caller's tensor and the value is only observable there.
        """
        channels = 2
        host = torch.randn(2, channels, 2, 2)
        weight = torch.randn(channels)
        bias = torch.randn(channels)
        running_mean, running_var = _running_pair(channels)
        device_var = _to_device(running_var)
        host_var = running_var.clone()
        torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            device_var,
            True,
            0.1,
            1e-5,
        )
        torch_fl._C._synchronize()
        # The reference's answer is read off the buffers it wrote, which is the
        # only place the update exists.
        _reference(host, weight, bias, running_mean.clone(), host_var, True, 0.1, 1e-5)
        torch.testing.assert_close(device_var.cpu(), host_var, rtol=1e-5, atol=1e-5)
        # The update is a real move off the starting point, not the identity.
        assert not torch.allclose(device_var.cpu(), running_var, rtol=1e-3, atol=1e-3)


class TestBatchNormDeclines:
    """The shapes the graph has no node for reach the reference's own answer."""

    @pytest.mark.anyplatform
    def test_missing_running_statistics_decline_rather_than_build_a_node(self):
        """No running pair means no inference node, so the body declines.

        The node normalises by the statistics it is fed, and with no running
        buffers there is nothing to feed it -- the reference instead computes the
        batch statistics, which is a reduction the graph would have to build. The
        body declines on the missing pair.

        The *value* is deliberately not held here, because the reference has none
        to hold it to: measured, ``native_batch_norm`` with no running statistics
        **segfaults in plain PyTorch on CPU tensors** (rc=139, ``torch_fl`` not
        involved), which is the ``null stats`` abort recorded in the design doc.
        The same call on the device crashes in the round trip for the same reason,
        so the body's decline is what leads into the reference's abort rather than
        away from it -- and a test cannot assert on the value of a call the
        reference cannot survive.

        What is held is narrower and still load-bearing: this is the shape the
        *next* increment is for. MPSGraph has the whole thing -- measured,
        ``meanOfTensor:axes:`` and ``varianceOfTensor:meanTensor:axes:`` build the
        batch statistics and ``normalizationWithTensor:`` accepts them -- so the
        way to serve this case is a graph the body has not grown, and a body that
        has grown one is visible here as the crash *disappearing*. That test is
        therefore written to pass on the reference's abort and to fail loudly once
        the shape is served, which is the increment's signal rather than a
        permanently red case.
        """
        env = os.environ.copy()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "x = torch.randn(2, 3, 4, 5, device='flagos:0'); "
                "w = torch.randn(3, device='flagos:0'); "
                "torch.ops.aten.native_batch_norm("
                "x, w, None, None, None, False, 0.1, 1e-5)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        # Segfault is the signal the decline took the reference, and -11 is the
        # shell's report of it.
        assert result.returncode == -11, (
            f"the missing-statistics call left the crash path with rc="
            f"{result.returncode}; either the body grew a node for it -- in which "
            f"case this case belongs in the correctness suite -- or the round trip "
            f"changed: {result.stderr[-400:]}"
        )

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_integer_inputs_are_reference_errors_not_a_graph_at_their_width(
        self, dtype
    ):
        """An integer input is refused, and refused the same way by the reference.

        This is the one dtype case where the reference itself has no answer:
        ``batch_norm`` is implemented for the float widths only, so
        ``native_batch_norm`` on an int tensor raises on the CPU too. The body
        declines on the reduction dtype gate and the error the caller sees is the
        reference's own rather than a graph that computed something in int.
        """
        channels = 3
        host = torch.randint(0, 8, (2, channels, 4, 5)).to(dtype)
        weight = torch.randn(channels)
        bias = torch.randn(channels)
        running_mean, running_var = _running_pair(channels)
        with pytest.raises(NotImplementedError) as on_host:
            _reference(host, weight, bias, running_mean, running_var, False)
        with pytest.raises(NotImplementedError) as on_device:
            torch.ops.aten.native_batch_norm(
                _to_device(host),
                _to_device(weight),
                _to_device(bias),
                _to_device(running_mean),
                _to_device(running_var),
                False,
                0.1,
                1e-5,
            )
        assert str(on_device.value) == str(on_host.value)


class TestBatchNormPlanReuse:
    """One compiled plan answers a convolutional block's repeated calls."""

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_another_dtype(self):
        """An operand of another width gets its own plan, not the first one's.

        The key records the dtypes it was built for, so a second call at another
        width must compile rather than inherit the first graph -- a plan reused
        across widths reads the wrong bytes rather than failing.
        """
        for dtype in FLOAT_DTYPES:
            host = (torch.randn(2, 3, 4, 5) * 3.0).to(dtype)
            weight = (torch.randn(3) + 1.0).to(dtype)
            bias = torch.randn(3).to(dtype)
            running_mean, running_var = _running_pair(3, dtype)
            out = torch.ops.aten.native_batch_norm(
                _to_device(host),
                _to_device(weight),
                _to_device(bias),
                _to_device(running_mean),
                _to_device(running_var),
                False,
                0.1,
                1e-5,
            )
            _assert_matches(
                out,
                _reference(host, weight, bias, running_mean, running_var, False),
                dtype=dtype,
            )

    @pytest.mark.anyplatform
    def test_eps_is_in_the_key(self):
        """Two ``eps`` values are two plans, and the second is not the first's.

        ``eps`` is the one host scalar the body bakes into the graph, and the plan
        key carries nothing else from the arguments -- so a key that lost it would
        answer the second call with the first call's normalisation. The two
        answers are made distinguishable by taking ``eps`` far enough from the
        running variance's own scale that the outputs differ by more than the
        tolerance.
        """
        host = torch.randn(4, 3, 8, 8) * 0.01
        device = _to_device(host)
        weight = torch.randn(3) + 1.0
        bias = torch.randn(3)
        # A running variance near zero, so the two eps values move the output by
        # visibly different amounts rather than both sitting inside the noise.
        running_mean, running_var = _running_pair(3)
        running_var = torch.full((3,), 1e-6)
        small = torch.ops.aten.native_batch_norm(
            device,
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-6,
        )
        large = torch.ops.aten.native_batch_norm(
            device,
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1.0,
        )
        _assert_matches(
            small,
            _reference(host, weight, bias, running_mean, running_var, False, 0.1, 1e-6),
        )
        _assert_matches(
            large,
            _reference(host, weight, bias, running_mean, running_var, False, 0.1, 1.0),
        )
        assert not torch.allclose(small[0].cpu(), large[0].cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_a_new_channel_count_compiles_its_own_graph(self):
        """The channel count reaches the graph as a shape, and shapes fold in.

        The key does not name the channel count -- the operand and result shapes
        do -- so this holds that two calls over different widths are two correct
        answers rather than the first graph being reused with a reshaped
        operand list, which the channel-view reshape inside the graph would turn
        into a wrong normalisation or a build failure.
        """
        host_a = torch.randn(2, 3, 4, 5)
        host_b = torch.randn(2, 7, 4, 5)
        for host in (host_a, host_b):
            channels = host.shape[1]
            weight = torch.randn(channels) + 1.0
            bias = torch.randn(channels)
            running_mean, running_var = _running_pair(channels)
            out = torch.ops.aten.native_batch_norm(
                _to_device(host),
                _to_device(weight),
                _to_device(bias),
                _to_device(running_mean),
                _to_device(running_var),
                False,
                0.1,
                1e-5,
            )
            _assert_matches(
                out, _reference(host, weight, bias, running_mean, running_var, False)
            )

    @pytest.mark.anyplatform
    def test_the_affine_substitutes_share_a_plan_with_the_weighted_call(self):
        """A weight-less and a weighted call over one input are one plan.

        The body fills the missing operands with an all-ones gamma and an
        all-zeros beta rather than dropping them, so both calls present the same
        operand signature and the plan cache serves the second from the first. The
        two answers have to differ -- the weighted call applies ``weight`` -- which
        is what a shared plan would break if the substitutes carried a value into
        the key.
        """
        host = torch.randn(2, 3, 4, 5)
        device = _to_device(host)
        weight = torch.randn(3) * 2.0 + 1.0
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        plain = torch.ops.aten.native_batch_norm(
            device,
            None,
            None,
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        affine = torch.ops.aten.native_batch_norm(
            device,
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        _assert_matches(
            plain, _reference(host, None, None, running_mean, running_var, False)
        )
        _assert_matches(
            affine, _reference(host, weight, bias, running_mean, running_var, False)
        )
        assert not torch.allclose(plain[0].cpu(), affine[0].cpu(), atol=1e-3)

    @pytest.mark.anyplatform
    def test_a_second_call_after_a_declined_one_is_served(self):
        """A declined call leaves no plan that the next one inherits.

        The training and missing-statistics declines return before the graph is
        built, so the plan cache is untouched; this holds that the next inference
        call still compiles and answers rather than reusing a half-built graph.
        """
        host = torch.randn(2, 3, 4, 5)
        device = _to_device(host)
        weight = torch.randn(3)
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        torch.ops.aten.native_batch_norm(
            device,
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            True,
            0.1,
            1e-5,
        )
        torch_fl._C._synchronize()
        _assert_matches(
            torch.ops.aten.native_batch_norm(
                device,
                _to_device(weight),
                _to_device(bias),
                _to_device(running_mean),
                _to_device(running_var),
                False,
                0.1,
                1e-5,
            ),
            _reference(host, weight, bias, running_mean, running_var, False),
        )

    @pytest.mark.anyplatform
    def test_the_result_lands_in_its_own_buffer(self):
        """``out`` does not share storage with the two statistics.

        The multi-result path lists one executable target per result; a body whose
        targets were ordered differently from ``outs`` would write the graph's
        answer over a statistics buffer. The two statistics are empty here, so the
        hold is that the graph's result went to ``out`` -- a wrong target order
        would have written four million floats past the end of a zero-element
        buffer rather than silently swapping values.

        The two statistics are *not* compared against each other: both are empty,
        and an empty ``at::empty({0})`` on this backend reports a null
        ``data_ptr()``, so ``save_mean.data_ptr() != save_invstd.data_ptr()`` would
        fail for two tensors that are correctly distinct. Emptiness is the
        contract, and ``TestBatchNormStatShapes`` holds the shape.
        """
        host = torch.randn(2, 3, 4, 5)
        weight = torch.randn(3)
        bias = torch.randn(3)
        running_mean, running_var = _running_pair(3)
        out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
            _to_device(host),
            _to_device(weight),
            _to_device(bias),
            _to_device(running_mean),
            _to_device(running_var),
            False,
            0.1,
            1e-5,
        )
        assert out.numel() == host.numel()
        assert save_mean.numel() == 0 and save_invstd.numel() == 0
        assert out.data_ptr() not in (save_mean.data_ptr(), save_invstd.data_ptr())
        torch.testing.assert_close(
            out.cpu(),
            _reference(host, weight, bias, running_mean, running_var, False)[0],
            rtol=1e-5,
            atol=1e-5,
        )


class TestBatchNormDispatch:
    """``native_batch_norm`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_batch_norm_to_mps(self):
        assert routed_backend_or_none("native_batch_norm") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """``native_batch_norm.out`` writes into caller tensors, like the rest."""
        assert routed_backend_or_none("native_batch_norm.out") is None

    @pytest.mark.main_ops
    def test_conf_leaves_the_training_spellings_on_the_fallback(self):
        """The legit variants are the training spellings, and training is declined.

        ``_native_batch_norm_legit*`` are what ``F.batch_norm`` with
        ``training=True`` lowers to, and ``_batch_norm_with_update`` is the
        functional spelling; none of them has an inference graph to route, so the
        conf leaves them where the reference lives rather than claiming a slot
        whose body would decline every call.
        """
        for op in (
            "_native_batch_norm_legit",
            "_native_batch_norm_legit.no_stats",
            "_native_batch_norm_legit.out",
            "_native_batch_norm_legit_no_training",
            "_native_batch_norm_legit_functional",
            "_batch_norm_no_update",
            "_batch_norm_with_update",
        ):
            assert routed_backend_or_none(op) is None, op

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
                "import torch, torch_fl; "
                "t = torch.randn(2, 3, 4, 5, device='flagos:0'); "
                "w = torch.randn(3, device='flagos:0'); "
                "rm = torch.zeros(3, device='flagos:0'); "
                "rv = torch.ones(3, device='flagos:0'); "
                "torch.ops.aten.native_batch_norm(t, w, None, rm, rv, False, 0.1, 1e-5); "
                "torch.nn.functional.batch_norm(t, rm, rv, w, None, False, 0.1, 1e-5)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] native_batch_norm -> mps") == 2

    @pytest.mark.main_ops
    def test_the_inference_front_end_does_not_reach_the_host(self):
        """``F.batch_norm`` in inference is served, and the fallback log is quiet.

        Before this increment the front end round-tripped and the fallback log
        printed ``native_batch_norm``. The front end lowers to the op under test in
        inference, so neither the op nor its decomposition reaches the host -- and
        if a later change reintroduces the round trip, this is where it shows.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "fallback"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "t = torch.randn(2, 3, 4, 5, device='flagos:0'); "
                "w = torch.randn(3, device='flagos:0'); "
                "rm = torch.zeros(3, device='flagos:0'); "
                "rv = torch.ones(3, device='flagos:0'); "
                "torch.nn.functional.batch_norm(t, rm, rv, w, None, False, 0.1, 1e-5); "
                "torch.ops.aten.native_batch_norm(t, w, None, rm, rv, False, 0.1, 1e-5)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "native_batch_norm" not in result.stderr

    @pytest.mark.main_ops
    def test_a_training_call_is_entered_and_answered_by_the_reference(self):
        """A declined training call still logs the dispatch, because the body ran.

        The dispatch log names the backend even when the body declines -- the
        kernel is what decided. What this holds is that the declined path answers
        correctly and that the dispatch is still attributed to ``mps`` rather than
        silently reaching the fallback, which is what makes the decline auditable.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl\n"
                "t = torch.randn(2, 3, 4, 5, device='flagos:0')\n"
                "w = torch.randn(3, device='flagos:0')\n"
                "rm = torch.zeros(3, device='flagos:0')\n"
                "rv = torch.ones(3, device='flagos:0')\n"
                "out, mean, invstd = torch.ops.aten.native_batch_norm(\n"
                "    t, w, None, rm, rv, True, 0.1, 1e-5)\n"
                "print('out', tuple(out.shape), 'mean', tuple(mean.shape),\n"
                "      'invstd', tuple(invstd.shape), 'moved',\n"
                "      bool(rm.abs().sum() > 0))\n",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] native_batch_norm -> mps" in result.stderr
        assert "out (2, 3, 4, 5) mean (3,) invstd (3,) moved True" in result.stdout


class TestBatchNormCost:
    """The op is served, not round-tripped, on the shapes a model presents."""

    @pytest.mark.anyplatform
    def test_the_device_call_is_far_cheaper_than_one_round_trip(self):
        """The served call is well below a *copy* of the operand.

        The regression this guards against is a silent return to the fallback,
        which a correctness test cannot see. The yardstick is the cheapest part of
        any round trip -- reading the operand back off the device -- over a shape
        large enough that the transfer, not the dispatch, sets the time. What the
        bound says is that the call does not carry a transfer; anything that
        copied the operand would cross it. The fold factor is the layer_norm
        suite's, whose measurement this op shares: the graph is a few ms against a
        transfer that dominates at this size.
        """
        import time

        shape = (8, 512, 4096)
        channels = shape[1]
        host = torch.randn(*shape)
        operand = host.to(DEVICE)
        weight = torch.randn(channels, device=DEVICE)
        running_mean = torch.zeros(channels, device=DEVICE)
        running_var = torch.ones(channels, device=DEVICE)

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
            lambda: torch.ops.aten.native_batch_norm(
                operand, weight, None, running_mean, running_var, False, 0.1, 1e-5
            )
        )
        one_read = per_call(lambda: operand.cpu())
        assert on_device < one_read * 8.0, (
            f"batch_norm on the device took {on_device:.3f} ms against "
            f"{one_read:.3f} ms to read the operand back once -- the op is "
            "paying a transfer it should not"
        )
        _assert_matches(
            torch.ops.aten.native_batch_norm(
                operand, weight, None, running_mean, running_var, False, 0.1, 1e-5
            ),
            _reference(
                host,
                weight.cpu(),
                None,
                running_mean.cpu(),
                running_var.cpu(),
                False,
            ),
        )
