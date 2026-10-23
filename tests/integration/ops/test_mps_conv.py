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

"""Stage-3 Apple Silicon (Metal / MPSGraph) convolution.

Convolution is the last lane of stage 3: the ``gemm`` and ``softmax`` categories
cover what a transformer calls, and a vision model -- or a mixer's depthwise
causal ``conv1d`` -- is what needs this one. Five properties are pinned here.

* **Which spelling is registered.** On this PyTorch build
  ``aten.convolution`` is ``CompositeImplicitAutograd``: it validates its
  arguments and decomposes above the backend key, so what reaches the
  ``PrivateUse1`` slot -- what ``F.conv2d``, ``F.conv3d``, ``nn.Conv2d`` and
  ``nn.ConvTranspose2d`` all lower to -- is
  ``aten.convolution_overrideable``. The dispatch-log cases below are what hold
  that, because a numeric-only test passes on the fallback too.
* **The geometry is host-side.** MPSGraph's convolution descriptor carries the
  stride, dilation, groups and an explicit front/back padding pair per axis,
  none of which the operand shapes can say, so every one of them is part of the
  plan key. The interleaved cases below rotate through the signatures on a warm
  cache, because a plan handed to a call whose geometry differs is a wrong
  number rather than a slow path.
* **The bias broadcast.** MPSGraph's convolution selector runs the convolution
  alone, so the bias is an add against the result. The only bias shape that
  broadcasts is ``(1, OC, 1, 1)``; ``(1, 1, 1, OC)`` crashes the MPSGraph
  compiler and ``broadcastTensor:toShape:`` expands along the wrong axes, so the
  bias cases below check the shape was reshaped rather than broadcast.
* **The widths.** float32, float16 and bfloat16 are all served: a bfloat16
  convolution came back bit-identical to a bfloat16 reference convolution on an
  M5, which is why this gate is wider than the softmax one. float64 has no
  MPSGraph element type and the integral and bool widths are the reference's own
  errors, so all three take the CPU round-trip and get exactly that.
* **One dimension is a unit-high two-dimensional convolution.** MPSGraph has
  no 1-D selector, so a 1-D call is served through its 2-D selector with a
  reshape on each side of the node. The cases below cover its geometry, its
  depthwise form and its symmetric padding, and the dispatch case proves the
  result did not come from the CPU round-trip.

Every case self-skips off an MPS build, like the stage-1 and stage-2 suites.
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

TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}

# float16 and bfloat16 only carry ~3 decimal digits, so an operand rounded once
# in the reference and once in the graph can differ by more than the float32
# tolerance above without being a defect.
LOW_PRECISION_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    return TOLERANCE if dtype == torch.float32 else LOW_PRECISION_TOLERANCE


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(
        out.cpu().float(), expected.float(), **_tolerance(expected.dtype)
    )


class TestConv2dCorrectness:
    """``F.conv2d`` on the flagos device equals the CPU reference."""

    @pytest.mark.anyplatform
    def test_plain_3x3(self):
        host_x = torch.randn(2, 3, 16, 16)
        host_w = torch.randn(8, 3, 3, 3)
        host_b = torch.randn(8)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x), _to_device(host_w), _to_device(host_b), padding=1
            ),
            torch.nn.functional.conv2d(host_x, host_w, host_b, padding=1),
        )

    @pytest.mark.anyplatform
    def test_without_bias(self):
        """A bias-less call hands the graph one operand fewer."""
        host_x = torch.randn(2, 3, 8, 8)
        host_w = torch.randn(5, 3, 3, 3)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x), _to_device(host_w), padding=1
            ),
            torch.nn.functional.conv2d(host_x, host_w, padding=1),
        )

    @pytest.mark.anyplatform
    def test_strided(self):
        host_x = torch.randn(2, 3, 16, 16)
        host_w = torch.randn(8, 3, 3, 3)
        host_b = torch.randn(8)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x),
                _to_device(host_w),
                _to_device(host_b),
                stride=2,
                padding=1,
            ),
            torch.nn.functional.conv2d(host_x, host_w, host_b, stride=2, padding=1),
        )

    @pytest.mark.anyplatform
    def test_dilated(self):
        host_x = torch.randn(2, 3, 16, 16)
        host_w = torch.randn(8, 3, 3, 3)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x), _to_device(host_w), padding=2, dilation=2
            ),
            torch.nn.functional.conv2d(host_x, host_w, padding=2, dilation=2),
        )

    @pytest.mark.anyplatform
    def test_pointwise_kernel(self):
        """A 1x1 kernel is the shape a projection layer uses."""
        host_x = torch.randn(2, 3, 16, 16)
        host_w = torch.randn(8, 3, 1, 1)
        host_b = torch.randn(8)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x), _to_device(host_w), _to_device(host_b)
            ),
            torch.nn.functional.conv2d(host_x, host_w, host_b),
        )

    @pytest.mark.anyplatform
    def test_depthwise(self):
        """``groups == channels``: the per-channel kernel a mixer uses."""
        host_x = torch.randn(2, 4, 8, 8)
        host_w = torch.randn(4, 1, 3, 3)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x), _to_device(host_w), padding=1, groups=4
            ),
            torch.nn.functional.conv2d(host_x, host_w, padding=1, groups=4),
        )

    @pytest.mark.anyplatform
    def test_grouped(self):
        host_x = torch.randn(2, 4, 8, 8)
        host_w = torch.randn(6, 2, 3, 3)
        host_b = torch.randn(6)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x),
                _to_device(host_w),
                _to_device(host_b),
                padding=1,
                groups=2,
            ),
            torch.nn.functional.conv2d(host_x, host_w, host_b, padding=1, groups=2),
        )

    @pytest.mark.anyplatform
    def test_asymmetric_padding(self):
        """The padding list is a front/back pair per axis, not a symmetric pair.

        ``nn.Conv2d(padding=(1, 2))`` pads the height by 1 on both sides and the
        width by 2, and the graph is handed the flat
        ``[front_0, front_1, back_0, back_1]`` list ``MpsConvResolve`` splits.
        A kernel reading the list as symmetric pairs would build a descriptor
        that pads the wrong axis.
        """
        module = torch.nn.Conv2d(3, 4, 3, padding=(1, 2))
        host_x = torch.randn(2, 3, 8, 8)
        _assert_matches_cpu(module(_to_device(host_x)), module(host_x))

    @pytest.mark.parametrize(
        "kernel,stride,padding,dilation",
        [
            ((5, 5), (1, 1), (2, 2), (1, 1)),
            ((3, 3), (2, 2), (1, 1), (2, 2)),  # strided and dilated together
            ((1, 3), (1, 1), (0, 1), (1, 1)),  # anisotropic kernel and padding
            ((5, 1), (1, 1), (2, 0), (1, 1)),
            ((2, 4), (3, 1), (0, 0), (1, 1)),
            ((4, 2), (2, 2), (3, 1), (1, 1)),
            ((3, 3), (1, 1), (2, 3), (1, 2)),
        ],
    )
    @pytest.mark.anyplatform
    def test_anisotropic_geometry(self, kernel, stride, padding, dilation):
        """The per-axis mapping between aten's order and the descriptor's.

        aten's axis order and MPSGraph's X/Y/Z properties both run
        fastest-moving-last, and the kernel maps between them with
        ``p = rank - 1 - axis``. Reading it the other way round is a wrong
        number for any non-square kernel rather than a shape error, so each case
        here moves exactly one axis and the reference is the same call on the
        host.
        """
        host_x = torch.randn(2, 3, 12, 12)
        host_w = torch.randn(4, 3, *kernel)
        host_b = torch.randn(4)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x),
                _to_device(host_w),
                _to_device(host_b),
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            torch.nn.functional.conv2d(
                host_x,
                host_w,
                host_b,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_every_served_dtype(self, dtype):
        host_x = torch.randn(2, 3, 8, 8, dtype=dtype)
        host_w = torch.randn(4, 3, 3, 3, dtype=dtype)
        host_b = torch.randn(4, dtype=dtype)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(
                _to_device(host_x), _to_device(host_w), _to_device(host_b), padding=1
            ),
            torch.nn.functional.conv2d(host_x, host_w, host_b, padding=1),
        )

    @pytest.mark.anyplatform
    def test_conv_transpose2d(self):
        """``nn.ConvTranspose2d`` declines to the CPU round-trip and stays right.

        A transposed convolution reaches this selector with ``transposed=True``,
        and MPSGraph spells it with a different selector whose channel roles are
        swapped. The kernel declines it rather than folding it into the forward
        node, and the answer is the reference's.
        """
        module = torch.nn.ConvTranspose2d(3, 2, 3)
        host_x = torch.randn(1, 3, 5, 5)
        _assert_matches_cpu(module(_to_device(host_x)), module(host_x))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        """A non-contiguous operand takes the same answer as the CPU.

        MPSGraph reads dense memory, so the kernel materialises the operand
        first. The narrowing leaves an unaligned innermost row too.
        """
        host_x = torch.randn(1, 3, 8, 8)
        host_w = torch.randn(4, 3, 3, 3)
        a = _to_device(host_x)[:, :, ::2, ::2]
        assert not a.is_contiguous()
        _assert_matches_cpu(
            torch.nn.functional.conv2d(a, _to_device(host_w)),
            torch.nn.functional.conv2d(host_x[:, :, ::2, ::2], host_w),
        )

    @pytest.mark.anyplatform
    def test_empty_batch(self):
        """A zero-length batch is an empty result, returned as one."""
        out = torch.nn.functional.conv2d(
            _to_device(torch.randn(0, 2, 4, 4)),
            _to_device(torch.randn(3, 2, 3, 3)),
        )
        assert out.shape == (0, 3, 2, 2)
        assert out.device.type == "flagos"


class TestConv3dCorrectness:
    """``F.conv3d`` takes the 3-D selector and its own per-axis mapping."""

    @pytest.mark.anyplatform
    def test_plain_3x3x3(self):
        host_x = torch.randn(1, 2, 6, 6, 6)
        host_w = torch.randn(3, 2, 3, 3, 3)
        host_b = torch.randn(3)
        _assert_matches_cpu(
            torch.nn.functional.conv3d(
                _to_device(host_x), _to_device(host_w), _to_device(host_b), padding=1
            ),
            torch.nn.functional.conv3d(host_x, host_w, host_b, padding=1),
        )

    @pytest.mark.anyplatform
    def test_asymmetric_padding(self):
        module = torch.nn.Conv3d(2, 3, 3, padding=(0, 1, 2))
        host_x = torch.randn(1, 2, 6, 6, 6)
        _assert_matches_cpu(module(_to_device(host_x)), module(host_x))

    @pytest.mark.parametrize(
        "kernel,stride,padding",
        [
            ((2, 3, 4), (1, 1, 1), (0, 1, 2)),  # all three axes distinct
            ((4, 1, 3), (1, 1, 1), (2, 0, 1)),
            ((3, 2, 2), (1, 2, 3), (1, 0, 0)),  # per-axis stride
        ],
    )
    @pytest.mark.anyplatform
    def test_anisotropic_geometry(self, kernel, stride, padding):
        """The 3-D case: a transpose between two axes shows up as a mismatch.

        Only the axes the kernel moves are compared against the reference, so a
        descriptor that read Z for X would give a different number here rather
        than a shape error.
        """
        host_x = torch.randn(1, 2, 8, 8, 8)
        host_w = torch.randn(3, 2, *kernel)
        host_b = torch.randn(3)
        _assert_matches_cpu(
            torch.nn.functional.conv3d(
                _to_device(host_x),
                _to_device(host_w),
                _to_device(host_b),
                stride=stride,
                padding=padding,
            ),
            torch.nn.functional.conv3d(
                host_x, host_w, host_b, stride=stride, padding=padding
            ),
        )


class TestConv1dCorrectness:
    """``F.conv1d`` takes the 2-D selector over a unit-high view."""

    @pytest.mark.anyplatform
    def test_plain_3x3(self):
        host_x = torch.randn(2, 3, 16)
        host_w = torch.randn(8, 3, 3)
        host_b = torch.randn(8)
        _assert_matches_cpu(
            torch.nn.functional.conv1d(
                _to_device(host_x), _to_device(host_w), _to_device(host_b), padding=1
            ),
            torch.nn.functional.conv1d(host_x, host_w, host_b, padding=1),
        )

    @pytest.mark.anyplatform
    def test_without_bias(self):
        host_x = torch.randn(2, 3, 16)
        host_w = torch.randn(5, 3, 3)
        _assert_matches_cpu(
            torch.nn.functional.conv1d(
                _to_device(host_x), _to_device(host_w), padding=1
            ),
            torch.nn.functional.conv1d(host_x, host_w, padding=1),
        )

    @pytest.mark.parametrize(
        "kernel,stride,padding,dilation,groups",
        [
            (5, 2, 2, 1, 1),
            (3, 1, 2, 2, 1),
            (1, 3, 0, 1, 1),
            (3, 2, 0, 1, 4),
        ],
    )
    @pytest.mark.anyplatform
    def test_geometry(self, kernel, stride, padding, dilation, groups):
        """The length geometry reaches the folded axis unchanged."""
        host_x = torch.randn(2, 4, 16)
        host_w = torch.randn(4, 4 // groups, kernel)
        host_b = torch.randn(4)
        _assert_matches_cpu(
            torch.nn.functional.conv1d(
                _to_device(host_x),
                _to_device(host_w),
                _to_device(host_b),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
            torch.nn.functional.conv1d(
                host_x,
                host_w,
                host_b,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_every_served_dtype(self, dtype):
        host_x = torch.randn(2, 3, 12, dtype=dtype)
        host_w = torch.randn(4, 3, 3, dtype=dtype)
        host_b = torch.randn(4, dtype=dtype)
        _assert_matches_cpu(
            torch.nn.functional.conv1d(
                _to_device(host_x), _to_device(host_w), _to_device(host_b), padding=1
            ),
            torch.nn.functional.conv1d(host_x, host_w, host_b, padding=1),
        )

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host_x = torch.randn(2, 3, 16)
        host_w = torch.randn(4, 3, 3)
        device_x = _to_device(host_x)[:, :, ::2]
        assert not device_x.is_contiguous()
        _assert_matches_cpu(
            torch.nn.functional.conv1d(device_x, _to_device(host_w)),
            torch.nn.functional.conv1d(host_x[:, :, ::2], host_w),
        )

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The answer comes from the kernel, not the CPU round-trip."""
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "x = torch.randn(1, 3, 8, device='flagos:0'); "
                "w = torch.randn(4, 3, 3, device='flagos:0'); "
                "torch.nn.functional.conv1d(x, w, padding=1)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (
            result.stderr.count("[flagos dispatch] convolution_overrideable -> apple")
            == 1
        )


class TestConvDeclined:
    """The cases the graph does not serve stay correct through the round-trip."""

    @pytest.mark.anyplatform
    def test_float64_is_correct(self):
        host_x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
        host_w = torch.randn(3, 2, 3, 3, dtype=torch.float64)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(_to_device(host_x), _to_device(host_w)),
            torch.nn.functional.conv2d(host_x, host_w),
        )

    @pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
    @pytest.mark.anyplatform
    def test_integral_is_correct(self, dtype):
        """An integral convolution is declined; the round-trip still computes it."""
        host_x = torch.randint(0, 3, (1, 2, 4, 4)).to(dtype)
        host_w = torch.randint(0, 3, (3, 2, 3, 3)).to(dtype)
        _assert_matches_cpu(
            torch.nn.functional.conv2d(_to_device(host_x), _to_device(host_w)),
            torch.nn.functional.conv2d(host_x, host_w),
        )

    @pytest.mark.anyplatform
    def test_bool_raises_exactly_like_cpu(self):
        host_x = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
        host_w = torch.zeros(3, 2, 3, 3, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            torch.nn.functional.conv2d(_to_device(host_x), _to_device(host_w))

    @pytest.mark.anyplatform
    def test_kernel_larger_than_input_raises_exactly_like_cpu(self):
        host_x = torch.randn(1, 2, 2, 2)
        host_w = torch.randn(3, 2, 5, 5)
        with pytest.raises(RuntimeError):
            torch.nn.functional.conv2d(_to_device(host_x), _to_device(host_w))

    @pytest.mark.anyplatform
    def test_mixed_precision_raises_exactly_like_cpu(self):
        """A float operand with a double weight is the reference's own error."""
        host_x = torch.randn(1, 2, 4, 4)
        host_w = torch.randn(3, 2, 3, 3).double()
        with pytest.raises(RuntimeError):
            torch.nn.functional.conv2d(_to_device(host_x), _to_device(host_w))


class TestConvScheduleAndPlanReuse:
    """The geometry a plan is compiled against (mps_common.mm)."""

    @pytest.mark.anyplatform
    def test_interleaved_geometry_never_share_a_plan(self):
        """Each stride, padding, dilation, group count and bias flag gets its own plan.

        A plan key that named only the op would let the first graph built serve
        every later call, and the geometry lives in the descriptor rather than in
        the operand shapes, so a stride-1 graph answering a stride-2 request is a
        wrong number. Rotating through the signatures puts a different geometry
        on a warm cache every step.
        """
        torch.manual_seed(7)
        host_x = torch.randn(2, 6, 9, 9)
        host_b = torch.randn(6)
        weight_plain = torch.randn(6, 6, 3, 3)
        weight_grouped = torch.randn(6, 2, 3, 3)
        cases = [
            ("plain", dict(padding=1), True, weight_plain),
            ("strided", dict(stride=2, padding=1), True, weight_plain),
            ("dilated", dict(dilation=2, padding=2), True, weight_plain),
            ("grouped", dict(groups=3, padding=1), True, weight_grouped),
            ("unpadded", dict(), True, weight_plain),
            ("biasless", dict(padding=1), False, weight_plain),
        ]

        def build(kwargs, use_bias, weight):
            module = torch.nn.Conv2d(
                6,
                6,
                3,
                groups=kwargs.get("groups", 1),
                bias=use_bias,
                **{k: v for k, v in kwargs.items() if k != "groups"},
            )
            with torch.no_grad():
                module.weight.copy_(weight)
                if use_bias:
                    module.bias.copy_(host_b)
            return module

        wanted = {name: build(kw, ub, w)(host_x) for name, kw, ub, w in cases}
        device_x = _to_device(host_x)

        for round_ in range(3):
            for offset in range(len(cases)):
                name, kwargs, use_bias, weight = cases[(offset + round_) % len(cases)]
                out = build(kwargs, use_bias, weight)(device_x)
                _assert_matches_cpu(out, wanted[name])

    @pytest.mark.anyplatform
    def test_results_survive_block_reuse_and_immediate_readback(self):
        """Nothing waits for the GPU, so both ends of the call are checked here.

        The operands' blocks go back to the pool as soon as the caller drops them,
        while the graph reading them has only been scheduled; and the value the
        caller reads back is produced by work the runtime has not awaited. Both
        rest on the sentinel command buffer the runtime commits behind the graph
        (see mps_common.h), so a call that returned before its work was visible
        to StreamQuery would show up here as a wrong result.
        """
        held = []
        for _ in range(16):
            x = torch.randn(1, 4, 32, 32, device=DEVICE)
            w = torch.randn(8, 4, 3, 3, device=DEVICE)
            wanted = torch.nn.functional.conv2d(x.cpu(), w.cpu(), padding=1)
            out = torch.nn.functional.conv2d(x, w, padding=1)
            del x, w  # recycled while the graph above may still be reading them
            for _ in range(4):  # blocks the allocator is free to hand back out
                torch.nn.functional.conv2d(
                    torch.randn(1, 4, 32, 32, device=DEVICE),
                    torch.randn(8, 4, 3, 3, device=DEVICE),
                    padding=1,
                )
            held.append((out.cpu(), wanted))  # read back without a synchronize

        for got, wanted in held:
            torch.testing.assert_close(got, wanted, **TOLERANCE)


class TestConvDispatch:
    """``convolution_overrideable`` is claimed on PrivateUse1."""

    @pytest.mark.main_ops
    def test_conf_routes_convolution_to_mps(self):
        assert routed_backend_or_none("convolution_overrideable") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """The ``out=`` form is deliberately unregistered, like every ``.out``."""
        assert routed_backend_or_none("convolution_overrideable.out") is None

    @pytest.mark.main_ops
    @pytest.mark.parametrize(
        "call,shape",
        [
            (
                "torch.nn.functional.conv2d(a, b, padding=1)",
                "a = torch.randn(1, 3, 8, 8, device='flagos:0'); "
                "b = torch.randn(4, 3, 3, 3, device='flagos:0')",
            ),
            (
                "torch.nn.functional.conv3d(a, b, padding=1)",
                "a = torch.randn(1, 2, 4, 4, 4, device='flagos:0'); "
                "b = torch.randn(3, 2, 3, 3, 3, device='flagos:0')",
            ),
        ],
    )
    def test_dispatch_log_names_mps_for_the_public_spellings(self, call, shape):
        """``F.conv2d``/``F.conv3d`` reach the kernel through ``convolution_overrideable``.

        This is the case that shows why ``aten.convolution`` is not registered:
        it decomposes above the backend key, so the log line for the op that
        actually runs is the one to hold. Run in a subprocess because
        ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import torch, torch_fl, torch.nn.functional as F; {shape}; {call}",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] convolution_overrideable -> apple" in result.stderr
