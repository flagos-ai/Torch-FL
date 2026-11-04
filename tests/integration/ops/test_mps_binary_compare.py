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

"""Apple Silicon (Metal / MPSGraph) elementwise comparisons and max/min.

Two gaps closed here, and they are two because they were two different
mistakes in the coverage rather than one family:

* **The comparison family was 3/12 served.** `eq.Scalar`, `ne.Scalar` and
  `le.Tensor` had been routed; the other nine spellings -- `eq`/`ne`/`lt`/`gt`/
  `ge` over tensors, and `lt`/`le`/`gt`/`ge` over a scalar -- were on the
  fallback. A model compares constantly (every mask, every branch, every
  calibration pass), so this was a large silent cost.
* **`maximum` and `minimum` were not served at all.** They are the same
  two-operand shape as `add`/`mul` and one MPSGraph selector each, so they are
  a row in the existing `binary` category rather than a new body.

What the cases below are actually about:

* **The result is bool, the operands keep their width.** Every comparison
  answers a `torch.bool` tensor whatever the operands are, which is a different
  node result from the arithmetic family and is asserted for each op.
* **NaN never compares.** `eq(nan, nan)` is false, `ne(nan, nan)` is true, and
  the four orderings are all false against a NaN. That is not a rounding detail
  -- it is the definition of the comparison, and a graph that answered anything
  else would be wrong for every `isnan` guard written in torch.
* **`maximum`/`minimum` propagate NaN** while `fmax`/`fmin` would not; the
  reference propagates, and the case pins that rather than the "obvious" choice.
* **Mixed widths take the round-trip.** aten promotes the pair before comparing;
  the gate serves equal widths only, so the promoted cases stay correct.

Every case self-skips off an MPS build.
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

# The tensor-spelling comparisons, as (aten name, callable over two tensors).
TENSOR_COMPARISONS = {
    "eq": torch.eq,
    "ne": torch.ne,
    "lt": torch.lt,
    "le": torch.le,
    "gt": torch.gt,
    "ge": torch.ge,
}

# The scalar-spelling ones take the opaque aten overload, which is the one that
# dispatches to the backend key.
SCALAR_COMPARISONS = {
    "lt": torch.ops.aten.lt.Scalar,
    "le": torch.ops.aten.le.Scalar,
    "gt": torch.ops.aten.gt.Scalar,
    "ge": torch.ops.aten.ge.Scalar,
}

BINARY_PICKS = {
    "maximum": torch.maximum,
    "minimum": torch.minimum,
}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _randn(shape) -> torch.Tensor:
    """`torch.randn` for a shape tuple, including the rank-0 one.

    ``torch.randn(*())`` passes no arguments at all, which is a TypeError rather
    than a scalar; the empty tuple is spelled out for that case.
    """
    return torch.randn(shape) if shape else torch.tensor(float(torch.randn(1)))


def _same_bool(got: torch.Tensor, expected: torch.Tensor) -> bool:
    assert got.dtype == torch.bool, got.dtype
    assert got.shape == expected.shape, (got.shape, expected.shape)
    return torch.equal(got, expected)


class TestTensorComparisons:
    """Each tensor-spelling comparison equals the reference, as a bool tensor."""

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, name, dtype):
        fn = TENSOR_COMPARISONS[name]
        a = torch.tensor([-2.5, -1.0, -0.0, 0.0, 1.0, 2.5], dtype=dtype)
        b = torch.tensor([-1.0, -1.0, 0.0, -0.0, 2.5, 2.5], dtype=dtype)
        out = fn(_to_device(a), _to_device(b))
        assert out.device.type == "flagos"
        assert _same_bool(out.cpu(), fn(a, b)), name

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_nan_never_compares(self, name):
        """The definition of a comparison, not a rounding detail.

        `eq(nan, nan)` is false, `ne(nan, nan)` is true, and none of the four
        orderings holds against a NaN. Every `isnan` guard in torch is written
        on this rule.
        """
        fn = TENSOR_COMPARISONS[name]
        a = torch.tensor([float("nan"), 1.0, float("nan")])
        b = torch.tensor([float("nan"), float("nan"), 1.0])
        out = fn(_to_device(a), _to_device(b)).cpu()
        assert _same_bool(out, fn(a, b)), name
        assert out[0].item() == (name == "ne"), (name, out)

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_infinities(self, name):
        fn = TENSOR_COMPARISONS[name]
        a = torch.tensor([float("-inf"), float("inf"), 0.0, float("inf")])
        b = torch.tensor([float("-inf"), float("inf"), float("inf"), 0.0])
        assert _same_bool(fn(_to_device(a), _to_device(b)).cpu(), fn(a, b)), name

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_broadcast_pairs(self, name):
        fn = TENSOR_COMPARISONS[name]
        for shape_a, shape_b in (
            ((4, 5), (5,)),
            ((4, 5), (4, 1)),
            ((2, 3, 4), (3, 1)),
            ((4, 5), ()),
        ):
            a = _randn(shape_a)
            b = _randn(shape_b)
            assert _same_bool(fn(_to_device(a), _to_device(b)).cpu(), fn(a, b)), (
                name,
                shape_a,
                shape_b,
            )

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self, name):
        fn = TENSOR_COMPARISONS[name]
        a = torch.randn(4, 5)  # 20-byte float32 rows
        b = torch.randn(4, 5)
        assert _same_bool(fn(_to_device(a), _to_device(b)).cpu(), fn(a, b))

        a_strided = _to_device(torch.randn(8))[::2]
        b_strided = _to_device(torch.randn(8))[::2]
        assert not a_strided.is_contiguous()
        assert _same_bool(
            fn(a_strided, b_strided).cpu(), fn(a_strided.cpu(), b_strided.cpu())
        )

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_rank_zero_and_empty(self, name):
        fn = TENSOR_COMPARISONS[name]
        a, b = torch.tensor(1.5), torch.tensor(2.5)
        out = fn(_to_device(a), _to_device(b))
        assert out.shape == () and out.dtype == torch.bool
        assert out.item() == fn(a, b).item()
        assert fn(_to_device(torch.empty(0)), _to_device(torch.empty(0))).numel() == 0


class TestScalarComparisons:
    """The scalar spellings, which is how a threshold is written."""

    @pytest.mark.parametrize("name", sorted(SCALAR_COMPARISONS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, name, dtype):
        fn = SCALAR_COMPARISONS[name]
        a = torch.tensor([-1.0, -0.0, 0.0, 1.0, 2.0], dtype=dtype)
        out = fn(_to_device(a), 1.0)
        assert out.dtype == torch.bool
        assert _same_bool(out.cpu(), fn(a, 1.0)), name

    @pytest.mark.parametrize("name", sorted(SCALAR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_the_scalar_is_part_of_the_plan_key(self, name):
        """Two thresholds over one operand must not share a compiled plan.

        The scalar is a host value the body bakes into the graph, so the two
        calls describe identical tensors at identical shapes; the key is what
        keeps them apart. The answers differ, which is what this checks.
        """
        fn = SCALAR_COMPARISONS[name]
        a = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        device_a = _to_device(a)
        for threshold in (0.5, 1.5, 0.5):
            assert _same_bool(fn(device_a, threshold).cpu(), fn(a, threshold)), (
                name,
                threshold,
            )

    @pytest.mark.parametrize("name", sorted(SCALAR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_nan_operand(self, name):
        fn = SCALAR_COMPARISONS[name]
        a = torch.tensor([float("nan"), 1.0])
        assert _same_bool(fn(_to_device(a), 1.0).cpu(), fn(a, 1.0)), name

    @pytest.mark.parametrize("name", sorted(SCALAR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_an_integral_scalar_beside_a_float_operand(self, name):
        """`x < 1` where `x` is float: the scalar promotes, the result is bool."""
        fn = SCALAR_COMPARISONS[name]
        a = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        out = fn(_to_device(a), 1)
        assert _same_bool(out.cpu(), fn(a, 1)), name


class TestBinaryPicks:
    """`maximum` and `minimum` select elementwise, as the same binary body."""

    @pytest.mark.parametrize("name", sorted(BINARY_PICKS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, name, dtype):
        fn = BINARY_PICKS[name]
        a = torch.tensor([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=dtype)
        b = torch.tensor([1.0, -2.0, 0.5, -0.5, 2.5], dtype=dtype)
        out = fn(_to_device(a), _to_device(b))
        assert out.dtype == dtype
        torch.testing.assert_close(out.cpu().float(), fn(a, b).float())

    @pytest.mark.parametrize("name", sorted(BINARY_PICKS))
    @pytest.mark.anyplatform
    def test_nan_propagates(self, name):
        """The reference propagates a NaN here, unlike `fmax`/`fmin`.

        A graph built from a comparison-and-select would answer the finite
        operand instead, which is the wrong rule for these two ops.
        """
        fn = BINARY_PICKS[name]
        a = torch.tensor([float("nan"), 1.0, float("nan")])
        b = torch.tensor([1.0, float("nan"), float("nan")])
        out = fn(_to_device(a), _to_device(b)).cpu()
        expected = fn(a, b)
        assert torch.isnan(out).equal(torch.isnan(expected)), (name, out, expected)

    @pytest.mark.parametrize("name", sorted(BINARY_PICKS))
    @pytest.mark.anyplatform
    def test_broadcast_and_row_pitch(self, name):
        fn = BINARY_PICKS[name]
        for shape_a, shape_b in (((4, 5), (5,)), ((4, 5), (4, 1)), ((4, 5), ())):
            a = _randn(shape_a)
            b = _randn(shape_b)
            torch.testing.assert_close(fn(_to_device(a), _to_device(b)).cpu(), fn(a, b))


class TestDeclines:
    """Cases the gates do not serve stay correct through the reference."""

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_a_mixed_width_pair_round_trips(self, name):
        """aten promotes the pair before comparing; the graph serves equal widths.

        So a float32 against an int64 operand takes the round-trip and gets the
        reference's promotion rather than a graph built at one of the two types.
        """
        fn = TENSOR_COMPARISONS[name]
        a = torch.tensor([-1.0, 0.0, 1.5, 2.0], dtype=torch.float32)
        b = torch.tensor([-1, 0, 1, 2], dtype=torch.int64)
        out = fn(_to_device(a), _to_device(b))
        assert out.dtype == torch.bool
        assert _same_bool(out.cpu(), fn(a, b)), name

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_float64_round_trips(self, name):
        fn = TENSOR_COMPARISONS[name]
        a = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        b = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
        assert _same_bool(fn(_to_device(a), _to_device(b)).cpu(), fn(a, b)), name


class TestRouting:
    """Every served spelling is routed here, and the log says so."""

    @pytest.mark.parametrize("name", sorted(TENSOR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_the_tensor_form_is_routed(self, name):
        assert routed_backend_or_none(f"{name}.Tensor") in ("apple", "mps")

    @pytest.mark.parametrize("name", sorted(SCALAR_COMPARISONS))
    @pytest.mark.anyplatform
    def test_the_scalar_form_is_routed(self, name):
        assert routed_backend_or_none(f"{name}.Scalar") in ("apple", "mps")

    @pytest.mark.parametrize("name", sorted(BINARY_PICKS))
    @pytest.mark.anyplatform
    def test_the_binary_pick_is_routed(self, name):
        assert routed_backend_or_none(name) in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_the_dispatch_log_names_the_backend(self):
        ops = sorted(TENSOR_COMPARISONS) + sorted(SCALAR_COMPARISONS)
        picks = sorted(BINARY_PICKS)
        script = (
            "import torch, torch_fl\n"
            f"d = torch.device({DEVICE!r})\n"
            "a = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=d)\n"
            "b = torch.tensor([0.0, 0.0, 2.0, 1.0], device=d)\n"
            + "".join(f"torch.{n}(a, b)\n" for n in ops)
            # `maximum`/`minimum` have no `.Scalar` overload: they are named
            # here by their one spelling.
            + "".join(f"torch.ops.aten.{n}(a, b)\n" for n in picks)
        )
        env = dict(os.environ, FLAGOS_LOG="dispatch", FLAGOS_ACCELERATOR="mps")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        for name in ops + picks:
            # The log line is `[flagos dispatch] eq.Tensor -> mps`, so the op
            # name is a *prefix* of the overload spelling, not a word.
            lines = [
                ln
                for ln in proc.stderr.splitlines()
                if "->" in ln and f"] {name}" in ln
            ]
            assert any("-> apple" in ln or "-> mps" in ln for ln in lines), (
                f"no dispatch line named the backend for {name}:\n" + "\n".join(lines)
            )
