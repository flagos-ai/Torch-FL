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

"""Dtype escape from the FlagGems route, on Ascend and on GCU.

A vendor conf is a per-op routing table, so it cannot express "FlagGems, except
for float64". On Ascend that exception is real and broad: BiShengHIR rejects
the float64 instantiation of nearly every kernel FlagGems' pointwise codegen
produces -- measured on Ascend910 / CANN 9.0.0 / FlagTree 0.6.2a1+ascend3.5,
add/sub/div/neg/abs/exp/log/sqrt/reciprocal/where/clamp/fill_/zeros_like/
ones_like/ones/full/arange over float64 all raise MLIRCompilationError, while
mul, cat and the comparisons compile -- so the conf has no way to state it. The
exception lives at runtime instead: ``FlagGemsRejectsDtype`` in
``csrc/aten/common.cc``, consulted by ``Dispatcher::ResolveFn``.

The same escape also covers the gaps that are neither per-op nor per-dtype but
the intersection of the two: ``neg`` over bool. ``flag_gems/ops/neg.py`` is an
unguarded pointwise ``-x``, so a bool operand is code-generated like any other
element type and lowers to ``hivm.hir.vadd`` over ``i1``, which BiShengIR
refuses to verify -- while FlagGems serves a bool operand fine for add, sub,
abs and the comparisons, and ``neg`` fine for every other dtype. That case is
``FlagGemsRejectsOpDtype``, consulted by the same ``ResolveFn``; the reference
behaviour it restores is the ``RuntimeError`` the CPU raises.

On GCU the dtype-wide predicate carries int64 instead, for a version-skew
reason rather than a hardware one: flag_gems 5.3.2 has its gcu300 pointwise
codegen append ``enable_i64=true`` to a pass whose binary -- the installed
``gcu-compiler-opt`` -- does not declare that option, so an i64 operand dies at
``no such option enable_i64`` before any kernel is produced; a kernel that gets
past that raises ``Pipeline run failed`` from the compiler's own 64-bit check.
The escape reaches the generated vendor kernel instead, and for
``remainder.Tensor`` it also repairs a wrong answer rather than a failure: the
FlagGems route returns an int32 tensor for an int64 input.

The escape is a per-dtype fallback *into the vendor slot*, so it can only move
an op that has one. On GCU that is the intersection of the ops the conf leaves
on FlagGems and the ops the GCU codegen registered a kernel for -- exactly the
seven the conf marks ``# gcu``, and all seven move: ``clamp``, ``fmod.Tensor``,
``gelu``, ``mean``, ``mean.dim``, ``remainder.Tensor`` and ``silu``.
``clamp_min`` is the negative control below: it is on the FlagGems route with no
vendor kernel, so the predicate is consulted and changes nothing. The 53 routes
the GCU survey records as failing on its `2d-i64` profile alone are all in that
position -- none of them has a vendor kernel, which is why the survey's own
numbers cannot move for any of this.

Only three of the seven *succeed* on int64 afterwards. ``remainder.Tensor`` is
the one that argues the change on correctness rather than availability: its
FlagGems route does not fail at all, it answers an int64 operand with an int32
tensor. The other four fail on both sides, but the failure changes: the
generated GCU template gates on ``TopsatenSupportsDtype``, which excludes int64,
so those calls reach the template's host round-trip and raise the reference
error a CPU caller sees, where FlagGems raised a compiler abort.

These tests are the CI-visible contract for that fallback: only the dtype
routes change, the dtypes FlagGems does serve keep the configured route, and
the vendor answer for float64 is numerically right rather than merely
non-crashing.

Usage:
    pytest tests/integration/ops/test_dtype_route_fallback.py -v
"""

import os
import re
import subprocess
import sys

import pytest
import torch

from backend_conf import routed_backend

DEVICE = "flagos:0"


# One subprocess covers both directions: same op names, same shapes, only the
# dtype differing. Running both in one process also proves the per-op backend
# cache (Dispatcher::cached_backend_) does not pin an op to one backend for
# good -- the route is re-derived from the arguments on every call.
_PROBE = r"""
import sys
import torch
import torch_fl

for dtype in (torch.float32, torch.float64):
    print(f"### {dtype}", file=sys.stderr, flush=True)
    a = torch.tensor([1.0, 2.0, 4.0], dtype=dtype).to("flagos")
    a + a
    a.reciprocal()
    a.sum()
    torch.ones(4, device="flagos", dtype=dtype)
    torch.zeros_like(a)
    torch.mul(a, a)
    a.clone().fill_(2.0)
"""

# Ops the Ascend conf routes to FlagGems and whose float64 instantiation the
# Ascend compiler rejects; each is reached through a different argument shape
# (Tensor operand, Tensor list, bare ScalarType).
_FALLBACK_OPS = ("add.Tensor", "reciprocal", "sum", "ones", "zeros_like")

# Already on the vendor kernel in the conf, whatever the dtype. It is the
# control: the fallback must not disturb a route the conf made itself.
_VENDOR_OPS = ("mul.Tensor",)

# The (op, dtype) probe. One process for all three dtypes, which also re-proves
# that the per-op backend cache (Dispatcher::cached_backend_) does not pin `neg`
# to whatever the first call resolved to. The bool call is expected to raise, so
# the probe catches it and reports the outcome rather than the process failing.
_OP_DTYPE_PROBE = r"""
import sys
import torch
import torch_fl

for dtype in (torch.float32, torch.int64, torch.bool):
    print(f"### {dtype}", file=sys.stderr, flush=True)
    a = torch.ones(8, device="flagos", dtype=dtype)
    try:
        torch.neg(a)
        print("### outcome ok", file=sys.stderr, flush=True)
    except RuntimeError as exc:
        first = str(exc).splitlines()[0]
        print(f"### outcome raised {first}", file=sys.stderr, flush=True)
"""


# The GCU probe. Two dtypes in one process, so the per-op backend cache
# (Dispatcher::cached_backend_) is re-proved not to pin an op to whatever the
# first call resolved to: every op below has to report two different backends
# within the same run. Each call is recorded twice -- the dispatch line says
# which route it took, and an ``### outcome`` line says what came back, because
# four of the seven routes move and still raise. `clamp_min` is the negative
# control -- on the FlagGems route with no vendor kernel behind it, so the
# predicate is consulted and cannot move it; that call still fails inside the
# compiler, which is exactly what the escape cannot fix and why the route, not
# the outcome, is what this file asserts about it.
_GCU_PROBE = r"""
import sys
import torch
import torch_fl

CALLS = {
    "clamp": lambda a, b: a.clamp(0, 4),
    "fmod.Tensor": lambda a, b: a.fmod(b),
    "gelu": lambda a, b: torch.ops.aten.gelu(a),
    "mean": lambda a, b: a.mean(),
    "mean.dim": lambda a, b: a.mean(0),
    "remainder.Tensor": lambda a, b: a.remainder(b),
    "silu": lambda a, b: torch.ops.aten.silu(a),
    "clamp_min": lambda a, b: a.clamp_min(1),
}

for dtype in (torch.int64, torch.float32):
    print(f"### {dtype}", file=sys.stderr, flush=True)
    a = torch.tensor([0, 1, 3, 8], dtype=dtype).to("flagos")
    b = torch.tensor([3, 3, 3, 3], dtype=dtype).to("flagos")
    for name, call in CALLS.items():
        try:
            call(a, b)
            outcome = "ok"
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        print(f"### outcome {name} {outcome}", file=sys.stderr, flush=True)
"""


def _run(extra_env: dict, code: str = _PROBE) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update({"FLAGOS_LOG": "dispatch", **extra_env})
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


def _routes_by_dtype(stderr: str) -> dict[str, dict[str, str]]:
    """``{dtype: {op: backend}}`` from the probe's section markers."""
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in stderr.splitlines():
        if line.startswith("### outcome "):
            continue
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = {}
        elif line.startswith("[flagos dispatch] ") and current is not None:
            op, _, backend = line[len("[flagos dispatch] ") :].partition(" -> ")
            sections[current][op] = backend
    return sections


def _outcomes_by_dtype(stderr: str) -> dict[str, str]:
    """``{dtype: outcome}`` from the (op, dtype) probe's ``###`` markers."""
    outcomes: dict[str, str] = {}
    current = None
    for line in stderr.splitlines():
        if line.startswith("### outcome "):
            if current is not None:
                outcomes[current] = line[len("### outcome ") :].strip()
        elif line.startswith("### "):
            current = line[4:].strip()
    return outcomes


def _gcu_outcomes(stderr: str) -> dict[str, dict[str, str]]:
    """``{dtype: {op: outcome}}`` from the GCU probe's ``### outcome`` markers.

    The per-call outcome is ``ok`` or ``<ExceptionType>: <first line>``; the
    dtype sections are the marker lines that name a dtype, so an outcome line
    cannot be mistaken for one.
    """
    outcomes: dict[str, dict[str, str]] = {}
    current = None
    for line in stderr.splitlines():
        if line.startswith("### outcome "):
            op, _, outcome = line[len("### outcome ") :].partition(" ")
            if current is not None:
                outcomes.setdefault(current, {})[op] = outcome
        elif line.startswith("### ") and line[4:].strip().startswith("torch."):
            current = line[4:].strip()
    return outcomes


@pytest.fixture(scope="module")
def routes():
    result = _run({})
    assert result.returncode == 0, f"probe failed:\n{result.stderr}"
    by_dtype = _routes_by_dtype(result.stderr)
    assert "torch.float32" in by_dtype and "torch.float64" in by_dtype, result.stderr
    return by_dtype


class TestFlagGemsDtypeFallback:
    """float64 leaves the FlagGems route; the other dtypes keep it."""

    @pytest.mark.ascend
    def test_float64_flagems_routes_land_on_the_vendor_kernel(self, routes):
        """Every op the conf sends to FlagGems answers with the native backend.

        ``add.Tensor`` is the op the AMP contract fails on
        (``test_eager_dtype_preservation[torch.float64]``). The others are here
        because the predicate is on the dtype alone: the argument that carries
        it differs -- a Tensor for add/reciprocal, a Tensor list for sum, a
        bare ``ScalarType`` for the two factories -- and all three forms have
        to be seen.
        """
        fp64 = routes["torch.float64"]
        for op in _FALLBACK_OPS:
            assert fp64[op] == "ascend", f"{op} ran on {fp64[op]}"

    @pytest.mark.ascend
    def test_float32_keeps_the_configured_route(self, routes):
        """The escape is dtype-keyed, and does not leak into float32.

        The assertion is against the conf's own value rather than a literal
        ``flagos_python`` so this stays meaningful if the conf is regenerated:
        what must hold is that float32 keeps whatever route the conf chose,
        while float64 does not.
        """
        fp32 = routes["torch.float32"]
        for op in _FALLBACK_OPS:
            assert fp32[op] == routed_backend(op), (
                f"{op} on float32 ran on {fp32[op]}, conf says {routed_backend(op)}"
            )

    @pytest.mark.ascend
    def test_vendor_routes_are_unaffected(self, routes):
        """An op the conf already sends to the vendor kernel behaves identically.

        ``mul.Tensor`` is on ``ascend`` in the conf for every dtype, so it is
        the control for "the fallback only moves ops that needed moving": both
        dtypes must report the same backend, and the dispatch decision must not
        depend on a previous failure (there is no runtime probe -- the dtype is
        known before the call).
        """
        for op in _VENDOR_OPS:
            expected = routed_backend(op)
            for dtype in ("torch.float32", "torch.float64"):
                assert routes[dtype][op] == expected, (
                    f"{op} on {dtype} ran on {routes[dtype][op]}, conf says {expected}"
                )


class TestFlagGemsOpDtypeFallback:
    """bool `neg` leaves the FlagGems route; the other dtypes keep it.

    Neither half decides this on its own, which is why it is not an entry in
    the conf and not the dtype-wide predicate above: bool is a dtype FlagGems
    serves for other ops, and `neg` is an op FlagGems serves for other dtypes.
    The gap is the intersection, and it is stated as such in
    ``FlagGemsRejectsOpDtype``.
    """

    @pytest.mark.ascend
    def test_bool_neg_lands_on_the_vendor_kernel(self):
        result = _run({}, code=_OP_DTYPE_PROBE)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        routes = _routes_by_dtype(result.stderr)
        assert routes["torch.bool"]["neg"] == "ascend", result.stderr

    @pytest.mark.ascend
    def test_bool_neg_raises_the_reference_error(self):
        """The point of the escape: reach the CPU contract, not just a failure.

        The FlagGems route fails inside the compiler, which is neither the
        reference behaviour nor a message a caller can act on. Routed to the
        native kernel the call reaches ``at::neg`` on the host copy, which is
        where ``RuntimeError: Negation, the `-` operator, on a bool tensor is
        not supported.`` is raised -- the contract
        ``tests/integration/test_dtype_coverage.py`` pins.
        """
        result = _run({}, code=_OP_DTYPE_PROBE)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        outcome = _outcomes_by_dtype(result.stderr)["torch.bool"]
        assert outcome.startswith("raised"), outcome
        assert "bool" in outcome, outcome

    @pytest.mark.ascend
    def test_other_dtypes_keep_the_configured_route(self):
        """The escape is per (op, dtype), so it does not leak into `neg` itself.

        float32 and int64 are the two halves the conf entry could not have
        separated: a per-op entry for `neg` would have moved these off FlagGems
        as well (and onto the Ascend template's CPU round-trip for the
        integral case), while the bool call above is the only one that needed
        moving.
        """
        result = _run({}, code=_OP_DTYPE_PROBE)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        routes = _routes_by_dtype(result.stderr)
        outcomes = _outcomes_by_dtype(result.stderr)
        for dtype in ("torch.float32", "torch.int64"):
            assert routes[dtype]["neg"] == routed_backend("neg"), routes[dtype]
            assert outcomes[dtype] == "ok", outcomes[dtype]


class TestFloat64ResultsAreCorrect:
    """The vendor kernel returns the right float64 answer, not just an answer."""

    @pytest.mark.ascend
    def test_float64_arithmetic_matches_cpu(self):
        code = (
            "import torch, torch_fl; "
            "a = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64).to('flagos'); "
            "b = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float64).to('flagos'); "
            "print((a + b).cpu().tolist()); "
            "print(a.reciprocal().cpu().tolist()); "
            "print(float(a.sum().cpu()))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # FlagTree's Ascend wheel can print this CANN probe warning while
        # torch_fl imports. It is unrelated to the arithmetic output below.
        lines = [
            ln.strip()
            for ln in result.stdout.strip().splitlines()
            if ln.strip()
            != "[WARNING] triton.backends.ascend.utils not found. CANN version check skipped."
        ]
        assert lines == [
            "[1.5, 2.25, 4.125]",
            "[1.0, 0.5, 0.25]",
            "7.0",
        ], result.stdout

    @pytest.mark.ascend
    def test_float64_factories_preserve_dtype_and_value(self):
        code = (
            "import torch, torch_fl; "
            "a = torch.ones(4, device='flagos', dtype=torch.float64); "
            "b = torch.zeros_like(a); "
            "print(a.dtype, b.dtype, float(a.sum().cpu()), float(b.sum().cpu()))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert re.search(r"torch\.float64 torch\.float64 4\.0 0\.0", result.stdout), (
            result.stdout
        )

    @pytest.mark.ascend
    def test_mixed_precision_promotes_and_still_falls_back(self):
        """Promotion happens first; the promoted dtype decides the route.

        ``add(float32, float64)`` must promote to float64 and therefore leave
        the FlagGems route even though neither operand is a float64 *tensor*
        tensor on its own -- at least one is, which is what the predicate
        checks.
        """
        code = (
            "import torch, torch_fl; "
            "a = torch.tensor([1.0, 2.0], dtype=torch.float32).to('flagos'); "
            "b = torch.tensor([0.5, 0.25], dtype=torch.float64).to('flagos'); "
            "out = torch.add(a, b); print(out.dtype, out.cpu().tolist())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={
                **os.environ,
                "FLAGOS_LOG": "dispatch",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "torch.float64 [1.5, 2.25]" in result.stdout, result.stdout
        assert "[flagos dispatch] add.Tensor -> ascend" in result.stderr, result.stderr


class TestGcuInt64Fallback:
    """int64 leaves the FlagGems route on GCU; the other dtypes keep it.

    The gap is a version skew and not a property of the hardware: flag_gems
    5.3.2's gcu300 pointwise codegen passes ``enable_i64=true`` to a
    ``--convert-gpu-to-gcu`` whose installed binary does not declare the option,
    so an i64 operand fails before a kernel exists. Nothing about the routing
    table can express that, which is why it is the runtime predicate again.

    Unlike the Ascend float64 case, the reach here is bounded: the fallback is
    into the vendor slot, and on GCU only the ops the conf leaves on FlagGems
    *and* the codegen registered a kernel for have one. ``clamp_min`` pins that
    boundary, and the fact that its call still fails is the deliberate,
    documented limit of this change rather than a test that was left out.
    """

    # Ops the GCU conf routes to FlagGems and whose vendor registration the
    # escape can substitute: exactly the conf's `# gcu` marker set. All seven
    # move; they do not all succeed, which is the point of the outcome test
    # below. `remainder.Tensor` is the case that argues for the change on
    # correctness rather than availability, since its FlagGems route does not
    # fail at all -- it returns an int32 tensor for an int64 input.
    _MOVED_OPS = (
        "clamp",
        "fmod.Tensor",
        "gelu",
        "mean",
        "mean.dim",
        "remainder.Tensor",
        "silu",
    )

    # What those seven do on int64 once they are on the vendor route. The first
    # three answer; the rest reach the generated template's dtype gate, which
    # declines int64 and hands the call to the host round-trip, where ATen's own
    # checks raise the same error a CPU caller gets.
    _RAISING_OPS = ("gelu", "mean", "mean.dim", "silu")

    @pytest.fixture(autouse=True)
    def _gcu_only(self):
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR != "gcu":
            pytest.skip("GCU build required")

    @pytest.fixture(scope="class")
    def gcu_probe(self) -> subprocess.CompletedProcess:
        result = _run({}, code=_GCU_PROBE)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        return result

    @pytest.fixture(scope="class")
    def gcu_routes(self, gcu_probe) -> dict[str, dict[str, str]]:
        by_dtype = _routes_by_dtype(gcu_probe.stderr)
        assert "torch.int64" in by_dtype and "torch.float32" in by_dtype, (
            gcu_probe.stderr
        )
        return by_dtype

    @pytest.mark.gcu
    def test_int64_flagems_routes_land_on_the_vendor_kernel(self, gcu_routes):
        """int64 answers with the native backend, in a process that also ran fp32.

        Both dtypes go through one probe run, so the different backends reported
        for the same op also show the decision is taken per call from the
        arguments and not cached from the first one.
        """
        i64 = gcu_routes["torch.int64"]
        for op in self._MOVED_OPS:
            assert i64[op] == "gcu", f"{op} ran on {i64[op]}"

    @pytest.mark.gcu
    def test_float32_keeps_the_configured_route(self, gcu_routes):
        """The escape is dtype-keyed and does not leak into float32.

        Asserted against the conf's own value rather than a literal
        ``flagos_python``, so the case stays meaningful if the conf is
        regenerated: what must hold is that float32 keeps whatever route the
        conf chose while int64 does not.
        """
        fp32 = gcu_routes["torch.float32"]
        for op in self._MOVED_OPS:
            assert fp32[op] == routed_backend(op), (
                f"{op} on float32 ran on {fp32[op]}, conf says {routed_backend(op)}"
            )

    @pytest.mark.gcu
    def test_an_op_without_a_vendor_kernel_keeps_the_flagems_route(self, gcu_routes):
        """The escape's boundary: no vendor slot, no substitution.

        ``clamp_min`` is on the FlagGems route in the GCU conf and has no
        ``gcu_register.inc`` entry, so ``ResolveFn`` finds the predicate true and
        then nothing to fall back to, and returns the FlagGems call unchanged.
        That is the intended behaviour -- the alternative would be a routing
        error where the call would have failed on its own -- and it is also why
        ``clamp_min``/``clamp_max``/``clamp_``/``rsub.Scalar`` and the other
        int64-skew routes are not fixed by this change.
        """
        assert gcu_routes["torch.int64"]["clamp_min"] == routed_backend("clamp_min"), (
            gcu_routes["torch.int64"]
        )

    @pytest.mark.gcu
    def test_the_four_the_vendor_kernel_cannot_serve_raise_the_reference_error(
        self, gcu_probe
    ):
        """Moving a route is not the same as fixing it, and this pins which is which.

        ``gelu``, ``mean``, ``mean.dim`` and ``silu`` still fail on int64 -- the
        reference implementation rejects those dtypes itself. What changes is
        *where* they fail: on the FlagGems route they aborted inside the
        compiler, with ``no such option enable_i64`` or ``Pipeline run failed``,
        and now they raise the first line of the error the CPU raises for the
        same call. Both sides are failures, so only the message distinguishes
        them, and the CPU is where the expected message is read from rather than
        written out here.
        """
        outcomes = _gcu_outcomes(gcu_probe.stderr)["torch.int64"]
        host = torch.tensor([0, 1, 3, 8], dtype=torch.int64)
        calls = {
            "gelu": lambda t: torch.ops.aten.gelu(t),
            "mean": lambda t: t.mean(),
            "mean.dim": lambda t: t.mean(0),
            "silu": lambda t: torch.ops.aten.silu(t),
        }
        assert sorted(calls) == sorted(self._RAISING_OPS), "controls drifted apart"
        for op, call in calls.items():
            with pytest.raises(Exception) as excinfo:  # noqa: B017 - the type varies
                call(host)
            expected = (
                f"{type(excinfo.value).__name__}: {str(excinfo.value).splitlines()[0]}"
            )
            assert outcomes[op] == expected, f"{op}: {outcomes[op]} != {expected}"

    @pytest.mark.gcu
    def test_int64_results_match_cpu_values_and_dtype(self):
        """The vendor answer is right, and keeps the dtype it was asked for.

        The dtype comparison is not decoration: before this change the FlagGems
        route returned a *plausible* int32 tensor for int64 ``remainder``, so a
        values-only assertion would have passed on the broken route. Negative
        operands are included because ``fmod`` and ``remainder`` disagree about
        the sign of a negative dividend and a kernel that got that wrong would
        still return the right shape and dtype.
        """
        values = [0, 1, 3, 8, -7]
        divisor = [3, 3, 3, 3, 3]
        expected = {
            "clamp": [0, 1, 3, 4, 0],
            "fmod": [0, 1, 0, 2, -1],
            "remainder": [0, 1, 0, 2, 2],
        }
        a = torch.tensor(values, dtype=torch.int64, device=DEVICE)
        b = torch.tensor(divisor, dtype=torch.int64, device=DEVICE)
        got = {
            "clamp": a.clamp(0, 4),
            "fmod": a.fmod(b),
            "remainder": a.remainder(b),
        }
        for name, reference in expected.items():
            assert got[name].dtype == torch.int64, f"{name} returned {got[name].dtype}"
            want = torch.tensor(reference, dtype=torch.int64)
            assert torch.equal(got[name].cpu(), want), (
                f"{name}: {got[name].cpu().tolist()} != {reference}"
            )
