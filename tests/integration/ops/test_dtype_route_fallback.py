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

"""Dtype escape from the FlagGems route on Ascend.

A vendor conf is a per-op routing table, so it cannot express "FlagGems, except
for float64". On Ascend that exception is real and broad: BiShengHIR rejects
the float64 instantiation of nearly every kernel FlagGems' pointwise codegen
produces -- measured on Ascend910 / CANN 9.0.0 / FlagTree 0.6.2a1+ascend3.5,
add/sub/div/neg/abs/exp/log/sqrt/reciprocal/where/clamp/fill_/zeros_like/
ones_like/ones/full/arange over float64 all raise MLIRCompilationError, while
mul, cat and the comparisons compile -- so the conf has no way to state it. The
exception lives at runtime instead: ``FlagGemsRejectsDtype`` in
``csrc/aten/common.cc``, consulted by ``Dispatcher::ResolveFn``.

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

from backend_conf import routed_backend


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


def _run(extra_env: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update({"FLAGOS_LOG": "dispatch", **extra_env})
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
    )


def _routes_by_dtype(stderr: str) -> dict[str, dict[str, str]]:
    """``{dtype: {op: backend}}`` from the probe's section markers."""
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in stderr.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = {}
        elif line.startswith("[flagos dispatch] ") and current is not None:
            op, _, backend = line[len("[flagos dispatch] ") :].partition(" -> ")
            sections[current][op] = backend
    return sections


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
        lines = [ln.strip() for ln in result.stdout.strip().splitlines()]
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
