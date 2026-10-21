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

"""The execution half of the contract: a FlagGems route computes the CPU answer.

Routing presence is not operator support. For each operator the active conf
routes to a FlagGems backend, this builds the inputs on the host, moves them to
the device and compares the device result against the answer the same inputs get
on the CPU.

Building on the host and moving is the point, not an implementation detail:
building once per arm with ``torch.randn(..., device=...)`` compares two random
draws and reports the difference as a numerics failure, which is how a set of
correct operators was once recorded as wrong.

Every case still runs on every configured platform. A case the conf routes away
from FlagGems is asserted on that route, so per-platform coverage can differ
without silently shrinking the execution set.
"""

import pytest
from device_init import torch

from flaggems_caps import (
    DEVICE,
    accel_label,
    flaggems_runtime_unavailable,
    routes_to_flaggems,
)

# Case tag -> (how the host inputs are built, the call under test). The tag is
# the conf's operator name, so the same string gates the case and names it in a
# failure. Every call is one the FlagGems kernel serves through the generated
# dispatcher; nothing here reaches into a module path, which is what the
# dispatcher resolves.
_EXECUTION_CASES = {
    "neg": (lambda: (torch.randn(64, 64),), lambda x: torch.neg(x)),
    "abs": (lambda: (torch.randn(64, 64),), lambda x: torch.abs(x)),
    "exp": (lambda: (torch.randn(64, 64),), lambda x: torch.exp(x)),
    # A strictly positive operand: `rsqrt` is undefined at and around zero, and a
    # value the CPU kernel reports as `inf` is not a numerics disagreement.
    "rsqrt": (lambda: (torch.rand(64, 64) + 0.5,), lambda x: torch.rsqrt(x)),
    "tanh": (lambda: (torch.randn(64, 64),), lambda x: torch.tanh(x)),
    "sigmoid": (lambda: (torch.randn(64, 64),), lambda x: torch.sigmoid(x)),
    "silu": (lambda: (torch.randn(64, 64),), lambda x: torch.nn.functional.silu(x)),
    "gelu": (lambda: (torch.randn(64, 64),), lambda x: torch.nn.functional.gelu(x)),
    "amax": (lambda: (torch.randn(64, 64),), lambda x: torch.amax(x, 1)),
    "amin": (lambda: (torch.randn(64, 64),), lambda x: torch.amin(x, 1)),
    "sum.dim_IntList": (
        lambda: (torch.randn(64, 64),),
        lambda x: torch.sum(x, 1),
    ),
    "sub.Tensor": (
        lambda: (torch.randn(64, 64), torch.randn(64, 64)),
        lambda x, y: x - y,
    ),
    "mul.Tensor": (
        lambda: (torch.randn(64, 64), torch.randn(64, 64)),
        lambda x, y: x * y,
    ),
    "where.self": (
        lambda: (torch.randn(64, 64), torch.randn(64, 64)),
        lambda x, y: torch.where(x > 0, x, y),
    ),
    "embedding": (
        lambda: (torch.randint(0, 3, (4, 8)), torch.randn(3, 16)),
        lambda idx, w: torch.nn.functional.embedding(idx, w),
    ),
}

# Cases compared at 1e-3 rather than 1e-4: a reduction over a whole row does not
# accumulate in the same order as its CPU counterpart, so the two answers agree
# to roughly single precision rather than to the last few bits.
_LOOSE_TOLERANCE = {"sum.dim_IntList"}


def requires_grad_for_backward(tensor):
    """Move an input and request gradients only where autograd permits them."""
    return tensor.requires_grad_(tensor.is_floating_point())


@pytest.mark.parametrize("tag", sorted(_EXECUTION_CASES))
@pytest.mark.flaggems_python
def test_operator_result_matches_the_cpu_reference(tag: str):
    if not accel_label():
        pytest.fail("the active conf is not a named backends_<platform>.conf")
    reason = flaggems_runtime_unavailable()
    if reason:
        pytest.fail(reason)

    build, run = _EXECUTION_CASES[tag]
    torch.manual_seed(0)
    host = build()
    # Autograd records the route at allocation time. Enable it for the move so a
    # result check also covers the backward route the generated kernel owns.
    on_device = [t.to(DEVICE) for t in host]
    on_device = [requires_grad_for_backward(t) for t in on_device]
    expected = run(*host)
    actual = run(*on_device)
    assert actual.device.type == "flagos", f"{tag} returned a {actual.device} tensor"
    tol = 1e-3 if tag in _LOOSE_TOLERANCE else 1e-4
    torch.testing.assert_close(
        actual.cpu(), expected, rtol=tol, atol=tol, msg=f"{tag} on {DEVICE}"
    )


_FLOAT64_MM_SHAPES = (
    (31, 32, 32),
    (32, 32, 32),
    (33, 32, 32),
    (33, 33, 33),
    (64, 64, 64),
    (32, 64, 64),
    (64, 64, 32),
    (512, 512, 512),
)


def test_float64_mm_matches_cpu_across_tile_boundaries():
    """A dtype-dependent shared-memory fit must hold on every mm platform.

    MUSA's declared matrix-multiply tiles asked more static shared memory for
    float64 than the device provides. Shapes on both sides of the 32-element
    tile boundary catch a space that works only because a smaller shape narrows
    before compilation. The CPU comparison also rejects a fit that chooses a
    configuration that runs but computes the wrong answer.
    """
    assert routes_to_flaggems("mm"), "mm is not configured for FlagGems"
    torch.manual_seed(0)
    for m, k, n in _FLOAT64_MM_SHAPES:
        a = torch.randn(m, k, dtype=torch.float64)
        b = torch.randn(k, n, dtype=torch.float64)
        expected = torch.mm(a, b)
        actual = torch.mm(a.to(DEVICE), b.to(DEVICE))
        torch.flagos.synchronize()
        torch.testing.assert_close(
            actual.cpu(),
            expected,
            rtol=1e-12,
            atol=1e-12,
            msg=f"float64 mm at {m}x{k}x{n}",
        )
