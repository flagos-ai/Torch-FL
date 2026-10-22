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

An operator the conf does not route to FlagGems here is a skip, with the conf's
own route as the reason -- the same statement the routing file makes, so the two
halves of the suite agree about what this chip claims.
"""

import pytest
import torch
import torch_fl  # noqa: F401  -- claiming the flagos device is the point

from flaggems_caps import (
    DEVICE,
    accel_label,
    conf_route,
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


@pytest.mark.anyplatform
@pytest.mark.parametrize("tag", sorted(_EXECUTION_CASES))
def test_operator_result_matches_the_cpu_reference(tag: str):
    accel = accel_label() or "this conf"
    if not routes_to_flaggems(tag):
        pytest.skip(
            f"{accel} routes {tag} to {conf_route(tag) or 'unlisted'}, not to a "
            "FlagGems backend"
        )
    reason = flaggems_runtime_unavailable()
    if reason:
        pytest.skip(reason)

    build, run = _EXECUTION_CASES[tag]
    torch.manual_seed(0)
    host = build()
    on_device = [t.to(DEVICE) for t in host]
    expected = run(*host)
    actual = run(*on_device)
    assert actual.device.type == "flagos", f"{tag} returned a {actual.device} tensor"
    tol = 1e-3 if tag in _LOOSE_TOLERANCE else 1e-4
    torch.testing.assert_close(
        actual.cpu(), expected, rtol=tol, atol=tol, msg=f"{tag} on {DEVICE}"
    )
