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

"""Apple Silicon (Metal / MPSGraph) differential sweep over the served set.

Every other test file in this directory checks one family against its own
reference. This one is different in kind: it runs **every routed operator over a
small grid of spellings** and compares each against the CPU reference, which is
how the two defects below were found. Both were live on `main` when this file was
written.

* **`clamp` aborted on an integral operand** and dropped a NaN (fixed in #734).
* **`maximum`/`minimum` replaced a broadcast NaN with a finite value** (fixed in
  #740).

Neither was caught by the per-family files, because each of those tests the
combination its author had in mind. `maximum`'s NaN test used matching shapes and
its broadcast test used finite values, so the intersection -- a NaN that has to be
widened -- went untested. A sweep does not have that blind spot.

## Why a table rather than prose

The cases below are a data table so that adding a spelling is one line. The
coverage goal for this backend has been met increment by increment, and each
increment adds a body; a table that grows with the conf is the only way the sweep
stays honest about what is actually routed.

## What it does not do

It compares **values**, not performance, and it uses a tolerance
(`rtol=2e-2`) loose enough for a graph that computes in float32 where the CPU may
use a different summation order. It also does not enumerate shapes -- one shape
per case -- so a shape-dependent defect needs the per-family files, which do go
wider on that axis.

Every case self-skips off an MPS build. A case whose operator is not routed here
is skipped rather than asserted on, because the sweep's job is to catch a routed
operator answering wrongly, not to police the conf.
"""

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

# Ops whose domain starts above zero, so a plain `rand` would feed them a value
# outside it and the reference would answer nan on both sides -- a green case that
# tests nothing.
DOMAIN_OFFSET = {
    "sqrt": 1.0,
    "rsqrt": 1.0,
    "reciprocal": 1.0,
    "log": 1.0,
    "log2": 1.0,
    "log10": 1.0,
    "acos": -1.0,
    "acosh": 2.0,
    "atanh": -1.0,
}

REDUCE_OPS = ("sum", "prod", "mean", "amax", "amin")

BINARY_OPS = ("add", "sub", "mul", "div", "pow", "fmod", "remainder", "floor_divide")

COMPARE_OPS = ("eq", "ne", "lt", "le", "gt", "ge")

UNARY_OPS = (
    "abs",
    "neg",
    "sign",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "exp",
    "log",
    "exp2",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "ceil",
    "floor",
    "trunc",
    "round",
    "sigmoid",
    "logical_not",
    "relu",
    "silu",
    "gelu",
    "leaky_relu",
    "elu",
    "softplus",
    "hardsigmoid",
    "hardswish",
    "mish",
    "selu",
    "logsigmoid",
)


def _positive(device, shape=(3, 4), offset=0.0):
    """Values in (0.1, 1.6) shifted by `offset`, so the two sides get the same
    numbers without relying on a shared RNG stream across devices."""
    return (
        torch.rand(shape, device=device, generator=torch.Generator().manual_seed(7))
        * 1.5
        + 0.1
        + offset
    )


def _seeded(device, shape=(3, 4), offset=2.0):
    return (
        torch.rand(shape, device=device, generator=torch.Generator().manual_seed(13))
        + offset
    )


def _with_nan(device, shape=(2, 5, 4), positions=((0, 0, 0),)):
    """`_positive` with NaNs written at fixed positions.

    Every case here is about what an operator does *when a NaN is present*, and
    the family of defects that motivated this helper -- `maximum` in #740 and
    `argmax` in #750 -- each shipped because the family was tested with finite
    values and the NaN behaviour was tested elsewhere, but never both at once.
    """
    t = _positive(device, shape).clone()
    for index in positions:
        t[index] = float("nan")
    return t


def _cases():
    """`(label, builder)` where `builder(device)` returns the op's result."""
    cases = []

    for op in REDUCE_OPS:
        for keepdim in (False, True):
            # `prod` over a dim takes a single int where the others take a list,
            # so the reduced operand is flattened for it.
            if op == "prod":
                cases.append(
                    (
                        f"{op}-all-kd{keepdim}",
                        lambda dev, keepdim=keepdim: torch.prod(
                            _positive(dev, (40,)), 0, keepdim=keepdim
                        ),
                    )
                )
            else:
                cases.append(
                    (
                        f"{op}-all-kd{keepdim}",
                        lambda dev, op=op, keepdim=keepdim: getattr(torch, op)(
                            _positive(dev, (2, 5, 4)), (0, 1, 2), keepdim=keepdim
                        ),
                    )
                )
                cases.append(
                    (
                        f"{op}-dim-kd{keepdim}",
                        lambda dev, op=op, keepdim=keepdim: getattr(torch, op)(
                            _positive(dev, (2, 5, 4)), 1, keepdim=keepdim
                        ),
                    )
                )

    for dim in (0, 1, 2, -1):
        cases.append(
            (
                f"softmax-d{dim}",
                lambda dev, dim=dim: torch.softmax(_positive(dev, (2, 5, 4)), dim),
            )
        )
        cases.append(
            (
                f"log_softmax-d{dim}",
                lambda dev, dim=dim: torch.log_softmax(_positive(dev, (2, 5, 4)), dim),
            )
        )

    # `argmax` takes a real operand; `all`/`any` take the predicate.
    for dim in (0, 1):
        cases.append(
            (f"argmax-d{dim}", lambda dev, dim=dim: torch.argmax(_seeded(dev), dim))
        )
    cases.append(("argmax-flat", lambda dev: torch.argmax(_seeded(dev))))

    # --- NaN in a reduction ------------------------------------------------
    # Not a shape variant of the cases above: a NaN propagates differently from
    # any finite value, and two of this backend's silent defects lived exactly
    # here. `sum`/`prod`/`mean`/`amax`/`amin` all answer NaN; `argmax` names the
    # position of the first NaN; `max`/`min` return it as the value.
    nan_positions = (
        ("head", ((0, 0, 0),)),
        ("middle", ((0, 2, 1),)),
        ("tail", ((1, 4, 3),)),
        ("two", ((0, 0, 0), (1, 4, 3))),
    )
    for where, positions in nan_positions:
        for keepdim in (False, True):
            cases.append(
                (
                    f"amax-nan-{where}-kd{keepdim}",
                    lambda dev, positions=positions, keepdim=keepdim: torch.amax(
                        _with_nan(dev, positions=positions), (0, 1, 2), keepdim=keepdim
                    ),
                )
            )
            cases.append(
                (
                    f"mean-nan-{where}-kd{keepdim}",
                    lambda dev, positions=positions, keepdim=keepdim: torch.mean(
                        _with_nan(dev, positions=positions), 1, keepdim=keepdim
                    ),
                )
            )
            cases.append(
                (
                    f"sum-nan-{where}-kd{keepdim}",
                    lambda dev, positions=positions, keepdim=keepdim: torch.sum(
                        _with_nan(dev, positions=positions), 1, keepdim=keepdim
                    ),
                )
            )
        # `argmax` over the axis each NaN sits on, so the answer is the position
        # of that NaN rather than a value it propagates to.
        cases.append(
            (
                f"argmax-nan-{where}",
                lambda dev, positions=positions: torch.argmax(
                    _with_nan(dev, positions=positions), 2
                ),
            )
        )
        cases.append(
            (
                f"maxdim-nan-{where}",
                lambda dev, positions=positions: (
                    torch.max(_with_nan(dev, positions=positions), 2).values
                ),
            )
        )

    # The same NaN through an activation or a pointwise function, which is where a
    # mask or a padded position usually enters. `gelu` and `silu` live on
    # `torch.nn.functional`, the rest on `torch`.
    nan_unary = (
        ("gelu", torch.nn.functional.gelu),
        ("silu", torch.nn.functional.silu),
        ("sigmoid", torch.sigmoid),
        ("exp", torch.exp),
        ("log", torch.log),
        ("sqrt", torch.sqrt),
        ("tanh", torch.tanh),
    )
    for name, fn in nan_unary:
        cases.append(
            (
                f"{name}-nan",
                lambda dev, fn=fn: fn(_with_nan(dev, positions=((0, 1, 2),))),
            )
        )
    for name, fn in (("all", torch.all), ("any", torch.any)):
        for dim in (0, 1):
            cases.append(
                (
                    f"{name}-d{dim}",
                    lambda dev, fn=fn, dim=dim: fn(_seeded(dev) > 2.5, dim),
                )
            )
        cases.append((f"{name}-flat", lambda dev, fn=fn: fn(_seeded(dev) > 2.5)))

    for lo, hi in ((0.0, 1.0), (-1.0, 1.0), (0.5, 0.5), (-2.0, 3.0)):
        cases.append(
            (
                f"clamp-float-{lo}-{hi}",
                lambda dev, lo=lo, hi=hi: torch.clamp(
                    _seeded(dev, offset=-1.0), lo, hi
                ),
            )
        )
    for lo, hi in ((-3, 3), (0, 5)):
        cases.append(
            (
                f"clamp-int-{lo}-{hi}",
                lambda dev, lo=lo, hi=hi: torch.clamp(
                    torch.arange(-6, 6, device=dev).reshape(3, 4), lo, hi
                ),
            )
        )
    cases.append(
        (
            "clamp-nan",
            lambda dev: torch.clamp(
                torch.tensor([float("nan"), -1.0, 0.5, 2.0], device=dev), 0.0, 1.0
            ),
        )
    )
    cases.append(
        ("clamp_min", lambda dev: torch.clamp_min(_seeded(dev, offset=-1.0), 0.0))
    )
    cases.append(
        ("clamp_max", lambda dev: torch.clamp_max(_seeded(dev, offset=-1.0), 0.0))
    )

    cases.append(
        (
            "nan_to_num",
            lambda dev: torch.nan_to_num(
                torch.tensor(
                    [float("nan"), float("inf"), -float("inf"), 1.0], device=dev
                )
            ),
        )
    )
    cases.append(
        (
            "masked_fill",
            lambda dev: _seeded(dev).masked_fill(_seeded(dev) > 2.5, 9.0),
        )
    )
    cases.append(
        (
            "where",
            lambda dev: torch.where(_seeded(dev) > 2.5, _seeded(dev), _seeded(dev)),
        )
    )

    for op in BINARY_OPS:
        fn = getattr(torch, op)
        cases.append((f"{op}-scalar", lambda dev, fn=fn: fn(_seeded(dev), 2.5)))
        cases.append((f"{op}-same", lambda dev, fn=fn: fn(_seeded(dev), _seeded(dev))))
        cases.append(
            (f"{op}-bcast", lambda dev, fn=fn: fn(_seeded(dev), _seeded(dev, (4,))))
        )

    for op in COMPARE_OPS:
        fn = getattr(torch, op)
        cases.append(
            (f"{op}-scalar", lambda dev, fn=fn: fn(_seeded(dev, offset=-1.0), 0.0))
        )
        cases.append(
            (f"{op}-tensor", lambda dev, fn=fn: fn(_seeded(dev), _seeded(dev)))
        )

    # The two ops whose NaN rule differs from their node's, at matching shapes and
    # through a broadcast -- the latter is the case #740 fixed.
    for op in ("maximum", "minimum"):
        fn = getattr(torch, op)
        cases.append(
            (
                f"{op}-nan-same",
                lambda dev, fn=fn: fn(
                    torch.tensor([float("nan"), 1.0, 2.0], device=dev),
                    torch.tensor([0.0, 2.0, 1.0], device=dev),
                ),
            )
        )
        cases.append(
            (
                f"{op}-nan-bcast",
                lambda dev, fn=fn: fn(
                    torch.tensor([float("nan"), 1.0, 2.0, 3.0], device=dev),
                    torch.arange(12.0, device=dev).reshape(3, 4),
                ),
            )
        )
        cases.append(
            (
                f"{op}-finite-bcast",
                lambda dev, fn=fn: fn(_seeded(dev), _seeded(dev, (4,))),
            )
        )

    for u in UNARY_OPS:
        fn = getattr(torch, u, None)
        if fn is None:
            continue
        offset = DOMAIN_OFFSET.get(u, 0.0)
        cases.append(
            (
                f"u-{u}",
                lambda dev, fn=fn, offset=offset: fn(_positive(dev, (3, 4), offset)),
            )
        )

    for u in ("cumsum", "cumprod"):
        fn = getattr(torch, u)
        for dim in (0, 1):
            cases.append(
                (
                    f"{u}-d{dim}",
                    lambda dev, fn=fn, dim=dim: fn(_positive(dev, (4, 5)), dim),
                )
            )

    cases.append(("mm", lambda dev: _seeded(dev, (8, 16)) @ _seeded(dev, (16, 32))))
    cases.append(
        (
            "bmm",
            lambda dev: torch.bmm(_seeded(dev, (4, 8, 16)), _seeded(dev, (4, 16, 32))),
        )
    )
    cases.append(
        (
            "addmm",
            lambda dev: torch.addmm(
                _seeded(dev, (8, 32)), _seeded(dev, (8, 16)), _seeded(dev, (16, 32))
            ),
        )
    )

    return cases


CASES = _cases()


# The aten key each case's op is routed under in the conf. The API name and the
# router's key are not always the same string -- `add` reaches the router as
# `add.Tensor`, and `where` as `where.self` -- so the mapping is written out
# rather than inferred.
ROUTED_AS = {
    "sum": "sum.dim_IntList",
    "prod": "prod.dim_int",
    "mean": "mean.dim",
    "amax": "amax",
    "amin": "amin",
    "softmax": "_softmax",
    "log_softmax": "_log_softmax",
    "argmax": "argmax",
    "all": "all.dim",
    "any": "any.dim",
    "clamp": "clamp",
    "clamp_min": "clamp_min",
    "clamp_max": "clamp_max",
    "nan_to_num": "nan_to_num",
    "masked_fill": "masked_fill.Scalar",
    "where": "where.self",
    "add": "add.Tensor",
    "sub": "sub.Tensor",
    "mul": "mul.Tensor",
    "div": "div.Tensor",
    "pow": "pow.Tensor_Tensor",
    "fmod": "fmod.Tensor",
    "remainder": "remainder.Tensor",
    "floor_divide": "floor_divide",
    "eq": "eq.Tensor",
    "ne": "ne.Tensor",
    "lt": "lt.Tensor",
    "le": "le.Tensor",
    "gt": "gt.Tensor",
    "ge": "ge.Tensor",
    "maximum": "maximum",
    "minimum": "minimum",
    "cumsum": "cumsum",
    "cumprod": "cumprod",
    "mm": "mm",
    "bmm": "bmm",
    "addmm": "addmm",
}


def _routed(label):
    """Whether the op behind `label` is served here, read from the conf.

    The sweep should not fail because an op is on the fallback -- that is a
    routing decision, tested elsewhere. It fails when a routed op answers wrongly.
    """
    name = label[2:] if label.startswith("u-") else label.split("-")[0]
    key = ROUTED_AS.get(name, name)
    try:
        return routed_backend_or_none(key) is not None
    except AssertionError:
        # A name the conf does not list at all is not routed.
        return False


@pytest.mark.parametrize("label,builder", CASES, ids=[c[0] for c in CASES])
@pytest.mark.anyplatform
def test_a_routed_op_matches_the_cpu_reference(label, builder):
    """Every routed spelling answers what the reference answers."""
    if not _routed(label):
        pytest.skip(f"{label} is not routed on this build")

    reference = builder(torch.device("cpu"))
    got = builder(DEVICE)

    assert got.shape == reference.shape, (label, got.shape, reference.shape)
    assert got.dtype == reference.dtype, (label, got.dtype, reference.dtype)

    got_host = got.cpu()
    # Inexact results are compared with a float32-sized tolerance; integral and
    # boolean results have one right answer and are compared exactly.
    if got_host.dtype == torch.bool or not got_host.is_floating_point():
        assert torch.equal(got_host, reference), (label, got_host, reference)
    else:
        torch.testing.assert_close(
            got_host.float(),
            reference.float(),
            rtol=2e-2,
            atol=2e-2,
            equal_nan=True,
            msg=lambda m: f"{label}: {m}",
        )
