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
        cases.append(
            (f"argmin-d{dim}", lambda dev, dim=dim: torch.argmin(_seeded(dev), dim))
        )
    cases.append(("argmax-flat", lambda dev: torch.argmax(_seeded(dev))))
    cases.append(("argmin-flat", lambda dev: torch.argmin(_seeded(dev))))
    # The (values, indices) pair, whose two halves are compared separately by the
    # test below -- a pair whose halves disagree is worse than a wrong value.
    for name in ("max", "min"):
        for dim in (1, 2):
            cases.append(
                (
                    f"{name}dim-d{dim}",
                    lambda dev, name=name, dim=dim: getattr(torch, name)(
                        _positive(dev, (2, 5, 4)), dim
                    ),
                )
            )

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

    # --- the rest of the routed list ---------------------------------------
    # These ops are routed but had no case here, so the one layer that compares
    # every routed spelling against the reference did not reach them. A dedicated
    # file usually covers what its author thought of; this covers the same op
    # against the same reference on the same operand, which is what caught the
    # three silent defects in this backend so far.
    unary = (
        "neg",
        "abs",
        "sign",
        "signbit",
        "reciprocal",
        "rsqrt",
        "trunc",
        "ceil",
        "floor",
        "round",
        "frac",
        "logical_not",
        "isnan",
        "isinf",
        "isposinf",
        "isneginf",
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
        "exp2",
        "log2",
        "log10",
        "expm1",
        "log1p",
        "sgn",
    )
    for name in unary:
        fn = getattr(torch, name, None)
        if fn is None:
            continue
        # `_positive` keeps log/sqrt/rsqrt in their domain; the ops that need
        # negatives (sign, signbit, abs, neg, trig) are unaffected by the shift.
        cases.append((f"{name}", lambda dev, fn=fn: fn(_positive(dev, (3, 4)))))
        cases.append(
            (
                f"{name}-neg",
                lambda dev, fn=fn: fn(_positive(dev, (3, 4), offset=-1.6)),
            )
        )

    # Scalar spellings: aten routes these separately from the tensor ones.
    for name, value in (
        ("add", 2.5),
        ("sub", 0.5),
        ("mul", 1.5),
        ("div", 2.0),
        ("rsub", 1.5),
        ("pow", 2.0),
        ("remainder", 1.5),
        ("fmod", 1.5),
        ("eq", 1.0),
        ("ne", 1.0),
        ("lt", 1.0),
        ("le", 1.0),
        ("gt", 1.0),
        ("ge", 1.0),
    ):
        fn = getattr(torch, name.replace("rsub", "rsub"), None)
        if fn is None:
            continue
        if name == "rsub":
            cases.append(("rsub.Scalar", lambda dev: torch.rsub(_seeded(dev), 1.5)))
            continue
        cases.append(
            (
                f"{name}.Scalar",
                lambda dev, fn=fn, value=value: fn(_seeded(dev), value),
            )
        )

    # Shape-only ops: the answer is the same tensor, but the routing and the
    # stride bookkeeping are theirs.
    for name, fn in (
        ("t", lambda t: t.t()),
        ("transpose", lambda t: t.transpose(0, 1)),
        ("squeeze", lambda t: t.unsqueeze(0).squeeze()),
        ("squeeze.dim", lambda t: t.unsqueeze(0).squeeze(0)),
        ("unsqueeze", lambda t: t.unsqueeze(1)),
    ):
        cases.append((f"{name}-shape", lambda dev, fn=fn: fn(_seeded(dev))))

    # Indexing and gathering.
    cases.append(("select", lambda dev: _seeded(dev).select(1, 2)))
    cases.append(("slice", lambda dev: _seeded(dev, (6, 8))[1:5:2, ::3]))
    cases.append(
        (
            "take",
            lambda dev: torch.take(
                _seeded(dev), torch.tensor([0, 3, 7, 11], device=dev)
            ),
        )
    )
    cases.append(
        (
            "gather",
            lambda dev: torch.gather(
                _seeded(dev, (4, 6)),
                1,
                torch.tensor([[0, 1, 2], [3, 4, 5], [0, 0, 0], [5, 4, 3]], device=dev),
            ),
        )
    )
    cases.append(
        (
            "index_select",
            lambda dev: torch.index_select(
                _seeded(dev, (6, 4)), 0, torch.tensor([0, 2, 5], device=dev)
            ),
        )
    )
    cases.append(
        (
            "index.Tensor",
            lambda dev: _seeded(dev, (5, 4))[torch.tensor([1, 3], device=dev)],
        )
    )
    cases.append(
        (
            "masked_fill.Tensor",
            lambda dev: _seeded(dev).masked_fill(
                _seeded(dev) > 2.5, torch.tensor(9.0, device=dev)
            ),
        )
    )

    # Concat and the normalisations.
    cases.append(
        ("cat", lambda dev: torch.cat([_seeded(dev), _seeded(dev, offset=3.0)], 1))
    )
    cases.append(
        (
            "cat-dim0",
            lambda dev: torch.cat([_seeded(dev, (2, 3)), _seeded(dev, (4, 3))], 0),
        )
    )
    cases.append(
        (
            "embedding",
            lambda dev: torch.nn.functional.embedding(
                torch.tensor([[0, 2], [1, 3]], device=dev),
                _seeded(dev, (5, 7)),
            ),
        )
    )
    cases.append(
        (
            "native_layer_norm",
            lambda dev: torch.native_layer_norm(
                _seeded(dev, (3, 4, 8)), (8,), None, None, 1e-5
            ),
        )
    )
    # `native_batch_norm` without running statistics in evaluation mode is not a
    # call the reference accepts: on this PyTorch build it segfaults inside the CPU
    # kernel itself (`F.batch_norm` validates first and raises). The buffers are
    # supplied so the case exercises the served inference node rather than an
    # illegal call.
    cases.append(
        (
            "native_batch_norm",
            lambda dev: torch.native_batch_norm(
                _seeded(dev, (2, 3, 4, 4)),
                None,
                None,
                torch.zeros(3, device=dev),
                torch.ones(3, device=dev),
                False,
                0.1,
                1e-5,
            ),
        )
    )
    cases.append(
        (
            "conv2d",
            lambda dev: torch.nn.functional.conv2d(
                _seeded(dev, (1, 3, 8, 8)),
                _seeded(dev, (4, 3, 3, 3), offset=-0.1),
                padding=1,
            ),
        )
    )

    # The factories and the two remaining reductions the earlier blocks miss.
    cases.append(("linspace", lambda dev: torch.linspace(0.0, 1.0, 9, device=dev)))
    cases.append(
        (
            "linspace-tensors",
            lambda dev: torch.linspace(
                torch.tensor(0.0, device=dev), torch.tensor(1.0, device=dev), 7
            ),
        )
    )
    cases.append(
        (
            "arange",
            lambda dev: torch.arange(0, 12, dtype=torch.float32, device=dev),
        )
    )
    cases.append(
        (
            "arange-step",
            lambda dev: torch.arange(1.0, 5.0, 0.5, dtype=torch.float32, device=dev),
        )
    )
    cases.append(
        (
            "solve_triangular",
            lambda dev: torch.linalg.solve_triangular(
                torch.triu(_seeded(dev, (5, 5))) + torch.eye(5, device=dev) * 2.0,
                _seeded(dev, (5, 3)),
                upper=True,
            ),
        )
    )

    # The activations with their own selectors, their backward, and the two
    # convolution spellings -- the last of these is what makes a conv model
    # trainable, so a silent difference there is worth catching.
    for name, extra in (("elu", -1.0), ("leaky_relu", 0.1), ("softplus", None)):
        fn = getattr(torch.nn.functional, name)
        # `elu`/`leaky_relu` take a slope, `softplus` does not.
        call = (
            (lambda t, fn=fn: fn(t))
            if extra is None
            else (lambda t, fn=fn, extra=extra: fn(t, extra))
        )
        cases.append((f"{name}-act", lambda dev, c=call: c(_positive(dev, (3, 4)))))
        cases.append(
            (
                f"{name}-act-neg",
                lambda dev, c=call: c(_positive(dev, (3, 4), offset=-1.6)),
            )
        )

    cases.append(
        (
            "silu_backward",
            lambda dev: torch.ops.aten.silu_backward(
                _seeded(dev), _positive(dev, offset=-0.5)
            ),
        )
    )

    # `convolution_overrideable` is the spelling aten routes a conv to, and
    # `convolution_backward_overrideable` its gradient.
    cases.append(
        (
            "conv1d",
            lambda dev: torch.nn.functional.conv1d(
                _seeded(dev, (1, 2, 9)),
                _seeded(dev, (3, 2, 3), offset=-0.1),
                padding=1,
            ),
        )
    )
    for rank in (1, 2):
        cases.append(
            (
                f"conv{rank + 1}d-stride",
                lambda dev, rank=rank: (
                    torch.nn.functional.conv2d
                    if rank == 2
                    else torch.nn.functional.conv1d
                )(
                    _seeded(dev, (1, 2, 8, 8) if rank == 2 else (1, 2, 8)),
                    _seeded(dev, (3, 2, 3, 3) if rank == 2 else (3, 2, 3), offset=-0.1),
                    stride=2,
                    padding=1,
                ),
            )
        )
    cases.append(
        (
            "conv2d-groups",
            lambda dev: torch.nn.functional.conv2d(
                _seeded(dev, (1, 4, 6, 6)),
                _seeded(dev, (2, 2, 3, 3), offset=-0.1),
                padding=1,
                groups=2,
            ),
        )
    )

    # The triangles, with the diagonal that shifts them -- the parameter a plan
    # key once failed to include, which is why more than one diagonal is here.
    for name in ("triu", "tril"):
        fn = getattr(torch, name)
        for diagonal in (-2, -1, 0, 1, 2):
            cases.append(
                (
                    f"{name}-d{diagonal}",
                    lambda dev, fn=fn, diagonal=diagonal: fn(
                        _seeded(dev, (5, 6)), diagonal
                    ),
                )
            )
        cases.append(
            (
                f"{name}-rank3",
                lambda dev, fn=fn: fn(_seeded(dev, (2, 4, 4))),
            )
        )

    # The backward of the convolution, which is what makes a conv model trainable.
    cases.append(
        (
            "conv_backward",
            lambda dev: torch.ops.aten.convolution_backward(
                _seeded(dev, (1, 2, 4, 4)),
                _seeded(dev, (1, 3, 6, 6)),
                _seeded(dev, (2, 3, 3, 3), offset=-0.1),
                [2],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
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
    # Spellings whose conf key is not the name aten dispatches under.
    "select": "select.int",
    "slice": "slice.Tensor",
    "transpose": "transpose.int",
    "solve_triangular": "linalg_solve_triangular",
    "conv1d": "convolution_overrideable",
    "conv2d": "convolution_overrideable",
    "conv3d": "convolution_overrideable",
    "conv_backward": "convolution_backward_overrideable",
    "sum": "sum.dim_IntList",
    "prod": "prod.dim_int",
    "mean": "mean.dim",
    "amax": "amax",
    "amin": "amin",
    "softmax": "_softmax",
    "log_softmax": "_log_softmax",
    "argmax": "argmax",
    "argmin": "argmin",
    "maxdim": "max.dim",
    "mindim": "min.dim",
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

    A label that resolves to no conf key at all is neither of those: it is a case
    that would never run, and an op that is served under a different spelling would
    look covered while nothing compared it. `routed_backend_or_none` raises for an
    absent key precisely to keep the two apart, so that raise propagates rather than
    becoming a third kind of skip. This caught eight cases in the change that
    introduced them: `select`, `slice`, `transpose`, `solve_triangular`, the three
    convolution spellings and the convolution backward were all routed, and all
    skipped, because their conf keys are `select.int`, `slice.Tensor`,
    `transpose.int`, `linalg_solve_triangular` and `convolution_overrideable`.
    """
    name = label[2:] if label.startswith("u-") else label.split("-")[0]
    key = ROUTED_AS.get(name, name)
    try:
        return routed_backend_or_none(key) is not None
    except AssertionError as error:
        raise AssertionError(
            f"sweep case {label!r} resolves to conf key {key!r}, which the conf "
            f"does not list. Either the label is wrong or ROUTED_AS needs an "
            f"entry for it -- a skip here would hide a case that never runs. "
            f"({error})"
        ) from error


@pytest.mark.parametrize("label,builder", CASES, ids=[c[0] for c in CASES])
@pytest.mark.anyplatform
def test_a_routed_op_matches_the_cpu_reference(label, builder):
    """Every routed spelling answers what the reference answers."""
    if not _routed(label):
        pytest.skip(f"{label} is not routed on this build")

    reference = builder(torch.device("cpu"))
    got = builder(DEVICE)

    # A several-result op -- `max.dim`'s (values, indices) pair, `native_layer_norm`'s
    # (out, mean, rstd) -- is compared result by result. Everything else is a single
    # tensor.
    if isinstance(reference, tuple):
        assert isinstance(got, tuple), (label, type(got))
        assert len(got) == len(reference), (label, len(got), len(reference))
        for index, (got_part, want) in enumerate(zip(got, reference)):
            if want is None:
                assert got_part is None, (label, index, got_part)
                continue
            _compare_tensor(f"{label}[{index}]", got_part, want)
        return

    _compare_tensor(label, got, reference)


def _compare_tensor(label, got, reference):
    """One tensor, with the tolerance its dtype calls for."""
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
