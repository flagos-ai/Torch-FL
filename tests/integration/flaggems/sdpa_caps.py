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

"""Measured data for the FlagGems ``scaled_dot_product_attention`` route.

``scaled_dot_product_attention`` is the one conf entry the dispatcher does not
select for itself. ``csrc/aten/sdp_choice_stub.cc`` registers a single
PrivateUse1 kernel for the composite under every conf, and that kernel reads
``GetBackendForOp`` to decide between the FlagGems callable and the boxed
composite -- so neither the conf text nor the dispatch log says which arm a call
took, because both arms are served by the same key. What decides it is one piece
of C++ shared by every platform whose conf routes the operator to a FlagGems
backend: the envelope in ``FlagGemsEligible()`` and the conversion in
``RouteMask()``.

That sharing is why this table exists. The clause admits or refuses a call, and
the cases below are the calls it was written around; the *expectation* for a
case therefore belongs to the clause and is stated once, while what was measured
per chip -- the numeric bounds, and how the probe has to be started on this
vendor -- is the only thing that differs between platforms. A chip that starts
routing SDPA to FlagGems adds a row here; it does not add a test file, and it
does not get a smaller case set than the chips that came before it.

Nothing in this module imports torch or torch_fl: the flaggems conftest loads it
before the device assets are preloaded, so it is stdlib-only and the tables are
plain data.
"""

from __future__ import annotations

from typing import NamedTuple

#: The device string ``[flagos dispatch]`` prints for the flagos device.
DEVICE = "flagos:0"

#: Source the probe interpreter defines before the cases run. The operands are
#: built on the host and moved, deliberately: ``torch.randn(..., device=DEVICE)``
#: is itself dispatched, privately, to ``flag_gems.randn``, so drawing them on
#: the host keeps every case's device work to the one call this suite is about
#: and keeps the values independent of the generator the runtime picks. The mask
#: builders are the model's own spellings: ``key_row`` is the ``(1,1,1,KV)``
#: all-true row Qwen-Image-2.1 passes on every prefill call.
PROBE_HELPERS = (
    "def bf16(*shapes):\n"
    "    return tuple(torch.randn(*s).to(DEVICE).to(torch.bfloat16)\n"
    "                 for s in shapes)\n"
    "def fp16(*shapes):\n"
    "    return tuple(torch.randn(*s).to(DEVICE).to(torch.float16)\n"
    "                 for s in shapes)\n"
    "def fp32(*shapes):\n"
    "    return tuple(torch.randn(*s).to(DEVICE) for s in shapes)\n"
    "def square():\n"
    "    return bf16((1, 2, 1024, 128), (1, 2, 1024, 128), (1, 2, 1024, 128))\n"
    "def nonsquare():\n"
    "    return bf16((1, 2, 1024, 128), (1, 2, 1040, 128), (1, 2, 1040, 128))\n"
    "def key_row(n, drop=0, n_batch=1):\n"
    "    row = torch.ones(n_batch, 1, 1, n, dtype=torch.bool)\n"
    "    if drop:\n"
    "        row[..., n - drop:] = False\n"
    "    return row.to(DEVICE)\n"
    "def key_row_with_one_drop(n):\n"
    "    row = torch.ones(1, 1, 1, n, dtype=torch.bool)\n"
    "    row[0, 0, 0, 7] = False\n"
    "    return row.to(DEVICE)\n"
)


class Case(NamedTuple):
    """One call the envelope is written around.

    ``build`` is python source returning the ``(q, k, v)`` triple on the device
    and ``kwargs`` is source for the remaining keyword arguments, both spliced
    into the probe. ``routed`` is how many calls may reach the FlagGems entry
    point for that case -- a count, not a flag, so an ineligible case that
    reaches the kernel fails at that case instead of moving a global total.
    ``mask_form`` is ``"none"`` when every routed call must have been made
    without a mask, ``"converted"`` when every routed call must have carried
    ``RouteMask``'s fp32 additive, and ``None`` when the case must not reach the
    kernel at all and there is therefore no call to describe.
    """

    tag: str
    build: str
    kwargs: str
    routed: int
    mask_form: str | None
    clause: str


#: Every case, in the order the probe runs them. The list is the union of what
#: each chip measured, so a case one platform needed is a case every platform
#: routing SDPA to FlagGems now runs; nothing was dropped in the merge, and the
#: comment on each entry names the clause it pins.
SDPA_CASES: tuple[Case, ...] = (
    Case(
        "eligible",
        "square",
        "{}",
        1,
        "none",
        "bf16, 4-D, head_dim 128, query seq 1024, no mask",
    ),
    Case(
        "joint",
        "nonsquare",
        "{}",
        1,
        "none",
        "the model's shape: query 1024 against key/value 1040",
    ),
    Case(
        "mask_bcast_true",
        "nonsquare",
        '{"attn_mask": key_row(1040)}',
        1,
        "converted",
        "the model's mask: an all-true (1,1,1,KV) bool row",
    ),
    Case(
        "mask_false",
        "nonsquare",
        '{"attn_mask": key_row_with_one_drop(1040)}',
        1,
        "converted",
        "one false in the row, which the additive must carry through",
    ),
    Case(
        "mask_bcast_true_b2",
        "lambda: bf16((2, 2, 1024, 128), (2, 2, 1024, 128), (2, 2, 1024, 128))",
        '{"attn_mask": key_row(1024, drop=128, n_batch=2)}',
        1,
        "converted",
        "the same key-valid row at batch 2, dropping keys rather than none",
    ),
    Case(
        "mask_full_true",
        "nonsquare",
        '{"attn_mask": torch.ones(1, 2, 1024, 1040, dtype=torch.bool).to(DEVICE)}',
        0,
        None,
        "a materialised (B,H,Q,KV) bool mask, whose numel is not KV",
    ),
    Case(
        "mask_4d_bool",
        "square",
        '{"attn_mask": torch.ones(1, 2, 1024, 1024, dtype=torch.bool).to(DEVICE)}',
        0,
        None,
        "a bool mask with one entry per (query, key) pair",
    ),
    Case(
        "mask_float_zeros",
        "nonsquare",
        '{"attn_mask": torch.zeros(1, 1, 1, 1040).to(DEVICE)}',
        0,
        None,
        "a float mask, which the clause admits as bool only",
    ),
    Case(
        "mask_additive",
        "square",
        '{"attn_mask": torch.zeros(1024, 1024, dtype=torch.bfloat16).to(DEVICE)}',
        0,
        None,
        "an additive mask at rank 2, which the route does not convert",
    ),
    Case(
        "mask_row_f32",
        "square",
        '{"attn_mask": torch.zeros(1, 1, 1, 1024).to(DEVICE)}',
        0,
        None,
        "a per-key row in float32, whose polarity is undefined",
    ),
    Case(
        "head_dim64",
        "lambda: bf16((1, 2, 1024, 64), (1, 2, 1040, 64), (1, 2, 1040, 64))",
        "{}",
        1,
        "none",
        "head_dim 64, inside the route's head_dim bounds",
    ),
    Case(
        "head_dim8",
        "lambda: bf16((1, 2, 1024, 8), (1, 2, 1024, 8), (1, 2, 1024, 8))",
        "{}",
        0,
        None,
        "head_dim 8, under the 16 a tile's BLOCK_N bottoms out at",
    ),
    Case(
        "head_dim256",
        "lambda: bf16((1, 2, 1024, 256), (1, 2, 1024, 256), (1, 2, 1024, 256))",
        "{}",
        0,
        None,
        "head_dim 256, over the 128 whose tile still fits",
    ),
    Case(
        "seq512",
        "lambda: bf16((1, 2, 512, 128), (1, 2, 512, 128), (1, 2, 512, 128))",
        "{}",
        1,
        "none",
        "query seq 512, which the shared clause no longer floors",
    ),
    Case(
        "masked_seq512",
        "lambda: bf16((2, 2, 512, 128), (2, 2, 512, 128), (2, 2, 512, 128))",
        '{"attn_mask": key_row(512, drop=128, n_batch=2)}',
        0,
        None,
        "that key row at seq 512, under the masked route floor",
    ),
    Case(
        "float16",
        "lambda: fp16((1, 2, 1024, 128), (1, 2, 1040, 128), (1, 2, 1040, 128))",
        "{}",
        1,
        "none",
        "fp16, the second dtype the route was measured on",
    ),
    Case(
        "float32",
        "lambda: fp32((1, 2, 1024, 128), (1, 2, 1040, 128), (1, 2, 1040, 128))",
        "{}",
        0,
        None,
        "float32, whose tile the kernel reports OutOfResources on",
    ),
    Case(
        "rank2",
        "lambda: bf16((1024, 128), (1040, 128), (1040, 128))",
        "{}",
        0,
        None,
        "2-D input, and the route requires the 4-D call form",
    ),
    Case(
        "causal",
        "nonsquare",
        '{"is_causal": True}',
        0,
        None,
        "is_causal, which the flag_gems entry point does not take",
    ),
    Case(
        "scale",
        "nonsquare",
        '{"scale": 0.1}',
        0,
        None,
        "an explicit scale, which the entry point does not take",
    ),
    Case(
        "dropout",
        "nonsquare",
        '{"dropout_p": 0.1}',
        0,
        None,
        "a nonzero dropout_p, which the entry point does not take",
    ),
    Case(
        "gqa",
        "nonsquare",
        '{"enable_gqa": True}',
        0,
        None,
        "enable_gqa, which the entry point does not take",
    ),
)


class RouteRow(NamedTuple):
    """What was measured on one chip, plus how its probe has to be started.

    ``imports`` is the order the probe interpreter imports torch in. DCU needs
    ``torch_fl`` first: with ``torch`` first the PrivateUse1 key is registered
    after the device module has been built and every copy to ``flagos:0`` raises
    "Cannot initialize CUDA without ATen_cuda library" instead of moving the
    tensor. MetaX has no such requirement, so it keeps the ordinary order.

    ``bounds_revalidated`` says whether the numeric bounds below were measured on
    this chip or inherited from the chip that did measure them. It is
    ``False`` for a platform the shared clause reaches but whose numbers were
    not re-taken; the gap is recorded here and in
    ``docs/reference/operator-support.md`` rather than being papered over.
    """

    imports: tuple[str, ...]
    triton_backend: str
    diff_bound: float
    identity_bound: float
    informative_bound: float
    bounds_revalidated: bool
    measured_on: str


#: Platform -> what the probe measured there. A platform absent from this table
#: does not route ``scaled_dot_product_attention`` to a FlagGems backend, and the
#: suite skips with the conf's own route as the reason.
SDPA_ROUTE_ROWS: dict[str, RouteRow] = {
    "dcu": RouteRow(
        imports=("torch_fl", "torch"),
        triton_backend="hcu",
        diff_bound=5e-2,
        identity_bound=1e-3,
        informative_bound=1e-2,
        bounds_revalidated=True,
        measured_on=(
            "Hygon DCU bw1000, DTK 6.3.26113, torch 2.10.0+cpu decoupled, "
            "flag_gems 5.4.0rc2.post1+g437ba3938"
        ),
    ),
    "metax": RouteRow(
        imports=("torch", "torch_fl"),
        triton_backend="metax",
        # The bounds below were taken on the DCU above; MetaX has not been
        # re-measured against the widened envelope. The clause that produces
        # them is shared, and the two facts they state -- that an all-true
        # key-valid row is the identity and that half a dropped row is not --
        # are properties of the flag_gems entry point both chips call.
        diff_bound=5e-2,
        identity_bound=1e-3,
        informative_bound=1e-2,
        bounds_revalidated=False,
        measured_on="MetaX C550, bounds inherited from the DCU measurement above",
    ),
}

#: The maskless call, the all-true-row call and a half-dropped-row call, on one
#: set of operands. Built once and reused, so both numbers are statements about
#: the mask and not about the RNG. The all-true row converts to an additive that
#: is zero everywhere, so it must leave the answer where the maskless call left
#: it; the third row must move it, or the additive is being built and not read.
#:
#: The two calls do not run the same kernel -- one is handed a KV-wide fp32
#: additive and the other ``attn_mask=None``, and the flag_gems entry point takes
#: a different branch on that -- so the first comparison is the bf16 rounding of
#: the output and not bit equality. Measured on a DCU bw1000 at
#: q(1,2,1024,128) x kv(1,2,1040,128) bf16: max|d| 0.0001220703125, which is
#: 2^-13, one bf16 ulp below the 0.287 peak, over 8 elements of 2.1M, with both
#: arms landing on the same 0.000694 error against an fp64 reference.
SDPA_IDENTITY = (
    "torch.manual_seed(4242)\n"
    "q = torch.randn(1, 2, 1024, 128).to(DEVICE).to(torch.bfloat16)\n"
    "k = torch.randn(1, 2, 1040, 128).to(DEVICE).to(torch.bfloat16)\n"
    "v = torch.randn(1, 2, 1040, 128).to(DEVICE).to(torch.bfloat16)\n"
    "base = torch.nn.functional.scaled_dot_product_attention(q, k, v)\n"
    "bcast = torch.nn.functional.scaled_dot_product_attention(\n"
    "    q, k, v, attn_mask=key_row(1040))\n"
    "half = torch.ones(1, 1, 1, 1040, dtype=torch.bool)\n"
    "half[0, 0, 0, :520] = False\n"
    "dropped = torch.nn.functional.scaled_dot_product_attention(\n"
    "    q, k, v, attn_mask=half.to(DEVICE))\n"
    "print('IDENTICAL max|d|='\n"
    "      f'{(bcast.float() - base.float()).abs().max().item():.9f}')\n"
    "print('INFORMATIVE max|d|='\n"
    "      f'{(dropped.float() - base.float()).abs().max().item():.6f}')\n"
    "print(f'ROUTED-MASKS {_sdpa_calls}')\n"
)

#: Calls the identity block above makes to the FlagGems entry point: the
#: maskless base, the all-true row and the half-dropped row.
SDPA_IDENTITY_CALLS = 3

#: Of those, the ones that carry a mask and therefore show up as converted in
#: the census: the all-true row and the half-dropped row.
SDPA_IDENTITY_CONVERTED = 2

#: Cases whose answer cannot be compared against a host reference, because both
#: arms draw from their own RNG. What such a case pins is a routing clause.
SDPA_NO_REFERENCE = frozenset({"dropout"})


def expected_routed_calls() -> int:
    """Calls to the FlagGems entry point a full run must make."""
    return sum(case.routed for case in SDPA_CASES) + SDPA_IDENTITY_CALLS


def expected_converted_masks() -> int:
    """Routed calls in a full run that must carry ``RouteMask``'s additive."""
    return (
        sum(1 for case in SDPA_CASES if case.mask_form == "converted")
        + SDPA_IDENTITY_CONVERTED
    )


def probe_source(imports: tuple[str, ...]) -> str:
    """The interpreter the cases run in, as one self-contained program.

    A fresh interpreter is what makes the counter the callable the route calls:
    ``PythonOpCache::GetFunc`` resolves ``flag_gems.scaled_dot_product_attention``
    with a package import plus a ``getattr``, memoized per qualname for the life
    of the process, so a counter installed on the package before the first call
    is the callable every later call reaches.
    """
    # `torch_fl` before `torch` where the row asks for it: the PrivateUse1 key
    # has to be registered before torch's device module is built.
    header = "".join(f"import {name}\n" for name in imports)
    oracle = (
        "import flag_gems\n"
        "_sdpa_calls = []\n"
        "_sdpa_real = flag_gems.scaled_dot_product_attention\n"
        "def _mask_repr(m):\n"
        "    if m is None:\n"
        "        return None\n"
        "    return (tuple(m.shape), str(m.dtype).replace('torch.', ''),\n"
        "            tuple(m.stride()))\n"
        "def _sdpa_counting(*args, **kwargs):\n"
        "    _sdpa_calls.append((tuple(tuple(t.shape) for t in args),\n"
        "                        _mask_repr(kwargs.get('attn_mask'))))\n"
        "    return _sdpa_real(*args, **kwargs)\n"
        "flag_gems.scaled_dot_product_attention = _sdpa_counting\n"
        "\n"
        # The host reference moves the arguments back rather than rebuilding them,
        # so both arms of every case see the same values. A bool mask keeps its
        # dtype on the way over -- it is the meaning of the mask, and a float mask
        # would take a different branch of the composite.
        "def host_ref(q, k, v, kw):\n"
        "    moved = {}\n"
        "    for n, t in kw.items():\n"
        "        if torch.is_tensor(t):\n"
        "            moved[n] = t.cpu() if t.dtype == torch.bool else t.cpu().float()\n"
        "        else:\n"
        "            moved[n] = t\n"
        "    return torch.nn.functional.scaled_dot_product_attention(\n"
        "        q.float().cpu(), k.float().cpu(), v.float().cpu(), **moved)\n"
        "\n"
        "def run_case(tag, build, kw, cmp=True):\n"
        "    q, k, v = build()\n"
        "    ref = host_ref(q, k, v, kw) if cmp else None\n"
        "    before = len(_sdpa_calls)\n"
        "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw)\n"
        "    made = _sdpa_calls[before:]\n"
        "    routed = len(made)\n"
        "    masks = [m for _, m in made]\n"
        "    diff = 'skip' if ref is None else \\\n"
        "        f'{(out.float().cpu() - ref).abs().max().item():.6f}'\n"
        "    print(f'RESULT {tag} routed={routed} maxdiff={diff} '\n"
        "          f'device={out.device} masks={masks!r}')\n"
    )
    body = "".join(
        f'run_case("{case.tag}", {case.build}, {case.kwargs}'
        + (", cmp=False),\n" if case.tag in SDPA_NO_REFERENCE else "),\n")
        for case in SDPA_CASES
    )
    return (
        f"{header}"
        "import torch\n"
        f"DEVICE = {DEVICE!r}\n"
        "torch.manual_seed(1729)\n"
        f"{PROBE_HELPERS}"
        f"{oracle}"
        f"{body}"
        f"{SDPA_IDENTITY}"
    )
