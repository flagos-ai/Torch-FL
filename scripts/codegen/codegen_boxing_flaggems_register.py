#!/usr/bin/env python3
"""Generate the per-platform FlagGems registration lists for the CUDA-ABI platforms.

``cuda``, ``metax``, ``dcu``, ``ppu`` and ``tsingmicro`` all share the ``#else``
arm of ``csrc/aten/register.cc``. With ``FLAGOS_BUILD_BOXING=ON`` that arm
includes the CUDA codegen's ``generated/register.inc`` and claims every generated
op on PrivateUse1 -- correct only because the wheel then carries the generated
CUDA kernels (``generated/cuda_kernels.cc``). Boxing is now off by default for
all five (``cmake/flagos_platforms.json``), so those wheels link neither the CUDA
kernels nor a vendor torch: serializing a ``cuda``-routed op there reaches an
empty PrivateUse1 slot and raises "backend not registered" instead of falling
back.

This generator emits two files per platform, because the two builds claim
different sets:

``csrc/aten/generated/register_flaggems_<platform>.inc`` -- the m.impl() lines
for exactly the ops that platform's ``backends_<platform>.conf`` routes to a
FlagGems backend. register.cc includes the matching file when
``FLAGOS_FLAGGEMS_PYTHON`` is on and boxing is off, so those ops land on their
kFlagGems dispatcher slot and every other op stays unregistered and reaches
cpu_fallback. The net effect is FlagGems-primary routing: accelerated ops on the
FlagGems Python path, the rest on cpu_fallback.

``csrc/aten/generated/register_boxing_<platform>.inc`` -- the same list minus
the ``none`` routes, for a boxing build. That build can serve an op from the
boxing kernels, but a default conf never routes ``boxing`` (AGENTS.md, "Day-0
boxing confs"), so an op the conf routes to ``none`` must stay unregistered and
reach cpu_fallback; claiming it raises "routed to 'none' ... but the op is
registered on PrivateUse1". Measured on DCU: a boxing build claimed all 2037
ops of ``register.inc`` and the first ``clone()`` of a flagos tensor failed on
``empty_like``, which ``backends_hygon.conf`` routes to ``none`` for a routing
reason (it is a view/factory contract, not a missing kernel).

An op is emitted when all three hold:

  * the conf routes it to something other than ``cuda``/``none``/``""`` -- i.e.
    a FlagGems backend names it (the ``# tileops`` trailing annotation on a
    ``flaggems`` line is a comment, so it does not change the value);
  * the CUDA codegen's ``generated/register.inc`` has a ``Wrapper<Name>`` for it
    (that wrapper block is always compiled into every CUDA-ABI build, so the
    symbol exists even without the boxing kernels); and
  * ``generated/flaggems_python_kernels.cc`` registers a kFlagGems dispatcher
    for it -- without one, ``m.impl()`` here would claim an op that the runtime
    then cannot serve, which is the "backend not registered" raise this file
    exists to avoid. The metax/dcu ``scaled_dot_product_attention`` and the
    tsingmicro ``embedding_dense_backward`` / ``index.Tensor`` routes are the
    ops this drops; sdp is claimed separately by ``csrc/aten/sdp_choice_stub.cc``.

The wrapper symbol is read back from ``generated/register.inc`` rather than
derived from the op name (``add.Tensor`` is served by ``WrapperAddTensor`` while
``_adaptive_avg_pool2d`` is by ``WrapperPrivAdaptiveAvgPool2d``), which is why
that file and not the conf is the source of truth for the symbol. Unlike the
vendor FlagGems generators this one does not import ``gen_vendor_confs`` or
``backend_coverage``: it needs only committed text, so ``--check`` runs in the
lean codegen-checks CI job with nothing but a stock CPU torch.

Usage:
    python3 scripts/codegen/codegen_boxing_flaggems_register.py            # rewrite
    python3 scripts/codegen/codegen_boxing_flaggems_register.py --check    # exit 1 if stale
"""

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
GENERATED = REPO / "csrc/aten/generated"
REGISTER_INC = GENERATED / "register.inc"
OPS_CC = GENERATED / "ops.cc"
FLAGGEMS_KERNELS = GENERATED / "flaggems_python_kernels.cc"
BOXING_KERNELS = GENERATED / "cuda_kernels.cc"
CONFIGS = REPO / "torch_fl/configs"

# The CUDA-ABI platforms that share the `#else` arm of register.cc. Order fixes
# the --check report order; each writes the file named after it.
PLATFORMS = ("nvidia", "metax", "hygon", "thead", "tsingmicro")

# Ops whose PrivateUse1 claim register.cc makes itself rather than through the
# generated list, with the reason each one is not a claim this file can emit.
#
# ``mm`` / ``bmm`` on dcu: the soft-lowp matrix prelude is claimed in
# register.cc beside the generated include, and its conf routes them to FlagGems
# (gen_vendor_confs.py, SOFT_LOWP_FLAGGEMS_MEASURED). Emitting the claim here as
# well is a duplicate m.impl in one TORCH_LIBRARY_IMPL body -- a compile error --
# and leaving the op unmentioned reads as "routed but unregistered", which is
# the list below. `_scaled_dot_product_attention` on metax/dcu is the same shape
# with a whole-op override (csrc/aten/sdp_choice_stub.cc).
HAND_REGISTERED_CLAIMS = {
    "hygon": {"mm", "bmm"},
}

# Ops register.cc claims on every FlagGems-primary build that compiles the
# soft-lowp prelude (its `mm`/`bmm` block), whatever their conf route. The
# op's own claim then follows whatever the conf says about its own key:
#
#   * a key the conf routes to a backend -- claim the generated wrapper;
#   * a key the conf routes `none` -- claim SoftLowpRoutedByPair instead, which
#     hands an ordinary call to the mm/bmm key the conf does route, so
#     registration and that route agree. Claiming the plain wrapper instead
#     made every `mm.out` raise "routed to 'none' ... but the op is registered"
#     on DCU (run 40615250313, test_mm_out): the wrapper calls the
#     whole-op dispatcher, which answers by the key the call is made under.
SOFT_LOWP_PAIR_CLAIMS = {
    "hygon": {
        "mm.dtype": "mm",
        "mm.dtype_out": "mm",
        "mm.out": "mm",
        "bmm.dtype": "bmm",
        "bmm.dtype_out": "bmm",
        "bmm.out": "bmm",
    },
}

# The wrapper a soft-lowp pair claim emits for a key the conf leaves `none`,
# one entry per route key the pair is followed to.
CLAIM_MACROS = {
    "mm": "FLAGOS_SOFT_LOWP_CLAIM_AS_MM",
    "bmm": "FLAGOS_SOFT_LOWP_CLAIM_AS_BMM",
}

# Conf values that are not a FlagGems backend: `cuda` is the boxing kernel,
# `none` is cpu_fallback, and the empty value is a malformed line.
# (`cuda` is the pre-rename spelling of the NVIDIA boxing key; a conf written
# before that rename still routes to it, so both spellings are listed.)
NON_FLAGGEMS_ROUTES = ("nvidia", "cuda", "boxing", "none", "")

# Per-platform values that name the platform's own vendor backend rather than a
# FlagGems one. TsingMicro's conf routes its metadata-only view ops to
# `tsingmicro` (csrc/aten/strided_ops.cc registers the at::native:: stride
# helpers under Backend::kTsingMicro); those resolve through the dispatcher's
# vendor slot, not kFlagGems, so they must not be treated as FlagGems routes
# here. A FlagGems route means "claim this op on PrivateUse1 so its kFlagGems
# slot is reachable"; a vendor route needs no such claim, and emitting one for
# it would hand the op a kFlagGems dispatcher the runtime cannot fill.
#
# THIS IS NOT A NO-OP. Do not delete it as an unused guard. Removing the
# `tsingmicro` entry makes every `= tsingmicro` op count as a FlagGems route;
# six of the fourteen view ops happen to have a kFlagGems dispatcher in
# generated/flaggems_python_kernels.cc (alias, _conj, _unsafe_view,
# slice.Tensor, transpose.int, unsqueeze), so the generator emits an m.impl()
# for them and register_flaggems_tsingmicro.inc grows from 51 to 57 entries --
# claims on exactly the kFlagGems slot that has no kernel behind it, which is
# the defect this file's exclusion exists to prevent. (The file is byte-identical
# across the routing commits only because the entry is present; compare
# with/without the entry to see the delta, not one commit against the next.)
VENDOR_ROUTES = {"tsingmicro": ("tsingmicro",)}

HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_boxing_flaggems_register.py -- DO NOT EDIT.
//
// m.impl() lines for the ops {platform} routes to a FlagGems backend.
// Included by register.cc inside TORCH_LIBRARY_IMPL(aten, PrivateUse1) when
// FLAGOS_FLAGGEMS_PYTHON is set and FLAGOS_BUILD_BOXING is off: the default
// FlagGems-primary build, which links no CUDA boxing kernels and no vendor
// torch. An op routed to a FlagGems backend that is not claimed here would
// reach an empty kFlagGems dispatcher slot and raise "backend not registered"
// instead of falling back to cpu_fallback, which is what this list prevents.
//
// Derived from torch_fl/configs/backends_{platform}.conf minus the ops with no
// kFlagGems dispatcher in csrc/aten/generated/flaggems_python_kernels.cc. This
// file is regenerated by its generator and gated in CI; do not edit it.
//
// Total: {count} ops registered here.{extra}
"""

BOXING_HEADER = """\
// Copyright (c) 2026, BAAI. All Rights Reserved.
//
// @generated by scripts/codegen/codegen_boxing_flaggems_register.py -- DO NOT EDIT.
//
// m.impl() lines for the ops {platform} claims on PrivateUse1 in a boxing build.
// Included by register.cc inside TORCH_LIBRARY_IMPL(aten, PrivateUse1) when
// FLAGOS_BUILD_BOXING is on: the wheel carries the generated CUDA boxing
// kernels, so a claimed op can be served. Ops the conf routes to `none` are
// left out -- AGENTS.md states a default conf never routes `boxing`, and a
// claim without a route raises "routed to 'none' ... but the op is registered
// on PrivateUse1" instead of reaching cpu_fallback.
//
// Total: {count} ops registered here.
"""

EXCLUDED_HEADER = """\
//
// Routed to a FlagGems backend in the conf but with no kFlagGems dispatcher to
// serve them, so left unregistered (they reach cpu_fallback):
"""


def wrapper_map():
    """op name -> Wrapper<Name>, read back from the CUDA codegen's register.inc."""
    text = REGISTER_INC.read_text()
    return dict(re.findall(r'^\s*m\.impl\("([^"]+)",\s*(\w+)\);', text, flags=re.M))


def flaggems_kernel_ops():
    """Op names that have a kFlagGems dispatcher registered in the CUDA codegen.

    ``flaggems_python_kernels.cc`` registers its kernels by dispatcher name
    (``REGISTER_IMPL_TO_DISPATCHER(..., <Dispatcher>, ...)``); ``ops.cc`` maps
    that name to the op string via ``ADD_IMPL_TO_DISPATCHER(..., <Dispatcher>,
    "<op>")``. Joining the two gives the set of ops whose PrivateUse1
    ``kFlagGems`` slot the runtime can actually fill.
    """
    ops_cc = OPS_CC.read_text()
    disp_to_op = dict(
        re.findall(r"ADD_IMPL_TO_DISPATCHER\(\w+,\s*(\w+),\s*\"([^\"]+)\"\)", ops_cc)
    )
    kernels = FLAGGEMS_KERNELS.read_text()
    dispatchers = set(
        re.findall(r"REGISTER_IMPL_TO_DISPATCHER\(\w+,\s*(\w+),", kernels)
    )
    return {disp_to_op[d] for d in dispatchers if d in disp_to_op}


def boxing_kernel_ops():
    """Op names whose ``kBoxing`` dispatcher the generated boxing kernels fill.

    Same join as ``flaggems_kernel_ops``, against ``cuda_kernels.cc`` -- the
    file the CUDA codegen scores boxing coverage into. The per-vendor narrowing
    happens at build time inside those kernels, so a listed op can still leave
    its C++ slot empty for a given vendor; the four bench-measured cases that do
    (cuda/dcu/ppu ``mm``/``bmm``, metax ``add.Tensor``) are registered by hand
    in ``register.cc``.
    """
    ops_cc = OPS_CC.read_text()
    disp_to_op = dict(
        re.findall(r"ADD_IMPL_TO_DISPATCHER\(\w+,\s*(\w+),\s*\"([^\"]+)\"\)", ops_cc)
    )
    kernels = BOXING_KERNELS.read_text()
    dispatchers = set(
        re.findall(
            r"REGISTER_IMPL_TO_DISPATCHER\(\w+,\s*(\w+),\s*Backend::kBoxing,", kernels
        )
    )
    return {disp_to_op[d] for d in dispatchers if d in disp_to_op}


def conf_routes(platform):
    """op name -> route value for one platform's conf, comments stripped."""
    routes = {}
    path = CONFIGS / f"backends_{platform}.conf"
    for line in path.read_text().splitlines():
        name, sep, rest = line.split("#", 1)[0].partition("=")
        if sep:
            routes[name.strip()] = rest.strip()
    return routes


def boxing_target_ops(platform, wrappers, kernel_ops):
    """(registered ops, ops left out, and why).

    The claim is what a boxing build can actually serve: the ops the conf routes
    to ``boxing`` that the generated boxing kernels have a dispatcher for, plus
    the ops it routes to any other backend, which are served through the vendor
    slot or a FlagGems arm and need nothing from this list to be registered.
    ``kernel_ops`` is the boxing kernel set (see ``flaggems_kernel_ops`` for the
    ``flaggems_python_kernels.cc`` side of the same join).

    Two kinds of op are left out, and both reach cpu_fallback:

    * a ``none`` route -- the conf states there is nothing to serve it;
    * a ``boxing`` route with no boxing kernel, which is a conf that describes
      a build other than this one (metax's ``backends_metax_boxing.conf`` names
      ``boxing`` for ``normal_`` while the wheel ships no boxing kernels:
      claiming it there raised from the dispatcher, run 40129896922);

    as is an op the conf routes elsewhere that ``register.inc`` has no wrapper
    for (metax/dcu's ``scaled_dot_product_attention``): claiming it would need a
    wrapper symbol that does not exist. Each is named in the file's excluded
    list, so the difference is on the record rather than implicit.
    """
    routes = conf_routes(platform)
    unserved = {op for op, value in routes.items() if value == "none"}
    unkerneled = {
        op for op, value in routes.items() if value == "boxing" and op not in kernel_ops
    }
    by_hand = HAND_REGISTERED_CLAIMS.get(platform, set())
    claimed = sorted(
        op
        for op, value in routes.items()
        if value != "none"
        and op not in by_hand
        and op in wrappers
        and (value != "boxing" or op in kernel_ops)
    )
    unwrapped = (
        {op for op, value in routes.items() if value != "none"} - set(wrappers)
    ) - by_hand
    return claimed, sorted(unserved | unkerneled | unwrapped)


def render_boxing(platform, wrappers, kernel_ops):
    ops, unserved = boxing_target_ops(platform, wrappers, kernel_ops)
    lines = [
        BOXING_HEADER.format(platform=platform, count=len(ops)),
        "//\n// Held back from the boxing claim: the conf routes them to `none`, or",
        "\n// `register.inc` has no wrapper for them:\n",
    ]
    lines.extend(f"//   - {op}\n" for op in unserved)
    lines.append("\n")
    lines.extend(f'  m.impl("{op}", {wrappers[op]});\n' for op in ops)
    return "".join(lines)


def target_ops(platform, wrappers, kernel_ops):
    """(sorted registered ops, sorted FlagGems-routed ops with no kernel)."""
    routes = conf_routes(platform)
    non_flaggems = NON_FLAGGEMS_ROUTES + VENDOR_ROUTES.get(platform, ())
    flaggems_routed = {op for op, value in routes.items() if value not in non_flaggems}
    # Claimed by register.cc, not here: neither registered nor excluded (see
    # HAND_REGISTERED_CLAIMS).
    flaggems_routed -= HAND_REGISTERED_CLAIMS.get(platform, set())
    registered = sorted(flaggems_routed & set(wrappers) & kernel_ops)
    unserved = sorted(flaggems_routed & kernel_ops - set(wrappers))
    no_kernel = sorted(flaggems_routed - kernel_ops)
    if unserved:
        raise SystemExit(
            f"backends_{platform}.conf routes these to a FlagGems backend but "
            f"{REGISTER_INC.relative_to(REPO)} has no wrapper: " + ", ".join(unserved)
        )
    # A matrix form the conf leaves `none` is still claimed, by the wrapper
    # that follows the pair's own route (see SOFT_LOWP_PAIR_CLAIMS). The op
    # comes off the excluded list and the route key is out of `none`.
    paired = SOFT_LOWP_PAIR_CLAIMS.get(platform, {})
    no_kernel = [op for op in no_kernel if op not in paired]
    return registered, no_kernel


def soft_lowp_pair_claimed(platform, wrappers):
    """The matrix forms this list claims for a conf that routes them `none`.

    Only a form the conf leaves `none` is claimed here; one the conf routes to
    a FlagGems backend is already in the plain list and needs nothing.
    """
    routes = conf_routes(platform)
    paired = SOFT_LOWP_PAIR_CLAIMS.get(platform, {})
    out = {}
    for op, key in paired.items():
        if routes.get(op) != "none":
            continue
        if op not in wrappers:
            raise SystemExit(
                f"backends_{platform}.conf: {op} is a soft-lowp pair claim but "
                f"{REGISTER_INC.relative_to(REPO)} has no wrapper for it"
            )
        out[op] = key
    return out


def render(platform, wrappers, kernel_ops):
    ops, no_kernel = target_ops(platform, wrappers, kernel_ops)
    paired = soft_lowp_pair_claimed(platform, wrappers)
    lines = [
        HEADER.format(
            platform=platform,
            count=len(ops) + len(paired),
            extra="",
        )
    ]
    if no_kernel:
        lines.append(EXCLUDED_HEADER)
        lines.extend(f"//   - {op}\n" for op in no_kernel)
    lines.append("\n")
    lines.extend(f'  m.impl("{op}", {wrappers[op]});\n' for op in ops)
    if paired:
        lines.append(
            "\n// Matrix forms the conf leaves `none`: claimed by the pair's\n"
            "// route, not by the slot this key names (see register.cc's\n"
            "// mm/bmm block).\n"
        )
        lines.extend(
            f'  {CLAIM_MACROS[key]}("{op}", {wrappers[op]});\n'
            for op, key in sorted(paired.items())
        )
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any committed .inc is stale instead of rewriting it",
    )
    args = ap.parse_args()

    if not REGISTER_INC.exists():
        raise SystemExit(
            f"{REGISTER_INC.relative_to(REPO)} is missing; run "
            "scripts/codegen/codegen_ops.py first -- the wrapper symbols are read from it."
        )
    wrappers = wrapper_map()
    kernel_ops = flaggems_kernel_ops()
    boxing_kernels = boxing_kernel_ops()

    stale = []
    for platform in PLATFORMS:
        text = render(platform, wrappers, kernel_ops)
        out = GENERATED / f"register_flaggems_{platform}.inc"
        boxing_text = render_boxing(platform, wrappers, boxing_kernels)
        boxing_out = GENERATED / f"register_boxing_{platform}.inc"

        if args.check:
            for path, expected in ((out, text), (boxing_out, boxing_text)):
                current = path.read_text() if path.exists() else ""
                if current != expected:
                    stale.append(path.relative_to(REPO))
            continue

        out.write_text(text)
        registered, no_kernel = target_ops(platform, wrappers, kernel_ops)
        print(
            f"[gen] {out.relative_to(REPO)}  ({len(registered)} m.impl lines, "
            f"{len(no_kernel)} FlagGems routes left unregistered)"
        )
        boxing_out.write_text(boxing_text)
        claimed, unserved = boxing_target_ops(platform, wrappers, boxing_kernels)
        print(
            f"[gen] {boxing_out.relative_to(REPO)}  ({len(claimed)} m.impl lines, "
            f"{len(unserved)} unrouted/unserved ops left unregistered)"
        )

    if args.check:
        if stale:
            raise SystemExit(
                "stale generated file(s): "
                + ", ".join(str(path) for path in stale)
                + f"; run: python3 {Path(__file__).name}"
            )
        print(
            f"[check] {len(PLATFORMS)} register_flaggems_*.inc and "
            "register_boxing_*.inc files are up to date"
        )


if __name__ == "__main__":
    main()
