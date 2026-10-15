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

This generator emits ``csrc/aten/generated/register_flaggems_<platform>.inc``:
the m.impl() lines for exactly the ops that platform's ``backends_<platform>.conf``
routes to a FlagGems backend. register.cc includes the matching file when
``FLAGOS_FLAGGEMS_PYTHON`` is on and boxing is off, so those ops land on their
kFlagGems dispatcher slot and every other op stays unregistered and reaches
cpu_fallback. The net effect is FlagGems-primary routing: accelerated ops on the
FlagGems Python path, the rest on cpu_fallback.

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
CONFIGS = REPO / "torch_fl/configs"

# The CUDA-ABI platforms that share the `#else` arm of register.cc. Order fixes
# the --check report order; each writes the file named after it.
PLATFORMS = ("cuda", "metax", "dcu", "ppu", "tsingmicro")

# Conf values that are not a FlagGems backend: `cuda` is the boxing kernel,
# `none` is cpu_fallback, and the empty value is a malformed line.
NON_FLAGGEMS_ROUTES = ("cuda", "none", "")

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


def conf_routes(platform):
    """op name -> route value for one platform's conf, comments stripped."""
    routes = {}
    path = CONFIGS / f"backends_{platform}.conf"
    for line in path.read_text().splitlines():
        name, sep, rest = line.split("#", 1)[0].partition("=")
        if sep:
            routes[name.strip()] = rest.strip()
    return routes


def target_ops(platform, wrappers, kernel_ops):
    """(sorted registered ops, sorted FlagGems-routed ops with no kernel)."""
    routes = conf_routes(platform)
    non_flaggems = NON_FLAGGEMS_ROUTES + VENDOR_ROUTES.get(platform, ())
    flaggems_routed = {op for op, value in routes.items() if value not in non_flaggems}
    registered = sorted(flaggems_routed & set(wrappers) & kernel_ops)
    unserved = sorted(flaggems_routed & kernel_ops - set(wrappers))
    no_kernel = sorted(flaggems_routed - kernel_ops)
    if unserved:
        raise SystemExit(
            f"backends_{platform}.conf routes these to a FlagGems backend but "
            f"{REGISTER_INC.relative_to(REPO)} has no wrapper: " + ", ".join(unserved)
        )
    return registered, no_kernel


def render(platform, wrappers, kernel_ops):
    ops, no_kernel = target_ops(platform, wrappers, kernel_ops)
    lines = [
        HEADER.format(
            platform=platform,
            count=len(ops),
            extra="",
        )
    ]
    if no_kernel:
        lines.append(EXCLUDED_HEADER)
        lines.extend(f"//   - {op}\n" for op in no_kernel)
    lines.append("\n")
    lines.extend(f'  m.impl("{op}", {wrappers[op]});\n' for op in ops)
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

    stale = []
    for platform in PLATFORMS:
        text = render(platform, wrappers, kernel_ops)
        out = GENERATED / f"register_flaggems_{platform}.inc"
        rel = out.relative_to(REPO)
        if args.check:
            current = out.read_text() if out.exists() else ""
            if current != text:
                stale.append(rel)
            continue
        out.write_text(text)
        registered, no_kernel = target_ops(platform, wrappers, kernel_ops)
        print(
            f"[gen] {rel}  ({len(registered)} m.impl lines, "
            f"{len(no_kernel)} FlagGems routes left unregistered)"
        )

    if args.check:
        if stale:
            raise SystemExit(
                "stale generated file(s): "
                + ", ".join(str(p) for p in stale)
                + f"; run: python3 {Path(__file__).name}"
            )
        print(f"[check] {len(PLATFORMS)} register_flaggems_*.inc files are up to date")


if __name__ == "__main__":
    main()
