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

"""Conf-route / registration consistency checks.

Every platform conf lists a route for every op the build can route, and each
route names a backend family. The call only reaches a kernel if the platform's
registration ``.inc`` also claims the op: an unregistered op with an
accelerated route is a "backend not registered" raise at runtime, and a
registered op left at ``none`` sends a real kernel to ``cpu_fallback``.

This is a pure text/parse check over the committed confs and generated
``.inc`` files. It needs no GPU, no ``flag_gems`` install, and no built
``torch_fl``, so it runs in milliseconds on any platform and is safe to wire
into the platform-agnostic CI job.

Run: pytest tests/unit/test_conf_registration_consistency.py -v
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "torch_fl" / "configs"
ATEN_DIR = REPO_ROOT / "csrc" / "aten"
GEN_DIR = ATEN_DIR / "generated"
BACKENDS_DIR = ATEN_DIR / "backends"

# Ops registered straight from register.cc rather than an ``.inc``, invisible
# to the ``m.impl`` scan. Mirrors ``EXTRA_NATIVE`` / ``EXTRA_ROUTED`` in
# ``scripts/codegen/gen_vendor_confs.py``.
EXTRA_NATIVE = {
    "ascend": {"matmul", "matmul_backward"},
    "iluvatar": {
        "_conj",
        "view_as_real",
        "add_.Tensor",
        "add_.Scalar",
        "div_.Tensor",
    },
    # The conv pair is claimed straight from register.cc's ``#else`` arm on
    # every platform there, because ATen's only other kernel for these ops
    # raises and cpu_fallback re-dispatches the *same* op, so it cannot reach
    # the raise. See CONV_CPU_CROSSING_PLATFORMS below.
    **{
        platform: {"convolution_overrideable"}
        for platform in ("nvidia", "hygon", "metax", "thead", "tsingmicro")
    },
}
EXTRA_ROUTED = {"scaled_dot_product_attention"}

# The native-kernel platforms whose build compiles that wrapper, so for them
# EXTRA_ROUTED is a real PrivateUse1 claim. Mirrors
# ``COMPOSITE_WRAPPER_VENDORS`` in ``scripts/codegen/gen_vendor_confs.py``; the
# `#if` it has to agree with is at csrc/aten/sdp_choice_stub.cc:99.
COMPOSITE_WRAPPER_PLATFORMS = {"iluvatar"}

# Platform -> registration ``.inc`` set that ``csrc/aten/register.cc`` compiles
# for that platform (the ``#include`` lines under each ``USE_*`` guard,
# register.cc:549-614).  Paths are relative to ``csrc/aten``.
PLATFORM_REGISTRATION = {
    "ascend": ["backends/ascend/generated/ascend_register.inc"],
    "enflame": [
        "backends/enflame/generated/gcu_register.inc",
        "backends/enflame/generated/gcu_flaggems_register.inc",
    ],
    "mthreads": [
        "backends/mthreads/generated/musa_register.inc",
        "backends/mthreads/generated/musa_flaggems_register.inc",
    ],
    # The BPU is native like apple, under the vendor's conf name: the generated
    # kernels submit single-operator .hbm artifacts from FLAGOS_BPU_OPLIB.
    "drobotics": ["backends/bpu/generated/bpu_register.inc"],
    # Kunlunxin is a native vendor backend like musa/gcu/ascend: it compiles its
    # own generated registration, which claims exactly the ops it routes to the
    # vendor XPU SDK operator library.  It no longer falls through to
    # generated/register.inc.
    "kunlunxin": ["backends/kunlunxin/generated/kunlunxin_register.inc"],
    # Iluvatar is FlagGems-primary with a native, non-boxing vendor slot
    # alongside it: the FlagGems dispatcher list is the default path, and a
    # separate native .inc (the corex ixdnn/ixblas kernels) claims exactly the
    # ops the conf routes to `iluvatar`. register.cc has an arm of its own
    # precisely so it does not reach the generated/register.inc fallback below,
    # which would claim every op and turn its `none` entries into "backend not
    # registered" instead of cpu_fallback. The two .inc files are disjoint --
    # both are included into the same PrivateUse1 TORCH_LIBRARY_IMPL body, so an
    # op in both would be a duplicate m.impl compile error.
    "iluvatar": [
        "backends/iluvatar/generated/iluvatar_register.inc",
        "backends/iluvatar/generated/iluvatar_flaggems_register.inc",
    ],
    # Boxing platforms (cuda, dcu, metax, ppu) and tsingmicro share the
    # register.cc ``#else`` arm. With boxing ON it includes
    # generated/register_boxing_<platform>.inc -- every op the conf routes
    # somewhere other than ``none`` that this wheel has a kernel for; with
    # boxing OFF -- the FlagGems-primary default -- it includes the matching
    # generated/register_flaggems_<platform>.inc, which claims only the ops the
    # conf routes to a FlagGems backend. Both are listed here because a build
    # compiles one or the other. A boxing build also includes the full
    # generated/register.inc after the boxed list, so the claimed set of that
    # arm is the boxed list plus whatever register.inc adds; the checks below
    # ask that the boxed list is a subset of it, which is what the claim gating
    # guarantees.
    "nvidia": [
        "generated/register_boxing_nvidia.inc",
        "generated/register_flaggems_nvidia.inc",
    ],
    "hygon": [
        "generated/register_boxing_hygon.inc",
        "generated/register_flaggems_hygon.inc",
    ],
    "metax": [
        "generated/register_boxing_metax.inc",
        "generated/register_flaggems_metax.inc",
    ],
    "thead": [
        "generated/register_boxing_thead.inc",
        "generated/register_flaggems_thead.inc",
    ],
    "tsingmicro": [
        "generated/register_boxing_tsingmicro.inc",
        "generated/register_flaggems_tsingmicro.inc",
    ],
    # MPS is a native vendor backend: register.cc's USE_MPS arm includes the
    # codegen'd apple registration, which claims only the ops with a MPSGraph
    # kernel (the `unary`, `binary`, `binary_scalar` and `reduce` categories so
    # far).  Note this is a PrivateUse1-dispatched include, not built-in MPS --
    # so the conf routes those ops to ``mps``, and the rest to ``none``.
    "apple": ["backends/apple/generated/mps_register.inc"],
}

# CUDA-boxing platforms spell the fallback key ``cuda`` (BOXING_FALLBACK in
# gen_vendor_confs.py), not the platform name.  Kunlunxin is no longer one of
# them: it is a native vendor backend, so its route key is ``kunlunxin``.
BOXING_PLATFORMS = {"nvidia", "hygon", "metax", "thead"}

PLATFORMS = sorted(PLATFORM_REGISTRATION)


def _parse_conf(path: Path) -> dict[str, str]:
    """Read ``<op> = <key>`` routes from a conf, stripping comments."""
    routes: dict[str, str] = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.split("#", 1)[0].partition("=")
        if sep:
            routes[name.strip()] = value.split("#", 1)[0].strip()
    return routes


def _soft_lowp_claims() -> set[str]:
    """The matrix ops register.cc claims from source for the soft-lowp path."""
    register_cc = (ATEN_DIR / "register.cc").read_text()
    # Comment lines count as neither a claim nor a duplicate: the block's own
    # explanation names the ops it holds back.
    code_lines = [
        line for line in register_cc.splitlines() if not line.lstrip().startswith("//")
    ]
    return set(
        re.findall(
            r'^\s*m\.impl\("((?:mm|bmm|addmm)[^"]*|_scaled_[^"]*)",',
            "\n".join(code_lines),
            re.M,
        )
    )


def _dropped_on(line: str, platform: str) -> bool:
    """Whether a claim line is excluded from ``platform``'s build by its guard.

    Only the one guard the block uses: ``#if !defined(FLAGOS_ACCEL_<X>)``.
    """
    match = re.match(r"\s*#if !defined\((FLAGOS_ACCEL_\w+)\)", line)
    return bool(match) and match.group(1) == SOFT_LOWP_PLATFORMS[platform]


def _lines_kept_on(block: str, platform: str) -> list[str]:
    """The lines of ``block`` compiled for ``platform``.

    Only the one guard the claim block uses is honoured: an
    ``#if !defined(<macro>)`` region is dropped when ``<macro>`` is the
    platform's own ``FLAGOS_ACCEL_*`` (and kept for every other platform).
    """
    own = SOFT_LOWP_PLATFORMS[platform]
    kept: list[str] = []
    guarded_on: str | None = None
    for line in block.splitlines():
        stripped = line.strip()
        match = re.match(r"#if !defined\((\w+)\)", stripped)
        if guarded_on is None and match:
            guarded_on = match.group(1)
            if guarded_on != own:
                kept.append(line)
            continue
        if guarded_on is not None and stripped == "#endif":
            if guarded_on != own:
                kept.append(line)
            guarded_on = None
            continue
        if guarded_on is None or guarded_on != own:
            kept.append(line)
    return kept


def _claim_block(register_cc: str, platform: str) -> set[str]:
    """The matrix-op claims register.cc makes for ``platform``, guard honoured.

    The block runs from its ``#if defined(FLAGOS_ACCEL_METAX) ||`` guard to the
    matching ``#endif`` -- nested guards and all -- not to the first one, which
    closes a nested region.
    """
    start = register_cc.rfind("#if defined(FLAGOS_ACCEL_METAX) ||")
    assert start != -1, "register.cc no longer opens the soft-lowp claim block"
    depth = 0
    offset = start
    end = None
    for line in register_cc[start:].splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#if"):
            depth += 1
        elif stripped.startswith("#endif"):
            depth -= 1
            if depth == 0:
                end = offset + len(line)
                break
        offset += len(line)
    assert end is not None, "the soft-lowp claim block is not closed"
    block = register_cc[start:end]
    return set(
        re.findall(
            r'^\s*m\.impl\("([^"]+)",', "\n".join(_lines_kept_on(block, platform)), re.M
        )
    )


# The matrix claims register.cc's soft-lowp block makes from source, keyed by
# the macro a build defines when it compiles the prelude (csrc/CMakeLists.txt).
PRELUDE_PLATFORMS = {"metax": "FLAGOS_ACCEL_METAX", "hygon": "FLAGOS_ACCEL_DCU"}


def _prelude_matrix_claims(platform: str) -> set[str]:
    """The matrix ops register.cc claims from source on ``platform``, or none."""
    if platform not in PRELUDE_PLATFORMS:
        return set()
    register_cc = (ATEN_DIR / "register.cc").read_text()
    start = register_cc.rfind("#if defined(FLAGOS_ACCEL_METAX) ||")
    end = register_cc.index("#endif", start)
    # The block's guards name the platform macro directly; the plain pair sits
    # in a nested `#if !defined(FLAGOS_ACCEL_METAX)`, so read the lines the
    # platform's own build keeps rather than the block's text.
    return set(
        re.findall(
            r'^\s*m\.impl\("([^"]+)",',
            "\n".join(_lines_kept_on(register_cc[start:end], platform)),
            re.M,
        )
    )


def _registered_ops(inc_path: Path) -> set[str]:
    """Read ``m.impl("op", ...)`` names from a registration ``.inc``."""
    if not inc_path.exists():
        return set()
    return set(re.findall(r'm\.impl\("([^"]+)"', inc_path.read_text()))


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", sorted(BOXING_PLATFORMS))
def test_a_boxing_claim_and_the_conf_route_agree_in_both_directions(platform):
    """The boxed list is the conf's route set, not a superset or a subset.

    register.cc's boxing arm claims exactly ``register_boxing_<platform>.inc``
    (plus the matrix ops the soft-lowp prelude claims from source in the same
    body, where that prelude's platform macro is defined).
    A claim with no route is "routed to 'none' ... but the op is registered on
    PrivateUse1", and a route with no claim -- an op the conf routes to
    ``boxing`` that nothing registered -- is "backend not registered" the first
    time a model calls it. Both directions are measured failures on DCU: mm
    after the arm stopped claiming unlisted wrappers (run 40156011560), and
    `empty_like` the other way round before the boxed list existed.
    """
    boxed = _registered_ops(GEN_DIR / f"register_boxing_{platform}.inc")
    boxed |= _prelude_matrix_claims(platform)
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    assert boxed, f"{platform} claims no boxed op"
    left_at_none = sorted(op for op in boxed if routes.get(op) == "none")
    assert not left_at_none, (
        f"{platform} claims ops its conf routes to none: {left_at_none[:10]}"
    )
    unlisted = sorted(op for op in boxed if op not in routes)
    assert not unlisted, (
        f"{platform} claims ops its conf does not list: {unlisted[:10]}"
    )
    routed_to_boxes = sorted(
        op for op, value in routes.items() if value.split("#")[0].strip() == "boxing"
    )
    unclaimed = sorted(set(routed_to_boxes) - boxed)
    assert not unclaimed, (
        f"{platform} routes {len(unclaimed)} op(s) to boxing that its boxed list "
        f"does not claim: {unclaimed[:10]}"
    )


def _platform_registered(platform: str) -> set[str]:
    """Union of ops registered by every ``.inc`` this platform compiles.

    Includes the two sets of hand-written claims that are invisible to the
    ``m.impl`` scan of any ``.inc``: ``register.cc``'s (EXTRA_NATIVE) and
    ``sdp_choice_stub.cc``'s (EXTRA_ROUTED, where that file is compiled --
    COMPOSITE_WRAPPER_PLATFORMS).
    """
    ops: set[str] = set(EXTRA_NATIVE.get(platform, set()))
    if platform in COMPOSITE_WRAPPER_PLATFORMS:
        ops |= EXTRA_ROUTED
    for rel in PLATFORM_REGISTRATION[platform]:
        ops |= _registered_ops(ATEN_DIR / rel)
    return ops


# ---------------------------------------------------------------------------
# Cross-file invariants
# ---------------------------------------------------------------------------


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", PLATFORMS)
def test_conf_exists_for_every_registered_platform(platform):
    conf = CONF_DIR / f"backends_{platform}.conf"
    assert conf.is_file(), f"missing conf for registered platform {platform}: {conf}"


@pytest.mark.anyplatform
def test_full_coverage_confs_share_one_op_set():
    """Every full-coverage conf covers the same (widened) op list.

    ``gen_vendor_confs.build_all()`` widens the CUDA-derived op list by the
    union of vendor-only ops (EXTRA_NATIVE + EXTRA_ROUTED) so the route counts
    in each header are commensurable.  The confs that get the full-coverage
    treatment are the native vendor confs (ascend, gcu, kunlunxin, musa), the
    # FlagGems-primary conf (iluvatar) and the CUDA-boxing confs (dcu, metax,
    # ppu).  ``backends_nvidia.conf`` is the *base*
    list (the one ``codegen_ops.py`` rewrites) and therefore lacks the three
    widened ops; ``backends_drobotics.conf`` routes 34 ops to its own kernel
    and ``backends_tsingmicro.conf`` is hand-written and sparse.  Those two are
    asserted separately.
    """
    full_coverage = {
        "ascend",
        "enflame",
        "kunlunxin",
        "mthreads",
        "hygon",
        "metax",
        "thead",
        "iluvatar",
    }
    reference = _parse_conf(CONF_DIR / "backends_mthreads.conf")
    for platform in sorted(full_coverage):
        routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
        assert set(routes) == set(reference), (
            f"{platform} conf covers {len(routes)} ops, "
            f"musa conf covers {len(reference)} ops; the full-coverage "
            "contract requires identical op sets"
        )


@pytest.mark.anyplatform
def test_cuda_conf_is_the_generated_base_list():
    """``backends_nvidia.conf`` is rewritten by ``codegen_ops.py`` and therefore
    covers exactly the generated op list, without the widened vendor-only ops.
    """
    routes = _parse_conf(CONF_DIR / "backends_nvidia.conf")
    widened = EXTRA_NATIVE["ascend"] | EXTRA_ROUTED
    assert not (set(routes) & widened), (
        f"backends_nvidia.conf should not list the widened vendor-only ops "
        f"{sorted(widened)}; it is the generated base list"
    )


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", PLATFORMS)
def test_every_route_uses_an_allowed_backend_key(platform):
    """Each route must name one of the legal backend keys.

    ``gen_vendor_confs.route()`` emits: ``flaggems_cpp``, ``flaggems``,
    ``tileops``, ``<vendor>`` (or ``cuda`` for a boxing platform), ``none``.
    Anything else is a generator bug or a hand-edit that will not match a
    ``Backend`` enum value at runtime.
    """
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    # A conf's route key is the vendor whose kernels serve it -- `nvidia` for
    # the CUDA boxing kernel, like every other vendor key.
    vendor_key = platform
    allowed = {
        "flaggems_cpp",
        "flaggems",
        "tileops",
        vendor_key,
        "none",
        "flagos_python",
    }
    unexpected = {
        op: value
        for op, value in routes.items()
        if value.split("#", 1)[0].strip() not in allowed
    }
    assert not unexpected, (
        f"{platform}: {len(unexpected)} routes use an unexpected key "
        f"(sample: {dict(list(unexpected.items())[:5])})"
    )


# ---------------------------------------------------------------------------
# Conf route -> registration containment
# ---------------------------------------------------------------------------


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "platform", ["mthreads", "enflame", "ascend", "kunlunxin", "iluvatar", "apple"]
)
def test_vendor_routes_are_registered(platform):
    """Every op routed to ``<vendor>`` must be in the platform's registration set.

    This is the invariant the earlier generator broke: routing an unregistered
    op to a vendor kernel names a backend no call can arrive at.  The call
    would raise "backend not registered" instead of falling back to CPU.
    """
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    registered = _platform_registered(platform)
    vendor_routed = {
        op for op, value in routes.items() if value.split("#", 1)[0].strip() == platform
    }
    unregistered = sorted(vendor_routed - registered)
    assert not unregistered, (
        f"{platform}: {len(unregistered)} ops routed to the vendor kernel "
        f"but not in {platform}'s registration set: {unregistered[:10]}"
    )


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "platform", ["mthreads", "enflame", "ascend", "kunlunxin", "iluvatar", "apple"]
)
def test_accelerated_routes_are_registered(platform):
    """Every non-``none`` route must be an op the platform registers.

    A route to ``flaggems`` / ``flaggems_cpp`` / ``tileops`` / ``<vendor>``
    only reaches a kernel if the op is claimed on PrivateUse1.  An op left
    unregistered with an accelerated route raises "backend not registered".
    """
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    registered = _platform_registered(platform)
    accelerated = {
        op for op, value in routes.items() if value.split("#", 1)[0].strip() != "none"
    }
    unregistered = sorted(accelerated - registered)
    assert not unregistered, (
        f"{platform}: {len(unregistered)} accelerated routes name ops not "
        f"in the registration set: {unregistered[:10]}"
    )


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "platform", ["mthreads", "enflame", "ascend", "kunlunxin", "iluvatar", "apple"]
)
def test_registered_ops_are_never_left_at_none(platform):
    """A registered op left at ``none`` sends a real kernel to ``cpu_fallback``.

    ``none`` is only correct for ops the platform *skips* on PrivateUse1.
    Registration claiming an op and the conf leaving it at ``none`` is drift:
    the kernel exists but no call reaches it.
    """
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    registered = _platform_registered(platform)
    none_ops = {
        op for op, value in routes.items() if value.split("#", 1)[0].strip() == "none"
    }
    stranded = sorted(none_ops & registered)
    assert not stranded, (
        f"{platform}: {len(stranded)} ops are registered but routed to "
        f"'none' (kernel exists but call falls back to CPU): {stranded[:10]}"
    )


# ---------------------------------------------------------------------------
# Conf shape invariants
# ---------------------------------------------------------------------------


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", sorted(BOXING_PLATFORMS))
def test_boxing_platforms_have_no_none_entries(platform):
    """The day-0 boxing variant boxes what it cannot accelerate, so ``none`` is
    never the right answer there -- that is what makes it a day-0 file. The
    default conf for the same platform is ``none``-based; see
    test_gen_vendor_confs.test_default_confs_never_route_boxing.
    """
    routes = _parse_conf(CONF_DIR / f"backends_{platform}_boxing.conf")
    none_ops = [op for op, v in routes.items() if v.split("#", 1)[0].strip() == "none"]
    assert not none_ops, f"{platform} routes {none_ops[:5]} to none"


@pytest.mark.anyplatform
def test_drobotics_conf_matches_its_register_file():
    """Every op ``backends_drobotics.conf`` routes has a kernel in the .inc.

    The BPU is native like apple: the conf routes exactly the ops
    backends/bpu/generated/bpu_register.inc claims, and nothing else. A route
    without a kernel behind it would fail the dispatcher's "backend not
    registered" check instead of falling back, so the two must agree.
    """
    routes = _parse_conf(CONF_DIR / "backends_drobotics.conf")
    registered = _registered_ops(
        REPO_ROOT / "csrc/aten/backends/bpu/generated/bpu_register.inc"
    )
    routed_to_kernel = {op for op, backend in routes.items() if backend == "drobotics"}
    assert routed_to_kernel == registered, (
        "backends_drobotics.conf and bpu_register.inc disagree: "
        f"conf-only={sorted(routed_to_kernel - registered)} "
        f"kernel-only={sorted(registered - routed_to_kernel)}"
    )
    assert routed_to_kernel, "drobotics conf routes nothing to its own kernel"
    # Everything else is explicitly `none` (cpu_fallback), as a full-coverage
    # conf states rather than omitting.
    assert set(routes.values()) <= {"drobotics", "none"}


@pytest.mark.anyplatform
def test_mps_conf_is_a_generated_full_coverage_conf():
    """``backends_apple.conf`` is a generated full-coverage conf like the vendor
    backends: it lists every routable op exactly once, routing only the
    MPSGraph-kernel cohort to ``mps`` and the rest to ``none``.  See
    docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md.
    """
    routes = _parse_conf(CONF_DIR / "backends_apple.conf")
    assert routes, "backends_apple.conf lists no routes"
    assert set(routes.values()) <= {"apple", "none"}, (
        "backends_apple.conf may only route ops to 'mps' or 'none'"
    )
    # One op per MPSGraph category the codegen emits (unary, binary,
    # binary_scalar, reduce, gemm, softmax, pow_scalar, conv). A category whose
    # entry silently dropped would leave its ops on cpu_fallback and be invisible
    # in a count-only check. Both mm and bmm are listed because the gemm category
    # emits one kernel per op: a rank guard that reached one of them and not the
    # other is exactly what a single representative case would miss. ``neg`` is
    # the second op of the ``unary`` category -- the same template, so it is the
    # category's registration that a representative would otherwise not cover.
    # ``convolution_overrideable`` is the conv category's only op, and the one
    # every ``F.conv2d``/``F.conv3d`` lowers to, because ``aten.convolution`` is
    # CompositeImplicitAutograd and decomposes above the backend key.
    # ``silu_backward`` is the first backward-only op, so it has no forward
    # representative.
    for op in (
        "abs",
        "neg",
        "add.Tensor",
        "sub.Scalar",
        "mul.Tensor",
        "mean.dim",
        "sum.dim_IntList",
        "_softmax",
        "pow.Tensor_Scalar",
        "mm",
        "bmm",
        "convolution_overrideable",
        "silu",
        "silu_backward",
        "embedding",
        "index_select",
        "take",
        "gather",
        "div.Tensor",
        "div.Scalar",
        "reciprocal",
        "native_layer_norm",
        "softplus",
        "transpose.int",
        "slice.Tensor",
        "unsqueeze",
        "t",
        "select.int",
        "squeeze.dim",
        "_unsafe_view",
        "cat",
        "where.self",
    ):
        assert routes.get(op) == "apple", f"the {op} kernel must be routed to mps"


@pytest.mark.anyplatform
def test_tsingmicro_conf_is_hand_written_and_sparse():
    """``backends_tsingmicro.conf`` is deliberately hand-written and covers
    only the ops tsingmicro routes; it is not a generated full-coverage conf.
    """
    routes = _parse_conf(CONF_DIR / "backends_tsingmicro.conf")
    assert routes, "backends_tsingmicro.conf is empty"
    assert len(routes) < 200, (
        f"backends_tsingmicro.conf covers {len(routes)} ops; it is expected "
        "to stay a small hand-written conf"
    )


def _derived_iluvatar_counts() -> dict[str, int]:
    """Iluvatar route counts derived from its two registration ``.inc`` files.

    Iluvatar is FlagGems-primary with a native, non-boxing vendor slot, and the
    native cohort is a *moving measurement* (scripts/codegen/codegen_iluvatar.py)
    rather than a fixed list: it grows as the ixdnn/ixblas coverage is measured.
    Pinning a literal here would mean hand-editing this test every time the
    cohort changes, so the counts are derived from what the platform actually
    registers instead. The FlagGems ``.inc`` claims the ``flaggems`` routes; the
    native ``.inc`` claims the ``iluvatar`` routes, and because FlagGems wins any
    overlap the native set is the native-minus-FlagGems difference; the ops
    register.cc claims directly (EXTRA_NATIVE, e.g. ``_conj``) belong to that
    same native slot; the composite ``sdp_choice_stub.cc`` claims
    (COMPOSITE_WRAPPER_PLATFORMS) belongs to the ``flaggems`` slot, since its
    route is what that wrapper reads to take the FlagGems path; every other op
    the conf lists is ``none``. A conf that disagrees with the registrations
    is caught by test_vendor_routes_are_registered / _accelerated_routes_are_
    registered / test_registered_ops_are_never_left_at_none above.
    """
    routes = _parse_conf(CONF_DIR / "backends_iluvatar.conf")
    flaggems = _registered_ops(
        ATEN_DIR / "backends/iluvatar/generated/iluvatar_flaggems_register.inc"
    )
    if "iluvatar" in COMPOSITE_WRAPPER_PLATFORMS:
        flaggems |= EXTRA_ROUTED
    native = _registered_ops(
        ATEN_DIR / "backends/iluvatar/generated/iluvatar_register.inc"
    )
    iluvatar = (native | EXTRA_NATIVE["iluvatar"]) - flaggems
    return {
        "flaggems": len(flaggems),
        "iluvatar": len(iluvatar),
        "none": len(routes) - len(flaggems) - len(iluvatar),
    }


@pytest.mark.anyplatform
def test_conf_route_counts_are_stable():
    """Snapshot of the route counts per conf.  A change here means an op
    moved between routes: update this test deliberately, not silently.
    """
    expected = {
        # mm and mm.out moved from the aclnn route to flaggems; see
        # NATIVE_TRITON_GAPS in scripts/codegen/gen_vendor_confs.py.
        #
        # flip moved from the flaggems route to ascend for the same reason;
        # the FlagTree 0.7.0 Ascend MLIR pipeline cannot lower the rank-2
        # copy_func kernel the FlagGems `_ascend` flip override builds.
        # Real-valued sgn now reaches ACLNN Sign for abs backward.
        "ascend": {"flaggems": 221, "none": 1658, "ascend": 158},
        # The BPU is native: 34 ops route to the generated kernel layer, the
        # rest to `none`. Those kernels submit .hbm artifacts from
        # FLAGOS_BPU_OPLIB; without it each takes its own CPU round-trip, which
        # a routing table cannot express.
        "drobotics": {"drobotics": 34, "none": 2003},
        # Measured on H100 after rebasing onto the main that carries #340.
        # #340 added three exception groups (cuda_route_exceptions for
        # contract/compile defects, measured_flaggems_rollback for ops that
        # failed a surveyed case, metadata_route_overrides for ops that buy no
        # kernel on either route) and moved cuda to 390/1644. This branch adds
        # the other half: the FlagGems ceiling is now derived from the generated
        # kernels instead of being hand-maintained, so ops that had a kFlagGems
        # kernel but were never listed -- mul.Tensor above all, 1522 host round
        # trips per forward on the OCR-RM value head -- reach the FlagGems route
        # instead of cpu_fallback. Both effects apply: 549/1485.
        "nvidia": {"flaggems": 549, "nvidia": 1485},
        # dcu/metax/ppu are CUDA-ABI platforms with no vendor kernel of their
        # own on this tree, and a default conf never routes boxing (AGENTS.md,
        # "Day-0 boxing confs"): what FlagGems does not cover -- including each
        # platform's measured triton gaps -- is `none` (cpu_fallback). The
        # boxing routes live in backends_<platform>_boxing.conf.
        "hygon": {"flaggems": 456, "none": 1581},
        # embedding and embedding.out gained a native topsaten kernel and now
        # route to gcu instead of falling through to none; see
        # NATIVE_TRITON_GAPS in scripts/codegen/gen_vendor_confs.py.
        #
        # cat is the exception to the cat-flaggems reroute: the FlagGems route
        # does not compile on this vendor, so NATIVE_TRITON_GAPS["enflame"] keeps it
        # on the gcu kernel. See the entry in scripts/codegen/gen_vendor_confs.py.
        #
        # rsub.Scalar / rsub.Tensor moved flaggems -> gcu: FlagGems' rsub Triton
        # kernel is rejected by the Enflame front end (64-bit data type not
        # supported on GCU300), so they are served by the native topsaten kernel
        # instead. See NATIVE_TRITON_GAPS["enflame"].
        "enflame": {"none": 1603, "enflame": 183, "flaggems": 251},
        # kunlunxin is a native vendor backend: every op routes to the vendor
        # XPU SDK (``kunlunxin``) or is left at ``none`` -- there is no ``cuda``
        # route.  The count is exactly what
        # backends/kunlunxin/generated/kunlunxin_register.inc registers: the ten
        # conformance ops with an xdnn_pytorch kernel, plus the RNG cohort the
        # handwritten rng.cc covers (see HANDWRITTEN_OPS in
        # scripts/codegen/codegen_kunlunxin.py).  Without that route the RNG ops
        # fell through to cpu_fallback, which the flagos PrivateUse1 generator
        # cannot steer -- the two failures the CI lane reported.
        "kunlunxin": {"kunlunxin": 57, "none": 1980},
        # metax: _softmax/embedding moved off flaggems_cpp to flaggems (#501).
        # 8 ops moved flaggems -> boxing (594 -> 586 / 1433 -> 1441) because
        # route_boxing now consults codegen_ops.py's FlagGems holds. These were
        # ops the metax conf routed to the Python path while
        # flaggems_runtime_broken listed them as broken -- a route the hold list
        # has always said not to take, which the generator could not see until
        # the sets moved to module scope. They are the `out=`-contract group
        # (_softmax.out, _log_softmax.out and their backward_data forms),
        # _cdist_backward (gems asserts a real CUDA tensor), and
        # _native_batch_norm_legit_functional.
        #
        # NOT REVALIDATED on MetaX hardware: the re-route is the hold list's
        # doing, applied uniformly, and was measured only on H100. MetaX needs
        # the differential survey run before this row can claim that moving
        # these 8 ops to `cuda` costs nothing there. If it does cost something,
        # the per-platform answer is NATIVE_TRITON_GAPS["metax"], not a
        # deletion from the shared holds.
        #
        # `mm` moved none -> flaggems (586 -> 587 / 1436 -> 1435): the soft-lowp
        # matrix claim needs the key registered, and a default conf may not route
        # boxing. The day-0 variant keeps `boxing`. Measured on a C550; see
        # METAX_FLAGGEMS_MEASURED in gen_vendor_confs.py.
        "metax": {"flaggems": 587, "none": 1435, "flaggems_cpp": 15},
        # cat moved to flaggems via the tensorlist_dim codegen category (#498).
        "mthreads": {"flaggems": 466, "none": 1518, "mthreads": 53},
        "thead": {"flaggems": 592, "none": 1445},
        # Iluvatar adds a native, non-boxing vendor slot on top of FlagGems
        # (scripts/codegen/codegen_iluvatar.py). Its counts are DERIVED, not
        # pinned: the native cohort is a moving measurement -- the first landing
        # is _convolution, convolution, native_batch_norm, the ops that route
        # `none` today and have an ixdnn/ixblas kernel, and it grows as coverage
        # is measured. Ops FlagGems already registers (the
        # _native_batch_norm_legit* variants) stay `flaggems` and are kept out of
        # the native .inc, since a second m.impl in the same PrivateUse1 body is a
        # duplicate-registration error; mm/bmm/addmm left that group with the
        # issue-#570 gaps and are native routes now. See
        # _derived_iluvatar_counts().
        "iluvatar": _derived_iluvatar_counts(),
        # MPS is a native vendor backend at stage 4: the codegen'd
        # mps_register.inc claims the MPSGraph kernels registered for the
        # ``unary``, ``binary``, ``binary_scalar``, ``reduce``, ``gemm``,
        # ``softmax``, ``pow_scalar``, ``conv``, ``silu``, ``silu_backward``,
        # ``embedding``, ``index_select``, ``gather``, ``div``, ``div_scalar``,
        # ``reciprocal``, ``take``, ``layer_norm``, ``batch_norm``, ``sigmoid``,
        # ``exp``, ``log``, ``rsqrt``, ``sqrt``, ``tanh``, ``erf``, ``where``
        # ``cat``, ``masked_fill``, ``all``, ``argmax``, ``triu`` and
        # ``SOLVE_TRIANGULAR``, ``index``, ``cumsum``, ``softplus``, the
        # trigonometric and predicate spellings, the vision interpolation
        # factories, clamp, GELU, rsub.Scalar and the eight metadata views
        # (66 ops so far), so the conf routes those to
        # ``mps`` and every other op to ``none``. The ``softmax`` category is registered
        # under ``_softmax`` alone because ``softmax.int`` is
        # CompositeImplicitAutograd and never reaches a backend key, and the
        # ``conv`` category under ``convolution_overrideable`` alone for the same
        # reason about ``aten.convolution``; see scripts/codegen/codegen_mps.py.
        "apple": {"apple": 66, "none": 1971},
        # Twelve ops that cpu_fallbacked in the Qwen-Image-2.1 census moved to
        # flaggems (all.dim, clamp_min, div.Tensor, gelu, gt.Scalar, gt.Tensor,
        # index_select, isneginf, nonzero, tanh, tril, zero_); 51 -> 63, then
        # native_batch_norm and where.self_out (the math-SDPA fallbacks) -> 65. The
        # survey's 12/12 STRICT result and the measurements behind the two
        # additions are recorded in docs/reference/operator-support.md.
        "tsingmicro": {"flaggems": 65, "flagos_python": 2, "tsingmicro": 14},
    }
    for platform, exp in expected.items():
        routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
        counts = dict(Counter(routes.values()))
        assert counts == exp, (
            f"{platform} route counts drifted: expected {exp}, got {counts}. "
            "If an op moved routes deliberately, update this snapshot."
        )


# ---------------------------------------------------------------------------
# The software low-precision matrix path: claimed from source, like conv
# ---------------------------------------------------------------------------

# The matrix ops whose generated wrapper opens with the ``FLAGOS_SOFT_LOWP``
# prelude, and the ``_scaled_mm`` family that prelude's gate must keep
# fail-closed. A wrapper is not a route: an op the conf leaves on ``none``
# reaches ATen's CompositeExplicitAutograd arm instead of the wrapper, so on a
# FlagGems-primary build the prelude is unreachable unless register.cc claims
# the op. The gate beside it (``_scaled_mm`` failing closed on an FP8 pair) is
# then the inconsistency: the same wheel gated one fingerprint and not the
# other, and an FP8 mm answered through the conf's route instead (measured on
# DCU, bw1100).
# The forms a decoded pair can be routed by where the conf leaves the form's
# own key `none`: claimed under the mm/bmm key, which is where the dispatcher
# sends the call (SOFT_LOWP_PAIR_CLAIMS in the registration list generator).
SOFT_LOWP_PAIR_FORMS = {
    "mm.dtype": "mm",
    "mm.dtype_out": "mm",
    "mm.out": "mm",
    "bmm.dtype": "bmm",
    "bmm.dtype_out": "bmm",
    "bmm.out": "bmm",
}
_FORM_SUFFIX = {
    "mm.dtype": "Mdtype",
    "mm.dtype_out": "MdtypeOut",
    "mm.out": "Mout",
    "bmm.dtype": "Bdtype",
    "bmm.dtype_out": "BdtypeOut",
    "bmm.out": "Bout",
}

SOFT_LOWP_CLAIMS = {
    "addmm",
    "addmm.dtype",
    "addmm.dtype_out",
    "addmm.out",
    "addmm_",
    "bmm",
    "bmm.dtype",
    "bmm.dtype_out",
    "bmm.out",
    "mm",
    "mm.dtype",
    "mm.dtype_out",
    "mm.out",
    "_scaled_grouped_mm",
    "_scaled_grouped_mm_v2",
    "_scaled_mm",
    "_scaled_mm.out",
    "_scaled_mm_v2",
}

# The CUDA-ABI platforms whose FlagGems-primary build defines FLAGOS_SOFT_LOWP.
SOFT_LOWP_PLATFORMS = {"hygon": "FLAGOS_ACCEL_DCU", "metax": "FLAGOS_ACCEL_METAX"}


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", sorted(SOFT_LOWP_PLATFORMS))
def test_soft_lowp_matrix_ops_are_claimed_on_flaggems_primary_builds(platform):
    """A platform that compiles the prelude must also reach it for every op.

    Reached means registered, from either list: ``register.cc`` claims the ops
    the conf leaves on ``none``, and the generated FlagGems list names the same
    ``Wrapper*`` symbols for the ops the conf routes to a FlagGems backend
    (which is why ``register.cc`` must not claim those again). An op in
    neither list has no PrivateUse1 kernel at all, so the prelude inside its
    wrapper never runs.
    """
    register_cc = (ATEN_DIR / "register.cc").read_text()
    # The block the claim sits in, not the whole file: `mm.out` is named by
    # every one of these guards (the conv crossing and the pair claims), and a
    # substring search of the file would credit a platform with a claim the
    # guard above is there to withhold.
    claimed_here = _claim_block(register_cc, platform)
    listed = _registered_ops(GEN_DIR / f"register_flaggems_{platform}.inc")
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    # The pair's own keys have a route on every platform (dcu's `mm`/`bmm` to
    # FlagGems, metax's `bmm` to `flaggems_cpp`; metax's `mm` is `none` and so
    # owes nothing, which is what its conf says).
    forms = {
        op for op, key in SOFT_LOWP_PAIR_FORMS.items() if routes.get(key) != "none"
    } | {
        op
        for op in SOFT_LOWP_CLAIMS
        if routes.get(op) != "none" or op in claimed_here or op in listed
    }
    claimed = claimed_here | _prelude_matrix_claims(platform)
    # The list's plain entry for the pair's own key serves every form of that
    # key (metax's `bmm` on `flaggems_cpp`): a `none` out form takes the pair
    # forwarder, whose route is that key. A form with a route of its own is the
    # list's plain wrapper.
    plain_served = {op for op, key in SOFT_LOWP_PAIR_FORMS.items() if key in listed}
    missing = sorted(
        op
        for op in forms
        if op not in claimed and op not in listed and op not in plain_served
    )
    assert not missing, (
        f"{platform}: the software low-precision matrix ops {missing} are "
        "registered on PrivateUse1 by neither register.cc nor the generated "
        "FlagGems list, so a FlagGems-primary build never reaches the "
        "FLAGOS_SOFT_LOWP prelude for them"
    )


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", sorted(SOFT_LOWP_PLATFORMS))
def test_soft_lowp_claims_are_gated_on_the_platform_macro(platform):
    """The claim must sit under a macro the platform defining the prelude sets."""
    register_cc = (ATEN_DIR / "register.cc").read_text()
    expected = SOFT_LOWP_PLATFORMS[platform]
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    # The claims this block makes itself: `mm` and `bmm` where the conf does not
    # route the pair to a list that serves it. The other matrix forms are the
    # list's (see SOFT_LOWP_PAIR_CLAIMS), so they are checked there.
    paired = routes.get("mm") == "none"
    claimed_here = _claim_block(register_cc, platform)
    claims = ['m.impl("mm", WrapperMm);', 'm.impl("bmm", WrapperBmm);']
    if not paired:
        idx = register_cc.rfind(claims[0])
        assert idx != -1, "mm is not claimed from register.cc"
        guard_block = register_cc[max(0, idx - 1600) : idx]
        assert expected in guard_block, (
            f"{platform}: the soft-lowp matrix claim is not gated on {expected}. "
            f"Guard text found:\n{guard_block}"
        )
    else:
        # Nothing to gate: `mm` is on `none` here, so this platform's build must
        # not compile the claim -- a claim on a `none` route is the dispatcher's
        # raise for every ordinary call (metax, run 40629858404). Read through
        # the guard, not the file: the line is still there for the platforms
        # that do want it.
        assert "mm" not in claimed_here and "bmm" not in claimed_here, (
            f"{platform}: the conf routes `mm` to `none`, so register.cc must "
            "not claim mm/bmm for it -- the claim turns every torch.mm into "
            "\"routed to 'none' ... but the op is registered on PrivateUse1\""
        )
    # Where the conf leaves a form `none`, the pair must be claimed (or, for
    # `bmm`/`bmm.out`, served by the list under its own key) -- and the claim
    # must follow the pair's route, not the form's slot. Where the form names a
    # FlagGems backend the plain wrapper is already the list's claim.
    listed_text = (GEN_DIR / f"register_flaggems_{platform}.inc").read_text()
    listed = _registered_ops(GEN_DIR / f"register_flaggems_{platform}.inc")
    for op, key in SOFT_LOWP_PAIR_FORMS.items():
        if routes.get(op) != "none":
            continue
        # Every route the pair can answer by: a claim register.cc makes, a
        # route of its own, or its plain key already claimed by the list.
        served = (
            key in claimed_here
            or key in _prelude_matrix_claims(platform)
            or routes.get(key) != "none"
            or key in listed
        )
        if not served:
            continue
        forwarder = {
            "mm": f"native::flagos::soft_lowp::Wrapper{_FORM_SUFFIX[op]}AsMm",
            "bmm": f"native::flagos::soft_lowp::Wrapper{_FORM_SUFFIX[op]}AsBmm",
        }[key]
        assert f'm.impl("{op}", {forwarder});' in listed_text, (
            f"{platform}: {op} is routed `none` and the generated list does not "
            "claim it by the mm/bmm key, so the call refuses before the prelude"
        )


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", sorted(SOFT_LOWP_PLATFORMS))
def test_soft_lowp_claims_do_not_duplicate_the_generated_list(platform):
    """A second m.impl for one op in one TORCH_LIBRARY_IMPL body is a compile error.

    The generated FlagGems list is included in the same body as the claims
    above, so an op it already names -- metax's ``bmm``, whose conf routes it
    to ``flaggems_cpp``, and every matrix form the soft-lowp pair claims -- must
    not be claimed again. Such an op is guarded out by name, on every platform:
    a guard that only excludes the platform it was written for still
    duplicates on the other one.
    """
    register_cc = (ATEN_DIR / "register.cc").read_text()
    claimed_here = _claim_block(register_cc, platform)
    listed = _registered_ops(GEN_DIR / f"register_flaggems_{platform}.inc")
    duplicated = sorted(claimed_here & listed)
    assert not duplicated, (
        f"{platform}: {duplicated} are claimed by register.cc and already named "
        "by the generated FlagGems list; the second m.impl does not compile"
    )


# ---------------------------------------------------------------------------
# Convolution: the one claim that bypasses both the conf and cpu_fallback
# ---------------------------------------------------------------------------

# ``register.cc`` claims these on PrivateUse1 from source rather than from a
# generated ``.inc``, because neither route can serve them: the conf's
# ``cuda``/``<vendor>`` key needs a boxed kernel a FlagGems-primary build does
# not compile, and cpu_fallback re-dispatches the *same* op onto ATen's raising
# CompositeExplicitAutograd stub. The forward and the backward are claimed
# separately because what the generated list already covers differs: the
# CUDA-ABI confs route conv to ``cuda`` on cuda/dcu (so the FlagGems list names
# neither op) and to ``flaggems`` on metax/ppu (so it already names the
# forward, and claiming it again would duplicate an m.impl in the same
# TORCH_LIBRARY_IMPL body -- a compile error).
CONV_CPU_CROSSING = {
    "nvidia": {"convolution_overrideable", "convolution_backward_overrideable"},
    "hygon": {"convolution_overrideable", "convolution_backward_overrideable"},
    "tsingmicro": {
        "convolution_overrideable",
        "convolution_backward_overrideable",
    },
    "metax": {"convolution_backward_overrideable"},
    "thead": {"convolution_backward_overrideable"},
}


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", sorted(CONV_CPU_CROSSING))
def test_conv_overrideable_is_claimed_on_flaggems_primary_builds(platform):
    """The conv pair must be claimed by the *registration*, not just the conf.

    FlagGems-primary builds compile ``register_flaggems_<platform>.inc`` and
    claim only what it names, so an op left out of it has no kernel at all.
    ``convolution_overrideable`` falls to the raising stub in that case, and
    every conv on the device fails -- which is how a Qwen3.5-VL
    (GatedDeltaNet) forward died with "convolution_overrideable not
    implemented" on 8x H100.
    """
    register_cc = (ATEN_DIR / "register.cc").read_text()
    for op in CONV_CPU_CROSSING[platform]:
        assert op in register_cc, (
            f"{platform}: {op} is not claimed anywhere in register.cc, so a "
            "FlagGems-primary build has no kernel for it"
        )


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", sorted(CONV_CPU_CROSSING))
def test_conv_claim_is_reachable_from_the_platform_arm(platform):
    """The conv claim must be gated on a macro this platform actually defines.

    The claim lives in the CUDA-ABI ``#else`` arm, which ``FLAGOS_ACCEL_<UPPER>``
    selects -- cuda and ppu have an empty ``USE_*`` macro because they re-tag a
    vendor CUDA build rather than own one. The regression this guards is the
    one that shipped: the claim was written under ``USE_TSINGMICRO`` alone, so
    it compiled everywhere and still left every CUDA-ABI platform with no conv
    kernel.
    """
    register_cc = (ATEN_DIR / "register.cc").read_text()
    expected = {
        "nvidia": "FLAGOS_ACCEL_CUDA",
        "hygon": "FLAGOS_ACCEL_DCU",
        "metax": "FLAGOS_ACCEL_METAX",
        "thead": "FLAGOS_ACCEL_PPU",
        "tsingmicro": "FLAGOS_ACCEL_TSINGMICRO",
    }[platform]

    # The claim sits immediately after the "already claim(ed)" explanatory
    # comment in the CUDA-ABI arm; take the guard lines just above it.
    # rfind, not find: BPU and Kunlunxin claim these ops earlier in the file
    # from their own arms, and only the last claim of each is the CUDA-ABI one.
    # metax/ppu claim the backward only, so anchor on that for them.
    op = (
        "convolution_backward_overrideable"
        if platform in ("metax", "thead")
        else "convolution_overrideable"
    )
    # Match on the op name only: the backward claim is wrapped across two
    # lines in register.cc, so anchoring on the dispatcher symbol would miss it.
    marker = f'm.impl("{op}",'
    idx = register_cc.rfind(marker)
    assert idx != -1, f"{op} is not claimed from register.cc"
    guard_block = register_cc[max(0, idx - 900) : idx]
    assert expected in guard_block, (
        f"{platform}: the convolution_overrideable claim is not gated on "
        f"{expected}. Guard text found:\n{guard_block}"
    )


@pytest.mark.anyplatform
@pytest.mark.parametrize("platform", ["metax", "thead"])
def test_flaggems_list_already_covers_the_forward_on_metax_and_ppu(platform):
    """These two must NOT claim the forward twice.

    Their conf routes conv to ``flaggems``, so the generated list claims
    ``convolution_overrideable``. A second ``m.impl`` for the same op inside
    the same ``TORCH_LIBRARY_IMPL`` body does not compile.
    """
    inc = _registered_ops(GEN_DIR / f"register_flaggems_{platform}.inc")
    assert "convolution_overrideable" in inc, (
        f"{platform}: the generated FlagGems list no longer names "
        "convolution_overrideable, so the forward now needs the CPU crossing "
        "too -- update CONV_CPU_CROSSING and the register.cc guard"
    )
    assert "convolution_backward_overrideable" not in inc, (
        f"{platform}: the generated list now claims the backward, so "
        "register.cc's wrapper claim would duplicate it"
    )
