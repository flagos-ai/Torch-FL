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
EXTRA_NATIVE = {"ascend": {"matmul", "matmul_backward"}}
EXTRA_ROUTED = {"scaled_dot_product_attention"}

# Platform -> registration ``.inc`` set that ``csrc/aten/register.cc`` compiles
# for that platform (the ``#include`` lines under each ``USE_*`` guard,
# register.cc:549-614).  Paths are relative to ``csrc/aten``.
PLATFORM_REGISTRATION = {
    "ascend": ["backends/ascend/generated/ascend_register.inc"],
    "gcu": [
        "backends/gcu/generated/gcu_register.inc",
        "backends/gcu/generated/gcu_flaggems_register.inc",
    ],
    "musa": [
        "backends/musa/generated/musa_register.inc",
        "backends/musa/generated/musa_flaggems_register.inc",
    ],
    "bpu": [],  # BPU registers no compute ops (register.cc:591-613)
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
    # register.cc ``#else`` arm. With boxing ON it includes generated/register.inc
    # (every generated op); with boxing OFF -- the FlagGems-primary default -- it
    # includes the matching generated/register_flaggems_<platform>.inc, which
    # claims only the ops the conf routes to a FlagGems backend. Both are listed
    # here because a build compiles one or the other.
    "cuda": ["generated/register.inc", "generated/register_flaggems_cuda.inc"],
    "dcu": ["generated/register.inc", "generated/register_flaggems_dcu.inc"],
    "metax": ["generated/register.inc", "generated/register_flaggems_metax.inc"],
    "ppu": ["generated/register.inc", "generated/register_flaggems_ppu.inc"],
    "tsingmicro": [
        "generated/register.inc",
        "generated/register_flaggems_tsingmicro.inc",
    ],
    # MPS is a native vendor backend: register.cc's USE_MPS arm includes the
    # codegen'd apple registration, which claims only the ops with a MPSGraph
    # kernel (the `unary`, `binary`, `binary_scalar` and `reduce` categories so
    # far).  Note this is a PrivateUse1-dispatched include, not built-in MPS --
    # so the conf routes those ops to ``mps``, and the rest to ``none``.
    "mps": ["backends/mps/generated/mps_register.inc"],
}

# CUDA-boxing platforms spell the fallback key ``cuda`` (BOXING_FALLBACK in
# gen_vendor_confs.py), not the platform name.  Kunlunxin is no longer one of
# them: it is a native vendor backend, so its route key is ``kunlunxin``.
BOXING_PLATFORMS = {"cuda", "dcu", "metax", "ppu"}

PLATFORMS = sorted(PLATFORM_REGISTRATION)


def _parse_conf(path: Path) -> dict[str, str]:
    """Read ``<op> = <key>`` routes from a conf, stripping comments."""
    routes: dict[str, str] = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.split("#", 1)[0].partition("=")
        if sep:
            routes[name.strip()] = value.split("#", 1)[0].strip()
    return routes


def _registered_ops(inc_path: Path) -> set[str]:
    """Read ``m.impl("op", ...)`` names from a registration ``.inc``."""
    if not inc_path.exists():
        return set()
    return set(re.findall(r'm\.impl\("([^"]+)"', inc_path.read_text()))


def _platform_registered(platform: str) -> set[str]:
    """Union of ops registered by every ``.inc`` this platform compiles.

    Includes the ops registered straight from ``register.cc`` (EXTRA_NATIVE),
    which are invisible to the ``m.impl`` scan of any ``.inc``.
    """
    ops: set[str] = set(EXTRA_NATIVE.get(platform, set()))
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
    # ppu).  ``backends_cuda.conf`` is the *base*
    list (the one ``codegen_ops.py`` rewrites) and therefore lacks the three
    widened ops; ``backends_bpu.conf`` is empty and ``backends_tsingmicro.conf``
    is hand-written and sparse.  Those three are asserted separately.
    """
    full_coverage = {
        "ascend",
        "gcu",
        "kunlunxin",
        "musa",
        "dcu",
        "metax",
        "ppu",
        "iluvatar",
    }
    reference = _parse_conf(CONF_DIR / "backends_musa.conf")
    for platform in sorted(full_coverage):
        routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
        assert set(routes) == set(reference), (
            f"{platform} conf covers {len(routes)} ops, "
            f"musa conf covers {len(reference)} ops; the full-coverage "
            "contract requires identical op sets"
        )


@pytest.mark.anyplatform
def test_cuda_conf_is_the_generated_base_list():
    """``backends_cuda.conf`` is rewritten by ``codegen_ops.py`` and therefore
    covers exactly the generated op list, without the widened vendor-only ops.
    """
    routes = _parse_conf(CONF_DIR / "backends_cuda.conf")
    widened = EXTRA_NATIVE["ascend"] | EXTRA_ROUTED
    assert not (set(routes) & widened), (
        f"backends_cuda.conf should not list the widened vendor-only ops "
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
    vendor_key = "cuda" if platform in BOXING_PLATFORMS else platform
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
    "platform", ["musa", "gcu", "ascend", "kunlunxin", "iluvatar", "mps"]
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
    "platform", ["musa", "gcu", "ascend", "kunlunxin", "iluvatar", "mps"]
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
    "platform", ["musa", "gcu", "ascend", "kunlunxin", "iluvatar", "mps"]
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
    """A CUDA-compatible platform claims the full generated op list, so every
    op has a boxing kernel behind it and ``none`` is never the right answer.
    """
    routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
    none_ops = [op for op, v in routes.items() if v.split("#", 1)[0].strip() == "none"]
    assert not none_ops, f"{platform} routes {none_ops[:5]} to none"


@pytest.mark.anyplatform
def test_bpu_conf_is_empty():
    """BPU registers no compute ops (register.cc:591-613), so its conf must
    list no routes: every op falls through to ``cpu_fallback`` and acceleration
    comes from the ``torch.compile`` backend.
    """
    routes = _parse_conf(CONF_DIR / "backends_bpu.conf")
    assert not routes, (
        f"backends_bpu.conf lists {len(routes)} routes but BPU registers none"
    )


@pytest.mark.anyplatform
def test_mps_conf_is_a_generated_full_coverage_conf():
    """``backends_mps.conf`` is a generated full-coverage conf like the vendor
    backends: it lists every routable op exactly once, routing only the
    MPSGraph-kernel cohort to ``mps`` and the rest to ``none``.  See
    docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md.
    """
    routes = _parse_conf(CONF_DIR / "backends_mps.conf")
    assert routes, "backends_mps.conf lists no routes"
    assert set(routes.values()) <= {"mps", "none"}, (
        "backends_mps.conf may only route ops to 'mps' or 'none'"
    )
    # One op per MPSGraph category the codegen emits (unary, binary,
    # binary_scalar, reduce, gemm, softmax, pow_scalar). A category whose entry
    # silently dropped would leave its ops on cpu_fallback and be invisible in a
    # count-only check. Both mm and bmm are listed because the gemm category
    # emits one kernel per op: a rank guard that reached one of them and not the
    # other is exactly what a single representative case would miss. ``neg`` is
    # the second op of the ``unary`` category -- the same template, so it is the
    # category's registration that a representative would otherwise not cover.
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
    ):
        assert routes.get(op) == "mps", f"the {op} kernel must be routed to mps"


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
    overlap the native set is the native-minus-FlagGems difference; every other
    op the conf lists is ``none``. A conf that disagrees with the registrations
    is caught by test_vendor_routes_are_registered / _accelerated_routes_are_
    registered / test_registered_ops_are_never_left_at_none above.
    """
    routes = _parse_conf(CONF_DIR / "backends_iluvatar.conf")
    flaggems = _registered_ops(
        ATEN_DIR / "backends/iluvatar/generated/iluvatar_flaggems_register.inc"
    )
    native = _registered_ops(
        ATEN_DIR / "backends/iluvatar/generated/iluvatar_register.inc"
    )
    iluvatar = native - flaggems
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
        "ascend": {"flaggems": 221, "none": 1659, "ascend": 157},
        "bpu": {},
        # 24 ops moved from flaggems to cuda: the conf writer now excludes
        # flaggems_recursive_fallback | flaggems_runtime_broken, which it could
        # not see before the sets were moved above the emitter. On a boxing-off
        # wheel -- the default -- GetFn(kCuda) degrades to the FlagGems Python
        # path, so this is a no-op there; on a FLAGOS_BUILD_BOXING=ON wheel the
        # ops now reach their boxing kernel instead of the FlagGems route.
        "cuda": {"flaggems": 392, "cuda": 1642},
        "dcu": {"flaggems": 454, "cuda": 1583},
        # embedding and embedding.out gained a native topsaten kernel and now
        # route to gcu instead of falling through to none; see
        # NATIVE_TRITON_GAPS in scripts/codegen/gen_vendor_confs.py.
        #
        # cat is the exception to the cat-flaggems reroute: the FlagGems route
        # does not compile on this vendor, so NATIVE_TRITON_GAPS["gcu"] keeps it
        # on the gcu kernel. See the entry in scripts/codegen/gen_vendor_confs.py.
        #
        # rsub.Scalar / rsub.Tensor moved flaggems -> gcu: FlagGems' rsub Triton
        # kernel is rejected by the Enflame front end (64-bit data type not
        # supported on GCU300), so they are served by the native topsaten kernel
        # instead. See NATIVE_TRITON_GAPS["gcu"].
        "gcu": {"none": 1603, "gcu": 183, "flaggems": 251},
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
        "metax": {"flaggems": 594, "cuda": 1433, "flaggems_cpp": 10},
        # cat moved to flaggems via the tensorlist_dim codegen category (#498).
        "musa": {"flaggems": 466, "none": 1518, "musa": 53},
        "ppu": {"flaggems": 592, "cuda": 1445},
        # Iluvatar adds a native, non-boxing vendor slot on top of FlagGems
        # (scripts/codegen/codegen_iluvatar.py). Its counts are DERIVED, not
        # pinned: the native cohort is a moving measurement -- the first landing
        # is _convolution, convolution, native_batch_norm, the ops that route
        # `none` today and have an ixdnn/ixblas kernel, and it grows as coverage
        # is measured. Ops FlagGems already registers (mm/bmm/addmm, the
        # _native_batch_norm_legit* variants) stay `flaggems` and are kept out of
        # the native .inc, since a second m.impl in the same PrivateUse1 body is a
        # duplicate-registration error. See _derived_iluvatar_counts().
        "iluvatar": _derived_iluvatar_counts(),
        # MPS is a native vendor backend at stage 3: the codegen'd
        # mps_register.inc claims the MPSGraph kernels the ``unary``, ``binary``,
        # ``binary_scalar``, ``reduce``, ``gemm``, ``softmax`` and ``pow_scalar``
        # categories generate (14 ops so far -- stage 3 added ``mm``/``bmm`` and
        # then ``_softmax``, ``neg`` and ``pow.Tensor_Scalar``), so the conf
        # routes those to ``mps`` and every other op to ``none``. The
        # ``softmax`` category is registered under ``_softmax`` alone because
        # ``softmax.int`` is CompositeImplicitAutograd and never reaches a
        # backend key; see scripts/codegen/codegen_mps.py.
        "mps": {"mps": 14, "none": 2023},
        "tsingmicro": {"flaggems": 51, "flagos_python": 2, "tsingmicro": 14},
    }
    for platform, exp in expected.items():
        routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
        counts = dict(Counter(routes.values()))
        assert counts == exp, (
            f"{platform} route counts drifted: expected {exp}, got {counts}. "
            "If an op moved routes deliberately, update this snapshot."
        )
