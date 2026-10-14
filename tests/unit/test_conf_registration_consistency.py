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
    # Iluvatar is FlagGems-primary with no vendor kernel library, so its ONLY
    # registration is the FlagGems dispatcher list -- there is no native .inc.
    # register.cc has an arm of its own precisely so it does not reach the
    # generated/register.inc fallback below, which would claim every op and turn
    # its `none` entries into "backend not registered" instead of cpu_fallback.
    "iluvatar": ["backends/iluvatar/generated/iluvatar_flaggems_register.inc"],
    # Boxing platforms (cuda, dcu, metax, ppu) and tsingmicro fall through to
    # generated/register.inc (register.cc:610-613), which claims every generated
    # op.
    "cuda": ["generated/register.inc"],
    "dcu": ["generated/register.inc"],
    "metax": ["generated/register.inc"],
    "ppu": ["generated/register.inc"],
    "tsingmicro": ["generated/register.inc"],
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
@pytest.mark.parametrize("platform", ["musa", "gcu", "ascend", "kunlunxin", "iluvatar"])
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
@pytest.mark.parametrize("platform", ["musa", "gcu", "ascend", "kunlunxin", "iluvatar"])
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
@pytest.mark.parametrize("platform", ["musa", "gcu", "ascend", "kunlunxin", "iluvatar"])
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


@pytest.mark.anyplatform
def test_conf_route_counts_are_stable():
    """Snapshot of the route counts per conf.  A change here means an op
    moved between routes: update this test deliberately, not silently.
    """
    expected = {
        # mm and mm.out moved from the aclnn route to flaggems; see
        # NATIVE_TRITON_GAPS in scripts/codegen/gen_vendor_confs.py.
        "ascend": {"flaggems": 224, "none": 1662, "ascend": 151},
        "bpu": {},
        "cuda": {"flaggems": 416, "cuda": 1618},
        "dcu": {"flaggems": 454, "cuda": 1583},
        # embedding and embedding.out gained a native topsaten kernel and now
        # route to gcu instead of falling through to none; see
        # NATIVE_TRITON_GAPS in scripts/codegen/gen_vendor_confs.py.
        #
        # cat is the exception to the cat-flaggems reroute: the FlagGems route
        # does not compile on this vendor, so NATIVE_TRITON_GAPS["gcu"] keeps it
        # on the gcu kernel. See the entry in scripts/codegen/gen_vendor_confs.py.
        "gcu": {"none": 1603, "gcu": 181, "flaggems": 253},
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
        # Iluvatar boxes nothing: every covered op is on the FlagGems Python
        # path, and the rest fall to cpu_fallback. 603 of the shared FlagGems
        # ceiling's 640 ops route here; the 37-op difference is the measured
        # BI-V150 overload survey's defect set (NATIVE_TRITON_GAPS['iluvatar']),
        # not drift -- re-running the survey and moving ops in or out of that
        # set is what changes this number.
        "iluvatar": {"flaggems": 603, "none": 1434},
        "tsingmicro": {"flaggems": 51, "flagos_python": 2},
    }
    for platform, exp in expected.items():
        routes = _parse_conf(CONF_DIR / f"backends_{platform}.conf")
        counts = dict(Counter(routes.values()))
        assert counts == exp, (
            f"{platform} route counts drifted: expected {exp}, got {counts}. "
            "If an op moved routes deliberately, update this snapshot."
        )
