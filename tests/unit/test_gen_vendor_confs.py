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

"""Unit coverage for scripts/codegen/gen_vendor_confs.py.

The generator turns shared FlagGems coverage sources plus each platform's own
conf into one full-coverage conf per platform, where every op torch_fl can route
is listed exactly once as `flaggems_cpp`, `flaggems`, `tileops`,
`<vendor>`/`cuda` or `none`. Stating `none` explicitly is the point: "absent
from the file" and "known to be unsupported" used to look identical, so operator
support could not be counted and sparse confs rotted silently against a growing
codegen.

The load-bearing property is that **a conf cannot claim more than the platform
registers**. FlagGems coverage is measured on CUDA, so it is a ceiling, not a
per-platform routing set: the FlagGems wrapper is reached through the op's
PrivateUse1 registration, so routing an unregistered op to `flaggems` names a
kernel no call arrives at while the file reports it as covered. Vendor
registration sets are therefore read from the generated `*_register.inc` files
csrc/aten/register.cc includes -- the same list the compiler sees --
and test_vendor_routing_never_exceeds_registration pins the containment.

That also decides which platforms get a generated conf at all. `none` boxes to
cpu_fallback only where registration *skips* the op; metax, tsingmicro and dcu
register the full generated list, so `none` there would raise instead. They are
either boxing platforms (fallback `cuda`, no `none`) or hand-written.

For boxing platforms the generator still reads the file it rewrites, so two
measured sets have to survive the round trip:

  * the triton gap set, spelled `cuda` on an op FlagGems covers (MetaX
    Xnack/ATU faults, DCU hcu VMFault);
  * the measured FlagGems-C++ subset, which is NOT the shared fg_cpp set --
    backends_metax_flaggems_cpp.conf verifies 17 of 18 on-device and
    deliberately keeps `mm` boxed.

Both are sets the generator could plausibly re-derive from shared sources, and
doing so would silently reroute working kernels. These tests pin them.

Run: pytest tests/unit/test_gen_vendor_confs.py
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "codegen" / "gen_vendor_confs.py"


def _load():
    """Import the generator by path -- scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("gen_vendor_confs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = pytest.importorskip("importlib") and _load()
CONF_DIR = REPO_ROOT / "torch_fl" / "configs"


# ---------------------------------------------------------------------------
# route(): the four-key priority
# ---------------------------------------------------------------------------


def test_route_priority_is_cpp_then_python_then_tileops_then_vendor_then_none():
    fg_cpp, fg_py, native = {"a"}, {"b"}, {"c", "t"}
    tileops = {"t"}
    reg = {"a", "b", "c", "d", "t"}
    assert g.route("a", "musa", fg_cpp, fg_py, native, reg, tileops) == "flaggems_cpp"
    assert g.route("b", "musa", fg_cpp, fg_py, native, reg, tileops) == "flaggems"
    # tileops outranks the vendor kernel, and still records that it exists.
    assert (
        g.route("t", "musa", fg_cpp, fg_py, native, reg, tileops) == "tileops  # musa"
    )
    assert g.route("c", "musa", fg_cpp, fg_py, native, reg, tileops) == "musa"
    assert g.route("d", "musa", fg_cpp, fg_py, native, reg, tileops) == "none"


def test_route_loses_tileops_to_flaggems_and_withholds_it_when_unregistered():
    """tileops sits below both FlagGems keys and above the vendor, and is subject
    to the same registration ceiling as every other accelerated route."""
    both = {"t"}
    assert g.route("t", "musa", both, set(), set(), both, both) == "flaggems_cpp"
    assert g.route("t", "musa", set(), both, set(), both, both) == "flaggems"
    # unregistered: no route may be claimed, tileops included
    assert g.route("t", "musa", set(), set(), set(), set(), both) == "none"


def test_route_omits_tileops_when_the_platform_cannot_compile_the_slot():
    """TILEOPS_PLATFORMS gates the key. setup.py forces TILEOPS_KERNEL=OFF for
    every non-cuda accelerator, where kTileOps degrades to an equally empty
    cuda_fn_ -- so an ungated key would route a real op at nothing."""
    assert not g.TILEOPS_PLATFORMS
    for _, (_, routes, _) in g.build_all(CONF_DIR).items():
        keys = {v.split("#", 1)[0].strip() for v in routes.values()}
        assert "tileops" not in keys


def test_route_withholds_flaggems_from_unregistered_ops():
    """The whole point of the rework: FlagGems coverage is a CUDA-measured
    ceiling. An op this platform never claims on PrivateUse1 cannot reach the
    FlagGems wrapper, so claiming `flaggems` would report coverage for a call
    path that boxes to CPU."""
    fg_cpp, fg_py = {"a"}, {"b"}
    assert g.route("a", "musa", fg_cpp, fg_py, set(), set()) == "none"
    assert g.route("b", "musa", fg_cpp, fg_py, set(), set()) == "none"
    # ...and it is the registration set that decides, not the coverage set.
    assert g.route("b", "musa", fg_cpp, fg_py, set(), {"b"}) == "flaggems"


def test_route_annotates_vendor_kernels_that_lose_to_flaggems():
    """The annotation is what tells a reader (and ALL_USE_VENDOR) that a kernel
    exists behind an op FlagGems currently wins."""
    both = {"mm"}
    assert g.route("mm", "musa", both, set(), both, both) == "flaggems_cpp  # musa"
    assert g.route("mm", "musa", set(), both, both, both) == "flaggems  # musa"


def test_route_boxing_never_emits_none():
    """Boxing reuses CUDA wholesale, so every op has an impl."""
    for op in ("a", "b", "c"):
        assert g.route_boxing(op, {"a"}, {"b"}, set()) != "none"
    assert g.route_boxing("c", {"a"}, {"b"}, set()) == g.BOXING_FALLBACK


def test_route_boxing_gap_beats_flaggems_coverage():
    """A measured triton gap outranks FlagGems coverage for that platform."""
    assert g.route_boxing("x", {"x"}, set(), {"x"}) == g.BOXING_FALLBACK
    assert g.route_boxing("y", set(), {"y"}, {"y"}) == g.BOXING_FALLBACK


def test_route_boxing_gap_also_beats_tileops():
    """Same reason: the gap set is measured on hardware, so it wins over any
    derived candidate -- and tileops is not a way around a broken op."""
    assert g.route_boxing("z", set(), set(), {"z"}, {"z"}) == g.BOXING_FALLBACK
    assert g.route_boxing("z", set(), set(), set(), {"z"}) == "tileops"


# ---------------------------------------------------------------------------
# Recovery helpers: both spellings, so the round trip is lossless
# ---------------------------------------------------------------------------


def test_registered_impls_reads_the_generated_registrations(tmp_path):
    inc = tmp_path / "x_register.inc"
    inc.write_text(
        "// comment mentioning m.impl in prose\n"
        'm.impl("abs", TORCH_FN(wrapper_abs));\n'
        'm.impl( "add.Tensor" , TORCH_FN(wrapper_add));\n'
    )
    assert g.registered_impls(inc) == {"abs", "add.Tensor"}


def test_registered_impls_is_empty_for_a_missing_inc(tmp_path):
    """A vendor built without its codegen artifacts registers nothing, which must
    read as no coverage rather than crash the generator."""
    assert g.registered_impls(tmp_path / "absent.inc") == set()


def test_vendor_registered_ops_includes_the_flaggems_inc():
    """MUSA claims a second file whose wrappers route to the FlagGems Python
    slot. Those ops are registered without being native kernels, so the
    registered set must be strictly wider than the native set."""
    native = g.vendor_native_ops("musa")
    registered = g.vendor_registered_ops("musa")
    assert native < registered


def test_musa_registers_all_flaggems_ops_except_known_failures():
    """MUSA registers every FlagGems Python op, then routes it to flaggems except
    for the ops in NATIVE_TRITON_GAPS['musa'].

    Strategy: register everything → run CI → move failures to NATIVE_TRITON_GAPS.
    This maximizes coverage while maintaining correctness via the fallback list.

    Known failures (as of 2026-09-15 on MTT S5000, FlagGems 4d9c34775 +
    flagtree 0.6.2a3+mthreads3.6); the full diagnosis for each lives on the set
    itself in scripts/codegen/gen_vendor_confs.py:
      - _conj: flag_gems materializes the conjugation that ATen keeps as a lazy
        view, so test_math_bits_contract's `is_conj()` contract breaks. No mudnn
        kernel either, so this one routes to `none` and ATen's composite runs.
      - add/sub/div/mul.Tensor and their in-place forms add_/sub_/div_/mul_.Tensor:
        a Python-float operand arrives as a float64 0-dim tensor (ATen
        wrapped-number boxing) and flag_gems' pointwise promotion does not honour
        is_wrapped_number, so the kernel promotes to fp64 and mthreads' LLVM
        lowering has no double overload for llvm.musa.float2bfloat16. bf16 only.
        The .Scalar forms are unaffected, but ATen boxes them (and
        _foreach_add_, and so AdamW's foreach path) onto the .Tensor overload, so
        the in-place entries are what actually keep those callers off the failing
        kernel. mul_.Tensor is listed even though mul.Tensor is already native.
      - div.Tensor_mode/div_.Tensor_mode/floor_divide/floor_divide_.Tensor
        (issue #266): on integer inputs flag_gems loses the trailing store, so
        every non-power-of-two numel returns a stale last element. Both
        rounding-mode overloads and the base/in-place floor_divide spellings hit
        it, and the Scalar forms decompose onto them, so `a // b`, `a // 2`,
        torch.floor_divide(a, b) and a.div_(b, rounding_mode='floor') all do.
      - sort/sort.stable: flag_gems' radix sort casts its histogram to uint32
        internally and mudnn's CAST has no UInt16/32/64 case, so the cast raises
        before the sort runs. mudnn's own sort is correct here; argsort and
        msort decompose onto sort.

    index_add/index_add_ and randn/randn_like were removed from this set on
    2026-09-15: re-measured on MTT S5000 against both the current FlagGems and
    the CI pin, neither recorded signature reproduces, so both pairs route back
    to flaggems (the index pair was previously `none`, not `musa`). See the set
    itself in scripts/codegen/gen_vendor_confs.py for the per-op evidence and
    docs/reference/operator-support.md for the report.

    Ops in NATIVE_TRITON_GAPS are NOT registered (to avoid "backend not registered"
    errors when they route to none/musa).
    """
    routes = g.build_all(CONF_DIR)["musa"][1]
    native = g.vendor_native_ops("musa")
    flaggems_py = g.flaggems_python_ops()
    registered = g.vendor_registered_ops("musa")
    gaps = g.NATIVE_TRITON_GAPS.get("musa", set())

    # The shared ceiling is discovered from the installed flag_gems package, so it
    # is wider than any one vendor's measured cohort. Ops it gained after MUSA's
    # last FlagGems run are held on the mudnn kernel (FLAGGEMS_PENDING_NATIVE_OPS,
    # scripts/gen_vendor_confs.py) and are neither registered nor routed here, so
    # they are out of scope for this test. The cohort below is derived rather than
    # pinned: widening either coverage set moves it without an edit here.
    pending = (
        g.FLAGGEMS_PENDING_NATIVE_OPS
        if "musa" in g.FLAGGEMS_PENDING_NATIVE_VENDORS
        else set()
    )
    cohort = flaggems_py - pending

    # All FlagGems ops in the measured cohort, except those in gaps, are registered
    expected_registered_from_flaggems = cohort - gaps
    missing = sorted(expected_registered_from_flaggems - registered)
    assert not missing, f"FlagGems ops not registered: {missing}"

    # Count flaggems routes
    flaggems_routes = {
        op for op, key in routes.items() if key.split("#")[0].strip() == "flaggems"
    }

    # The cohort minus the gaps above is registered and routed to flaggems.
    short = sorted(expected_registered_from_flaggems - flaggems_routes)
    extra = sorted(flaggems_routes - expected_registered_from_flaggems)
    assert not short and not extra, (
        f"cohort not routed to flaggems: {short}; "
        f"routed to flaggems outside the cohort: {extra}"
    )
    assert gaps == {
        "_conj",
        "add.Tensor",
        "add_.Tensor",
        "div.Tensor",
        "div.Tensor_mode",
        "div_.Tensor",
        "div_.Tensor_mode",
        "floor_divide",
        "floor_divide_.Tensor",
        "mul_.Tensor",
        "sort",
        "sort.stable",
        "sub.Tensor",
        "sub_.Tensor",
    }

    # Known failures should NOT be routed to flaggems or be registered
    for op in gaps:
        assert op not in flaggems_routes, (
            f"{op} in NATIVE_TRITON_GAPS but routed to flaggems"
        )
        # Ops in gaps that mudnn doesn't have will be unregistered
        if op not in native:
            assert op not in registered, (
                f"{op} in gaps without native impl should be unregistered"
            )

    # Coverage: every FlagGems op outside the gaps reaches the flaggems key, and
    # mudnn still backs the ops FlagGems has no kernel for. The absolute
    # accelerated total is reported in the PR; what is pinned here is the
    # invariant, so that adding an op to either coverage set never needs an edit.
    mudnn_only = native - flaggems_py
    assert mudnn_only, "mudnn must still back the ops FlagGems does not cover"
    unrouted = sorted(op for op in cohort - gaps if routes.get(op) == "none")
    assert not unrouted, f"FlagGems ops left with no route: {unrouted}"


def test_ascend_matmul_is_native_though_absent_from_its_inc():
    """matmul is claimed straight from register.cc under `#if defined(USE_ASCEND)`
    so the call hits fused aclnnMatmul instead of decomposing. WrapperMatmul
    reads this conf to decide, so losing the entry makes the kernel
    unreachable."""
    assert {"matmul", "matmul_backward"} <= g.vendor_native_ops("ascend")
    routes = g.build_all(CONF_DIR)["ascend"][1]
    assert routes["matmul"] == "ascend"


def test_boxing_triton_gaps_only_counts_ops_flaggems_covers(tmp_path):
    (tmp_path / "b.conf").write_text(
        "real_gap = cuda\n"  # FlagGems covers it, pinned to cuda anyway
        "never_covered = cuda\n"  # just uncovered, not a gap
        "routed = flaggems\n"
    )
    gaps = g.boxing_triton_gaps(tmp_path, "b.conf", {"real_gap"}, set())
    assert gaps == {"real_gap"}


def test_boxing_cpp_ops_recovers_measured_subset(tmp_path):
    """Returns METAX_CPP_MEASURED ∩ fg_cpp for backends_metax.conf; empty for all others."""
    # Superset fg_cpp — only measured ops should survive the intersection.
    fg_cpp = g.METAX_CPP_MEASURED | {"extra_op_not_measured"}
    result = g.boxing_cpp_ops(tmp_path, "backends_metax.conf", fg_cpp)
    assert result == g.METAX_CPP_MEASURED
    # Any other filename returns empty regardless of fg_cpp contents.
    assert g.boxing_cpp_ops(tmp_path, "backends_other.conf", fg_cpp) == set()


# ---------------------------------------------------------------------------
# The shipped confs
# ---------------------------------------------------------------------------


def test_shipped_confs_are_up_to_date():
    """Same contract as `gen_vendor_confs.py --check` in CI."""
    built = g.build_all(CONF_DIR)
    stale = [
        f"backends_{p}.conf"
        for p, (text, _, _) in built.items()
        if (CONF_DIR / f"backends_{p}.conf").read_text() != text
    ]
    assert not stale, f"stale: {stale}; run scripts/codegen/gen_vendor_confs.py"


def test_generation_is_idempotent():
    """Regenerating from already-generated confs reproduces them exactly.

    This is the round trip: build_all() re-reads the confs it wrote, so any
    measured set it fails to recover shows up here as a second-run diff.
    """
    first = {p: t for p, (t, _, _) in g.build_all(CONF_DIR).items()}
    second = {p: t for p, (t, _, _) in g.build_all(CONF_DIR).items()}
    assert first == second


@pytest.mark.parametrize("vendor", sorted(g.VENDORS))
def test_every_vendor_conf_covers_the_whole_op_list(vendor):
    built = g.build_all(CONF_DIR)
    routes = built[vendor][1]
    op_count = len(next(iter(built.values()))[1])
    assert len(routes) == op_count
    allowed = {"flaggems_cpp", "flaggems", "tileops", vendor, "none"}
    for op, value in routes.items():
        key = value.split("#", 1)[0].strip()
        assert key in allowed, f"{vendor}: {op} routed to unexpected {key!r}"


@pytest.mark.parametrize("platform", sorted(g.BOXING_PLATFORMS))
def test_boxing_confs_are_fully_covered(platform):
    """No `none` on a CUDA-compatible platform -- boxing always has a kernel."""
    routes = g.build_all(CONF_DIR)[platform][1]
    assert routes, f"{platform} produced no routes"
    for op, value in routes.items():
        key = value.split("#", 1)[0].strip()
        assert key != "none", f"{platform}: {op} routed to none"
        assert key in {"flaggems_cpp", "flaggems", "tileops", g.BOXING_FALLBACK}


def test_every_platform_covers_the_same_op_set():
    """Coverage is only comparable across files if the op set is identical.

    build_all() widens the CUDA-derived op list by the union of vendor-only ops
    (metax's _to_copy / _local_scalar_dense, ascend's matmul) precisely so this
    holds; a per-platform op list would make the counts in each header
    incommensurable.
    """
    built = g.build_all(CONF_DIR)
    op_sets = {platform: frozenset(v[1]) for platform, v in built.items()}
    assert len(set(op_sets.values())) == 1, {
        platform: len(ops) for platform, ops in op_sets.items()
    }


def test_flaggems_cpp_only_appears_where_the_slot_is_compiled_in():
    """`flaggems_cpp` is Backend::kFlagGemsCpp, registered in flaggems_cpp_kernels.cc
    behind `#ifdef FLAGOS_FLAGGEMS_CPP` -- which csrc/CMakeLists.txt defines only
    for FLAGGEMS_KERNEL=ON. CMakeLists.txt force-sets that OFF for ascend, dcu,
    musa, bpu, tsingmicro and a non-boxing metax build. For those, the slot is
    empty and Dispatcher::GetFn degrades to the boxing kernel instead of raising.

    backends_metax.conf is the only generated conf that routes any ops to the C++
    path. The vendor confs (musa/gcu/ascend) omit the key entirely: every C++ op
    is also a Python op, so they route it to `flaggems` and reach the same kernel
    through python_op_caller, losing only the GIL-free entry point.
    """
    for platform, (_, routes, _) in g.build_all(CONF_DIR).items():
        uses_cpp = {
            op
            for op, v in routes.items()
            if v.split("#", 1)[0].strip() == "flaggems_cpp"
        }
        if platform in g.FLAGGEMS_CPP_PLATFORMS:
            assert uses_cpp, f"{platform} should route to the C++ runtime"
        else:
            assert not uses_cpp, (
                f"{platform} routes {sorted(uses_cpp)[:5]} to flaggems_cpp, but that "
                "build has no kFlagGemsCpp kernel registered"
            )


def test_flaggems_cpp_set_is_a_subset_of_the_python_set():
    """Why withholding `flaggems_cpp` costs no coverage: every C++ op is also a
    Python op, so it falls to `flaggems` and reaches the same FlagGems kernel --
    only the GIL-free C++ entry point is given up, not the operator."""
    cpp, py = g.flaggems_cpp_ops(CONF_DIR), g.flaggems_python_ops(CONF_DIR)
    assert cpp, "expected a non-empty C++ coverage set"
    assert cpp <= py, f"C++-only ops would lose coverage: {sorted(cpp - py)}"


def test_tileops_set_survived_the_conf_deletion():
    """backends_tileops.conf carried this set as 60 `tileops` lines in a 2060-line
    file that was otherwise a copy of backends_cuda.conf. It is now
    TILEOPS_OPS in scripts/codegen/backend_coverage.py, regenerated in place by
    scripts/codegen/codegen_tileops.py. The count is pinned because losing entries here
    silently shrinks what FLAGOS_USE_TILEOPS=1 can repin."""
    ops = g.tileops_ops()
    assert len(ops) == 60, f"expected 60 tileops ops, got {len(ops)}"
    assert {"abs", "add.Tensor", "_softmax"} <= ops


@pytest.mark.parametrize("vendor", sorted(g.VENDORS))
def test_vendor_routing_never_exceeds_registration(vendor):
    """Every accelerated route must be an op the platform actually claims.

    This is the invariant the earlier generator broke: it used the CUDA-measured
    FlagGems set as every vendor's coverage, so backends_musa.conf routed 482 ops
    to `flaggems` against 158 registered -- 324 claims for calls that box to CPU.
    Equality (not just containment) also pins the converse: a registered op is
    never left at `none`, which would send a real kernel to cpu_fallback.
    """
    routes = g.build_all(CONF_DIR)[vendor][1]
    routed = {op for op, v in routes.items() if v.split("#", 1)[0].strip() != "none"}
    assert routed == g.vendor_registered_ops(vendor)


def test_only_registration_subset_platforms_get_a_generated_vendor_conf():
    """`none` boxes to cpu_fallback only where registration skips the op.

    metax, tsingmicro and dcu fall through to the `#else` branch in
    csrc/aten/register.cc and claim the full generated list, so a `none` entry
    there would reach the dispatcher and raise on an empty slot instead of
    falling back. They must not be generated as native-kernel vendors: metax and
    dcu are covered by their boxing confs, tsingmicro stays hand-written.
    """
    assert set(g.VENDORS) == {"musa", "gcu", "ascend"}
    assert "metax" not in g.VENDORS
    assert "tsingmicro" not in g.VENDORS
    # backends_metax.conf is now AUTO-GENERATED (boxing conf), so skip that check
    text = (CONF_DIR / "backends_tsingmicro.conf").read_text()
    assert "AUTO-GENERATED" not in text, (
        "backends_tsingmicro.conf must stay hand-written"
    )
    assert " = none" not in text, "tsingmicro registers every op; none would raise"


def test_metax_conf_keeps_mm_boxed():
    """A measured exception, not a derivable one: mm is in the shared fg_cpp set
    but only 17 of 18 verified on-device, mm/mm.out failing on shared-memory size.
    METAX_CPP_VERIFIED encodes that measurement, so mm must stay on boxing."""
    routes = g.build_all(CONF_DIR)["metax"][1]
    assert routes["mm"].split("#", 1)[0].strip() == g.BOXING_FALLBACK
    # Must exist in METAX_CPP_MEASURED (test_metax_conf_keeps_mm_boxed ensures it
    # stays pinned to "mm not in the measured set")
    assert "mm" not in g.METAX_CPP_MEASURED


@pytest.mark.parametrize("platform", sorted(g.BOXING_PLATFORMS))
def test_boxing_confs_have_no_none_entries(platform):
    """A CUDA-compatible platform claims the full generated op list, so every op
    has a boxing kernel behind it and `none` is never the right answer -- it would
    reach the dispatcher and raise on an empty slot instead of boxing."""
    routes = g.build_all(CONF_DIR)[platform][1]
    none_ops = [op for op, v in routes.items() if v.split("#", 1)[0].strip() == "none"]
    assert not none_ops, f"{platform} routes {none_ops[:5]} to none"
