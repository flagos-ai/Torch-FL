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

"""Vendor-routing unit tests for ProcessGroupFlagOS.

Pure-logic tests for the GEMS_VENDOR -> (flagcx devName, view, native backend)
routing in torch_fl.comm.process_group. No GPU / no real backend needed: we
import the module directly (it has no import-time torch_fl._C dependency) and
drive _build_inner with fakes to assert the decision table, not the transport.

Run: pytest tests/unit/test_vendor_routing.py
"""

import sys
import types
import warnings

import pytest

# Import the module WITHOUT importing torch_fl (which preloads cuda). The module
# only needs torch + torch.distributed at import time.
pg = pytest.importorskip("torch_fl.comm.process_group")


# ---------------------------------------------------------------------------
# NCCL extension loading
# ---------------------------------------------------------------------------


def _new_nccl_process_group(monkeypatch):
    monkeypatch.setattr(pg.torch.distributed, "ProcessGroupNCCL", None, raising=False)
    return pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)


def test_nccl_extension_none_sentinel_falls_back_cleanly(monkeypatch):
    import torch_fl.comm._nccl_ext as nccl_ext

    monkeypatch.setattr(nccl_ext, "_flagos_nccl", None)
    monkeypatch.setitem(sys.modules, "_flagos_nccl", None)

    obj = _new_nccl_process_group(monkeypatch)
    assert obj._try_build_nccl(None, 0, 1, None) is False


def test_nccl_extension_none_sentinel_uses_loose_module(monkeypatch):
    import torch_fl.comm._nccl_ext as nccl_ext

    sentinel = object()
    seen = {}

    def make_nccl_backend(store, rank, world_size, timeout_ms, high_priority):
        seen["args"] = (store, rank, world_size, timeout_ms, high_priority)
        return sentinel

    loose_extension = types.ModuleType("_flagos_nccl")
    loose_extension.make_nccl_backend = make_nccl_backend
    monkeypatch.setattr(nccl_ext, "_flagos_nccl", None)
    monkeypatch.setitem(sys.modules, "_flagos_nccl", loose_extension)

    obj = _new_nccl_process_group(monkeypatch)
    assert obj._try_build_nccl("store", 2, 8, 123) is True
    assert obj._inner is sentinel
    assert seen["args"] == ("store", 2, 8, 123, False)


def test_nccl_extension_prefers_packaged_module(monkeypatch):
    import torch_fl.comm._nccl_ext as nccl_ext

    sentinel = object()
    seen = {}

    def make_nccl_backend(store, rank, world_size, timeout_ms, high_priority):
        seen["args"] = (store, rank, world_size, timeout_ms, high_priority)
        return sentinel

    packaged_extension = types.SimpleNamespace(make_nccl_backend=make_nccl_backend)
    loose_extension = types.ModuleType("_flagos_nccl")

    def fail_if_called(*args):
        raise AssertionError("loose extension must not be used")

    loose_extension.make_nccl_backend = fail_if_called
    monkeypatch.setattr(nccl_ext, "_flagos_nccl", packaged_extension)
    monkeypatch.setitem(sys.modules, "_flagos_nccl", loose_extension)

    obj = _new_nccl_process_group(monkeypatch)
    assert obj._try_build_nccl("store", 1, 4, None) is True
    assert obj._inner is sentinel
    assert seen["args"] == ("store", 1, 4, 0, False)


# ---------------------------------------------------------------------------
# Profile table
# ---------------------------------------------------------------------------


def test_all_vendors_have_consistent_profiles():
    for name, prof in pg._VENDOR_PROFILES.items():
        assert prof.flagcx_dev, f"{name} missing flagcx_dev"
        # cuda-ABI vendors must expose the zero-copy cuda view + NCCL fallback;
        # non-cuda vendors must NOT claim the cuda view. Enflame accepts
        # PrivateUse1 directly; MUSA uses an explicit identity view.
        if prof.flagcx_dev == "cuda":
            assert prof.view == "_flagos_to_cuda_view", name
            assert prof.native == "_try_build_nccl", name
        else:
            assert prof.view in (None, "_flagos_identity_view"), (
                f"{name} is not a cuda alias but claims view {prof.view!r}"
            )


def test_unknown_vendor_falls_back_to_default_profile():
    with pytest.warns(UserWarning, match="unknown GEMS_VENDOR"):
        prof = pg._get_profile("totally-made-up")
    assert prof is pg._VENDOR_PROFILES[pg._DEFAULT_VENDOR]


def test_known_cuda_vendors():
    for v in ("nvidia", "metax", "iluvatar", "kunlunxin", "du", "thead", "hygon"):
        prof = pg._get_profile(v)
        assert prof.flagcx_dev == "cuda"
        assert prof.view == "_flagos_to_cuda_view"


def test_thead_ppu_routes_without_warning():
    """PPU reports GEMS_VENDOR=thead from FlagGems' own detection (PPU_SDK set),
    while torch_fl sets nvidia. Both must resolve to the same CUDA-ABI profile,
    and thead must not trip the unknown-vendor warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        prof = pg._get_profile("thead")
    assert prof.flagcx_dev == "cuda"
    assert prof.native == "_try_build_nccl"
    assert prof.view == pg._VENDOR_PROFILES["nvidia"].view


def test_hygon_dcu_routes_without_warning():
    """DCU (DTK) sets GEMS_VENDOR=hygon. torch there is a hipified CUDA build, so
    flagos is a cuda alias and ProcessGroupNCCL is RCCL: the CUDA-ABI profile
    applies verbatim. Must not trip the unknown-vendor warning -- before the row
    existed, an unset GEMS_VENDOR on DCU landed on the ascend profile instead and
    init_process_group failed with "no suitable inner backend"."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        prof = pg._get_profile("hygon")
    assert prof.flagcx_dev == "cuda"
    assert prof.native == "_try_build_nccl"
    assert prof.view == pg._VENDOR_PROFILES["nvidia"].view


# ---------------------------------------------------------------------------
# Collective virtual coverage
# ---------------------------------------------------------------------------


def test_single_tensor_base_collectives_are_overridden():
    """_allgather_base / _reduce_scatter_base are separate virtuals from
    allgather / reduce_scatter. If they are not overridden, ProcessGroup's C++
    base tries to resolve a Backend for the tensor's device and raises
    "No backend type associated with device type flagos" -- which breaks
    dist.all_gather_into_tensor / reduce_scatter_tensor, i.e. the FSDP and ZeRO
    hot paths. Guard against silently dropping them again."""
    for name in ("_allgather_base", "_reduce_scatter_base"):
        assert name in pg.ProcessGroupFlagOS.__dict__, (
            f"{name} not overridden on ProcessGroupFlagOS; "
            f"dist.{'all_gather_into_tensor' if 'allgather' in name else 'reduce_scatter_tensor'}"
            f" will fail on flagos tensors"
        )


def test_base_collectives_delegate_with_converted_views(monkeypatch):
    """Both new virtuals must pass the flagos->cuda view through, not the raw
    privateuseone tensor."""

    class _FakeInner:
        def __init__(self):
            self.seen = {}

        def _allgather_base(self, out, inp, opts):
            self.seen["allgather"] = (out, inp)
            return "work-ag"

        def _reduce_scatter_base(self, out, inp, opts):
            self.seen["rs"] = (out, inp)
            return "work-rs"

    obj = pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)
    obj._inner = _FakeInner()
    # view_fn tags anything it converts so we can assert it was applied
    obj._view_fn = lambda t: ("viewed", t)

    class _FlagosTensor:
        device = types.SimpleNamespace(type="flagos")

    a, b = _FlagosTensor(), _FlagosTensor()
    assert obj._allgather_base(a, b) == "work-ag"
    assert obj._inner.seen["allgather"] == (("viewed", a), ("viewed", b))
    assert obj._reduce_scatter_base(a, b) == "work-rs"
    assert obj._inner.seen["rs"] == (("viewed", a), ("viewed", b))


# ---------------------------------------------------------------------------
# _build_inner routing (with fakes)
# ---------------------------------------------------------------------------


def _make(monkeypatch, vendor, *, flagcx_ok, native_ok, view_present=True):
    """Configure a fake ProcessGroupFlagOS and stub out its build helpers.

    __init__ (and thus the ProcessGroup C++ base ctor) is bypassed via __new__
    so we can drive _build_inner in isolation without a real store/rank/size.
    """
    obj = pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)
    calls = {"flagcx": False, "native": False}

    def fake_flagcx(self, store, rank, ws, timeout):
        calls["flagcx"] = True
        if flagcx_ok:
            self._inner = object()
        return flagcx_ok

    def fake_nccl(self, store, rank, ws, timeout):
        calls["native"] = True
        if native_ok:
            self._inner = object()
        return native_ok

    def fake_hccl(self, store, rank, ws, timeout):
        calls["native"] = True
        if native_ok:
            self._inner = object()
        return native_ok

    def fake_mccl(self, store, rank, ws, timeout):
        calls["native"] = True
        if native_ok:
            self._inner = object()
        return native_ok

    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_flagcx", fake_flagcx)
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_nccl", fake_nccl)
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_hccl", fake_hccl)
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_mccl", fake_mccl)

    # Fake torch_fl._C so _resolve_view finds (or misses) the view helper.
    # _resolve_view does `import torch_fl._C as _C`; when torch_fl is already
    # imported that resolves via the parent package attribute, so patch both the
    # sys.modules entry and the attribute on the (possibly real) torch_fl pkg.
    sentinel_view = object()  # unique marker so tests can assert identity
    fake_c = types.ModuleType("torch_fl._C")
    if view_present:
        fake_c._flagos_to_cuda_view = sentinel_view
        fake_c._flagos_identity_view = sentinel_view  # same marker for identity
    monkeypatch.setitem(sys.modules, "torch_fl._C", fake_c)
    torch_fl_pkg = sys.modules.get("torch_fl")
    if torch_fl_pkg is not None:
        monkeypatch.setattr(torch_fl_pkg, "_C", fake_c, raising=False)
    monkeypatch.setenv("GEMS_VENDOR", vendor)
    return obj, calls, sentinel_view


def test_flagcx_preferred_over_native(monkeypatch):
    obj, calls, sentinel = _make(monkeypatch, "nvidia", flagcx_ok=True, native_ok=True)
    view = obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and not calls["native"]
    assert view is sentinel  # cuda view resolved from the (faked) torch_fl._C


def test_native_used_when_flagcx_unavailable(monkeypatch):
    obj, calls, _ = _make(monkeypatch, "nvidia", flagcx_ok=False, native_ok=True)
    obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and calls["native"]
    assert obj._inner is not None


def test_no_backend_raises(monkeypatch):
    obj, _, _ = _make(monkeypatch, "nvidia", flagcx_ok=False, native_ok=False)
    with pytest.raises(RuntimeError, match="no suitable inner backend"):
        obj._build_inner(None, 0, 1, None)


def test_mthreads_aliases_musa_profile():
    assert pg._get_profile("mthreads") is pg._VENDOR_PROFILES["mthreads"]
    assert pg._VENDOR_PROFILES["mthreads"].flagcx_dev == "musa"
    assert pg._VENDOR_PROFILES["mthreads"].native == "_try_build_mccl"


def test_musa_flagcx_identity_view(monkeypatch):
    """MUSA uses _flagos_identity_view: FlagCX's MUSA adaptor receives
    privateuseone tensors as-is, no storage conversion needed."""
    obj, calls, _ = _make(
        monkeypatch, "musa", flagcx_ok=True, native_ok=False, view_present=True
    )
    # Fake the identity view helper in torch_fl._C
    identity_marker = object()
    fake_c = sys.modules["torch_fl._C"]
    fake_c._flagos_identity_view = identity_marker

    view_fn = obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and not calls["native"]
    assert view_fn is identity_marker  # identity view resolved


def test_musa_uses_mccl_native_when_flagcx_unavailable(monkeypatch):
    # musa has native=_try_build_mccl: if flagcx fails, MCCL is attempted.
    obj, calls, _ = _make(
        monkeypatch, "musa", flagcx_ok=False, native_ok=True, view_present=True
    )
    # Fake the identity view helper
    identity_marker = object()
    fake_c = sys.modules["torch_fl._C"]
    fake_c._flagos_identity_view = identity_marker

    view_fn = obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and calls["native"]  # flagcx tried first, mccl succeeded
    assert obj._inner is not None
    assert view_fn is identity_marker  # identity view for MCCL too


def test_musa_no_backend_raises(monkeypatch):
    # Neither FlagCX nor MCCL available -> must raise with clear error.
    obj, calls, _ = _make(
        monkeypatch, "musa", flagcx_ok=False, native_ok=False, view_present=True
    )
    with pytest.raises(RuntimeError, match="no suitable inner backend"):
        obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and calls["native"]  # both attempted


def test_musa_flagcx_ok_but_identity_view_missing_raises(monkeypatch):
    # musa profile claims _flagos_identity_view, but the C++ binding wasn't
    # compiled in -> must fail with a clear error pointing to the missing symbol.
    obj, calls, _ = _make(
        monkeypatch, "musa", flagcx_ok=True, native_ok=False, view_present=False
    )
    with pytest.raises(
        RuntimeError, match=r"torch_fl\._C\._flagos_identity_view not found"
    ):
        obj._build_inner(None, 0, 1, None)


def test_flagcx_plain_signature_is_accepted(monkeypatch):
    """FlagCX only compiles the extended_api creator for the NVIDIA and MetaX
    adaptors; ppu/du/kunlunxin/ascend/enflame export the plain
    (store, rank, size, timeout) form. Calling the extended form on those
    raises TypeError ("incompatible function arguments"), which must be retried
    as the plain form -- otherwise FlagCX silently degrades to NCCL."""
    sentinel = object()
    seen = {}

    def creator(*args):
        # extended form is (opts, extra) or (opts,); reject like pybind11 does
        if len(args) < 3:
            raise TypeError("createFlagcxBackend(): incompatible function arguments")
        seen["args"] = args
        return sentinel

    fake_flagcx = types.ModuleType("flagcx")
    fake_flagcx.createFlagcxBackend = creator
    monkeypatch.setitem(sys.modules, "flagcx", fake_flagcx)

    obj = pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must not warn on the expected retry
        assert obj._try_build_flagcx("store", 1, 4, None) is True
    assert obj._inner is sentinel
    assert seen["args"] == ("store", 1, 4, None)


def test_flagcx_both_signatures_failing_warns_and_falls_back(monkeypatch):
    def creator(*args):
        raise RuntimeError("boom")

    fake_flagcx = types.ModuleType("flagcx")
    fake_flagcx.createFlagcxBackend = creator
    monkeypatch.setitem(sys.modules, "flagcx", fake_flagcx)

    obj = pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)
    with pytest.warns(UserWarning, match="FlagCX init failed"):
        assert obj._try_build_flagcx("store", 0, 1, None) is False


def test_ascend_uses_hccl_native(monkeypatch):
    # ascend flagcx fails -> native hccl succeeds -> but view is None and ascend
    # is not direct=True -> raise NotImplementedError (documents that only the
    # FlagCX(cann) path is viable).
    obj, calls, _ = _make(
        monkeypatch, "ascend", flagcx_ok=False, native_ok=True, view_present=False
    )
    with pytest.raises(NotImplementedError, match="no flagos->device view"):
        obj._build_inner(None, 0, 1, None)
    assert calls["native"]  # hccl was attempted


def test_enflame_profile():
    """Enflame's torch-fl FlagCX build registers device 'flagos' and consumes
    privateuseone tensors directly, with no native fallback."""
    prof = pg._get_profile("enflame")
    assert prof.flagcx_dev == "flagos"
    assert prof.view is None
    assert prof.native is None
    assert prof.direct is True


def test_enflame_flagcx_selects_flagos_backend(monkeypatch):
    """Enflame defaults to the torch-fl FlagCX mode before importing flagcx."""
    monkeypatch.setenv("GEMS_VENDOR", "enflame")
    monkeypatch.delenv("FLAGCX_TORCH_BACKEND", raising=False)

    pg._configure_flagcx_torch_backend("enflame")
    assert pg.os.environ["FLAGCX_TORCH_BACKEND"] == "flagos"


def test_enflame_flagcx_preserves_explicit_backend(monkeypatch):
    """An explicit FlagCX backend selection remains authoritative."""
    monkeypatch.setenv("GEMS_VENDOR", "enflame")
    monkeypatch.setenv("FLAGCX_TORCH_BACKEND", "torch_gcu")

    pg._configure_flagcx_torch_backend("enflame")
    assert pg.os.environ["FLAGCX_TORCH_BACKEND"] == "torch_gcu"


def test_direct_is_opt_in_per_vendor():
    """direct=True suppresses the missing-view error, so it must stay opt-in: a
    vendor that merely lacks a view must NOT silently inherit it. Guards against
    flipping the default and handing raw flagos tensors to HCCL/MUSA/CNCL."""
    assert pg._VENDOR_PROFILES["enflame"].direct is True
    for v in ("ascend", "musa", "cambricon"):
        assert pg._VENDOR_PROFILES[v].direct is False, (
            f"{v} has no flagos->device view; direct=True would pass it a raw "
            f"privateuseone tensor instead of failing loudly"
        )


def test_enflame_flagcx_returns_identity_view(monkeypatch):
    """Enflame is direct=True, so _resolve_view returns the identity rather than
    raising for the absent view."""
    obj, calls, _ = _make(
        monkeypatch, "enflame", flagcx_ok=True, native_ok=False, view_present=False
    )
    view_fn = obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and not calls["native"]
    # view_fn should be identity for enflame
    sentinel = object()
    assert view_fn(sentinel) is sentinel


def test_enflame_no_fallback_when_flagcx_missing(monkeypatch):
    """Enflame has no native fallback; if FlagCX is unavailable, init fails."""
    obj, calls, _ = _make(
        monkeypatch, "enflame", flagcx_ok=False, native_ok=False, view_present=False
    )
    with pytest.raises(RuntimeError, match="no suitable inner backend.*none wired"):
        obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and not calls["native"]
