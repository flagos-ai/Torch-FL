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

"""Unit coverage for the _flagos_nccl sentinel handling in ProcessGroupFlagOS.

torch_fl/comm/_nccl_ext installs a ``None`` sentinel when the compiled
_flagos_nccl module is absent, and importing that sentinel is not an error -- so
the native NCCL fallback has to test for it explicitly. It used to import the
sentinel and call ``make_nccl_backend`` on it, so the failure surfaced as
``AttributeError: 'NoneType' object has no attribute 'make_nccl_backend'`` with
the ImportError that explained why the extension was missing thrown away (#365).

No device and no compiled extension are needed: the extension is faked. The last
two cases build a scratch package from the real _nccl_ext source, which is what
pins the sentinel's import semantics without requiring the .so to exist.

Run: pytest tests/unit/test_nccl_extension_fallback.py
"""

import importlib
import sys
from datetime import timedelta
from pathlib import Path

import pytest

pg = pytest.importorskip("torch_fl.comm.process_group")
_nccl_ext = pytest.importorskip("torch_fl.comm._nccl_ext")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_INIT = REPO_ROOT / "torch_fl" / "comm" / "_nccl_ext" / "__init__.py"


class _FakeExtension:
    """Stand-in for the compiled _flagos_nccl module."""

    def __init__(self):
        self.calls = []

    def make_nccl_backend(self, *args):
        self.calls.append(args)
        return "inner-backend"


def _refuse(self, store, rank, world_size, timeout):
    """Stand-in for _try_build_flagcx that declines, as when flagcx is absent."""
    return False


def _bare_group():
    """A ProcessGroupFlagOS built without __init__ or the C++ base ctor.

    The same construction tests/unit/test_vendor_routing.py uses to drive the
    fallback chain without a store, a rank or a world size.
    """
    return pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)


@pytest.fixture
def extension_branch(monkeypatch):
    """Make the _flagos_nccl branch of _try_build_nccl reachable.

    _try_build_nccl returns from its first branch whenever torch.distributed
    exposes ProcessGroupNCCL, which a hipified torch (the Hygon DCU runner) does
    through RCCL. Removing it is what a CPU-only wheel looks like, and is the
    only way to reach the extension code path at all.
    """
    monkeypatch.delattr(pg.torch.distributed, "ProcessGroupNCCL", raising=False)
    assert getattr(pg.torch.distributed, "ProcessGroupNCCL", None) is None


@pytest.fixture
def unbuilt_extension(monkeypatch):
    """Make both extension lookups fail, as in a slim install.

    ``sys.modules["_flagos_nccl"] = None`` makes ``import _flagos_nccl`` raise
    ImportError deterministically, whether or not a loose build exists on this
    machine.
    """
    loader_error = ImportError("cannot open shared object file: libc10_cuda.so")
    monkeypatch.setattr(_nccl_ext, "get_extension", lambda: None)
    monkeypatch.setattr(_nccl_ext, "load_error", lambda: loader_error)
    monkeypatch.setitem(sys.modules, "_flagos_nccl", None)
    return loader_error


# ---------------------------------------------------------------------------
# _try_build_nccl: the sentinel must never be dereferenced
# ---------------------------------------------------------------------------


def test_unbuilt_extension_reports_unavailable_instead_of_none_attribute_error(
    monkeypatch, extension_branch, unbuilt_extension
):
    """The absent-extension case returns False; it must not raise AttributeError.

    This is the reported symptom: the old code reached
    ``_flagos_nccl.make_nccl_backend`` on the sentinel and lost the loader error.
    """
    group = _bare_group()

    assert group._try_build_nccl("store", 0, 1, None) is False
    assert not hasattr(group, "_inner")
    assert "libc10_cuda.so" in group._nccl_skip_reason


def test_present_extension_is_called_with_the_expected_arguments(
    monkeypatch, extension_branch
):
    """The factory receives the c10d arguments and the timeout in milliseconds."""
    fake = _FakeExtension()
    monkeypatch.setattr(_nccl_ext, "get_extension", lambda: fake)

    group = _bare_group()

    assert group._try_build_nccl("store", 3, 8, timedelta(seconds=90)) is True
    assert group._inner == "inner-backend"
    assert fake.calls == [("store", 3, 8, 90000, False)]
    assert getattr(group, "_nccl_skip_reason", None) is None


def test_absent_timeout_becomes_zero_milliseconds(monkeypatch, extension_branch):
    """c10d passes None when the caller set no timeout; the C side wants an int."""
    fake = _FakeExtension()
    monkeypatch.setattr(_nccl_ext, "get_extension", lambda: fake)

    group = _bare_group()

    assert group._try_build_nccl("store", 0, 1, None) is True
    assert fake.calls == [("store", 0, 1, 0, False)]


def test_loose_build_layout_is_still_supported(monkeypatch, extension_branch):
    """A top-level _flagos_nccl import still works when the package one misses.

    build.py --inplace drops the .so next to the source, but a wheel-style
    install can leave only the top-level module importable.
    """
    fake = _FakeExtension()
    monkeypatch.setattr(_nccl_ext, "get_extension", lambda: None)
    monkeypatch.setitem(sys.modules, "_flagos_nccl", fake)

    group = _bare_group()

    assert group._try_build_nccl("store", 0, 1, None) is True
    assert fake.calls == [("store", 0, 1, 0, False)]


# ---------------------------------------------------------------------------
# The final diagnostic carries the loader failure
# ---------------------------------------------------------------------------


def test_missing_extension_is_reported_in_the_final_diagnostic(
    monkeypatch, extension_branch, unbuilt_extension
):
    """With FlagCX and the extension both gone, the error names the real cause.

    Anything mentioning NoneType means the sentinel leaked into the message
    instead of the loader failure.
    """
    monkeypatch.setenv("GEMS_VENDOR", "nvidia")
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_flagcx", _refuse)
    group = _bare_group()

    with pytest.raises(RuntimeError) as excinfo:
        group._build_inner("store", 0, 1, None)

    message = str(excinfo.value)
    assert "no suitable inner backend" in message
    assert "Native NCCL fallback unavailable" in message
    assert "libc10_cuda.so" in message
    assert "NoneType" not in message


def test_diagnostic_is_unchanged_when_the_extension_was_never_consulted(
    monkeypatch, extension_branch
):
    """A vendor whose native backend is not the NCCL one gets no NCCL footnote."""
    monkeypatch.setenv("GEMS_VENDOR", "musa")
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_flagcx", _refuse)
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_mccl", _refuse)
    group = _bare_group()

    with pytest.raises(RuntimeError) as excinfo:
        group._build_inner("store", 0, 1, None)

    assert "Native NCCL fallback unavailable" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# The _nccl_ext package's own contract
# ---------------------------------------------------------------------------


def test_extension_package_state_is_self_consistent():
    """Either the extension loaded, or load_error() says why it did not."""
    extension = _nccl_ext.get_extension()
    error = _nccl_ext.load_error()
    if extension is None:
        assert isinstance(error, ImportError)
    else:
        assert error is None


def test_absent_submodule_yields_a_none_sentinel_that_imports_cleanly(
    tmp_path, monkeypatch
):
    """Pin the trap the fix works around, using the real __init__.py source.

    ``from pkg import _flagos_nccl`` succeeds and binds None, which is why the
    caller cannot rely on ImportError to detect a missing extension.
    """
    package = tmp_path / "sentinel_absent_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        EXT_INIT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    module = importlib.import_module("sentinel_absent_pkg")
    try:
        assert module.get_extension() is None
        assert isinstance(module.load_error(), ImportError)

        from sentinel_absent_pkg import _flagos_nccl

        assert _flagos_nccl is None
    finally:
        sys.modules.pop("sentinel_absent_pkg", None)


def test_present_submodule_is_loaded_despite_the_sentinel(tmp_path, monkeypatch):
    """The sentinel is assigned in the except, never before the import.

    ``from . import _flagos_nccl`` consults ``hasattr(package, "_flagos_nccl")``
    first, so a pre-assigned ``None`` makes the import a silent no-op and the
    compiled extension would never load even when it is present -- the failure
    mode the fix must not introduce. A stub child stands in for the .so.
    """
    package = tmp_path / "sentinel_present_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        EXT_INIT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (package / "_flagos_nccl.py").write_text("MARKER = 'compiled'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    module = importlib.import_module("sentinel_present_pkg")
    try:
        assert module.load_error() is None
        assert module.get_extension().MARKER == "compiled"
    finally:
        sys.modules.pop("sentinel_present_pkg._flagos_nccl", None)
        sys.modules.pop("sentinel_present_pkg", None)
