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

"""Unit coverage for DCU vendor-core decoupling: mode selection and preload.

The decisive property is that the default mode never touches the installed torch
wheel. These tests drive the loader with fake bundles and a fake ``ctypes.CDLL``,
so they run anywhere: no DTK, no GPU, no torch import side effects.
"""

import os
import types

import pytest
from dcu_module_loader import REPO, load_module

_LINK_REL = "torch_fl/accelerator/dcu/_dcu_libtorch_link.py"

link = load_module(
    _LINK_REL,
    "_flagos_test_dcu_libtorch_link",
    deps=(
        (
            "torch_fl.accelerator._vendor_libtorch",
            "torch_fl/accelerator/_vendor_libtorch.py",
        ),
    ),
)


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test gets a fresh loader: the real one is a process-wide singleton."""
    monkeypatch.setattr(link, "_preloaded", False)
    monkeypatch.setattr(link, "_device_handles", [])
    monkeypatch.delenv("FLAGOS_DCU_VENDOR_CORE", raising=False)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """A bundle with only device libs, plus an official torch/lib beside it."""
    bundle = tmp_path / "lib_dcu"
    official = tmp_path / "torch" / "lib"
    bundle.mkdir()
    official.mkdir(parents=True)

    for name in (
        "libflagos_dtk_core_compat.so",
        "libcaffe2_nvrtc.so",
        "libc10_hip.so",
        "libmagma.so",
        "libtorch_hip.so",
    ):
        (bundle / name).write_bytes(b"\x7fELF fake")
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch.so"):
        (official / name).write_bytes(b"\x7fELF fake")

    loaded = []

    class FakeCDLL:
        def __init__(self, path, mode=0):
            loaded.append((os.path.basename(path), os.path.dirname(path)))

    monkeypatch.setattr(link.ctypes, "CDLL", FakeCDLL)
    monkeypatch.setattr(link, "_discover_dcu_torch_lib", lambda: str(bundle))
    monkeypatch.setattr(link, "active_torch_lib", lambda: str(official))
    return types.SimpleNamespace(
        bundle=bundle, official=official, loaded=loaded, root=tmp_path
    )


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def test_decoupled_is_the_default(monkeypatch):
    assert link.vendor_core_mode() is False


@pytest.mark.parametrize("value", ["1", "on", "true", "YES", "True"])
def test_vendor_core_mode_accepts_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("FLAGOS_DCU_VENDOR_CORE", value)
    assert link.vendor_core_mode() is True


@pytest.mark.parametrize("value", ["0", "off", "false", "no", ""])
def test_vendor_core_mode_rejects_falsy_spellings(monkeypatch, value):
    monkeypatch.setenv("FLAGOS_DCU_VENDOR_CORE", value)
    assert link.vendor_core_mode() is False


def test_setup_dispatches_to_preload_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(
        link, "preload_dcu_device_libs", lambda: calls.append("preload") or True
    )
    monkeypatch.setattr(
        link,
        "ensure_dcu_libtorch_links",
        lambda: calls.append("relink") or True,
    )
    assert link.setup_dcu_runtime() is True
    assert calls == ["preload"]


def test_setup_dispatches_to_relink_in_vendor_core_mode(monkeypatch):
    calls = []
    monkeypatch.setenv("FLAGOS_DCU_VENDOR_CORE", "1")
    monkeypatch.setattr(
        link, "preload_dcu_device_libs", lambda: calls.append("preload") or True
    )
    monkeypatch.setattr(
        link,
        "ensure_dcu_libtorch_links",
        lambda: calls.append("relink") or True,
    )
    assert link.setup_dcu_runtime() is True
    assert calls == ["relink"]


# ---------------------------------------------------------------------------
# Decoupled preload
# ---------------------------------------------------------------------------


def test_preload_never_mutates_the_torch_install(fake_env):
    assert link.preload_dcu_device_libs() is True
    # No symlink, no backup dir: the whole point of the decoupled mode.
    assert not (fake_env.official / "_orig_backup").exists()
    assert sorted(p.name for p in fake_env.official.iterdir()) == [
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
    ]
    assert not any(p.is_symlink() for p in fake_env.official.iterdir())


def test_preload_order_is_core_then_shim_then_device(fake_env):
    link.preload_dcu_device_libs()
    names = [name for name, _ in fake_env.loaded]
    assert names == [
        "libc10.so",
        "libtorch_cpu.so",
        "libtorch.so",
        "libflagos_dtk_core_compat.so",
        "libcaffe2_nvrtc.so",
        "libc10_hip.so",
        "libmagma.so",
        "libtorch_hip.so",
    ]
    # The shim must precede libtorch_hip.so, else its 32 exports are not in the
    # global scope when the loader binds the device library.
    assert names.index("libflagos_dtk_core_compat.so") < names.index("libtorch_hip.so")


def test_official_core_is_loaded_from_the_torch_wheel_not_the_bundle(fake_env):
    link.preload_dcu_device_libs()
    sources = dict(fake_env.loaded)
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch.so"):
        assert sources[name] == str(fake_env.official)
    for name in ("libtorch_hip.so", "libc10_hip.so"):
        assert sources[name] == str(fake_env.bundle)


def test_preload_is_idempotent(fake_env):
    assert link.preload_dcu_device_libs() is True
    first = len(fake_env.loaded)
    assert link.preload_dcu_device_libs() is True
    assert len(fake_env.loaded) == first


def test_preload_noops_without_a_discoverable_bundle(monkeypatch):
    monkeypatch.setattr(link, "_discover_dcu_torch_lib", lambda: None)
    assert link.preload_dcu_device_libs() is False


def test_missing_shim_raises_actionable_error(fake_env, monkeypatch):
    (fake_env.bundle / "libflagos_dtk_core_compat.so").unlink()
    # _compat_shim_path also falls back to the in-tree build output, which does
    # exist on a DCU build host; pin it to "not found" to exercise the message.
    monkeypatch.setattr(link, "_compat_shim_path", lambda lib_dir: None)
    with pytest.raises(RuntimeError, match="FLAGOS_DCU_VENDOR_CORE=1"):
        link.preload_dcu_device_libs()
    assert not (fake_env.official / "_orig_backup").exists()


def test_shim_is_found_in_the_bundle(fake_env):
    resolved = link._compat_shim_path(str(fake_env.bundle))
    assert resolved == str(fake_env.bundle / "libflagos_dtk_core_compat.so")


def test_missing_device_lib_names_the_path_and_the_fix(fake_env):
    (fake_env.bundle / "libtorch_hip.so").unlink()
    with pytest.raises(FileNotFoundError, match="bundle_dcu_libtorch.sh"):
        link.preload_dcu_device_libs()


def test_dlopen_failure_is_reported_with_causes(fake_env, monkeypatch):
    def boom(path, mode=0):
        if path.endswith("libtorch_hip.so"):
            raise OSError("libgalaxyhip.so.5: cannot open shared object file")

    monkeypatch.setattr(link.ctypes, "CDLL", boom)
    with pytest.raises(RuntimeError, match="DTK driver stack"):
        link.preload_dcu_device_libs()


def test_missing_official_core_is_reported(fake_env):
    (fake_env.official / "libtorch_cpu.so").unlink()
    with pytest.raises(FileNotFoundError, match="official PyTorch core"):
        link.preload_dcu_device_libs()


def test_no_torch_at_all_is_reported(fake_env, monkeypatch):
    monkeypatch.setattr(link, "active_torch_lib", lambda: None)
    with pytest.raises(RuntimeError, match="no importable torch"):
        link.preload_dcu_device_libs()


# ---------------------------------------------------------------------------
# Legacy mode guard
# ---------------------------------------------------------------------------


def test_legacy_mode_refuses_a_decoupled_bundle(fake_env, monkeypatch):
    """The generic linker would fail midway with a message about libc10.so that
    does not explain the real cause (bundle built in the other mode)."""
    monkeypatch.setenv("FLAGOS_DCU_VENDOR_CORE", "1")
    monkeypatch.setattr(link, "_bundled_dcu_lib", lambda: str(fake_env.bundle))

    def must_not_run(*args, **kwargs):
        raise AssertionError("ensure_vendor_libtorch_links must not be reached")

    monkeypatch.setattr(link, "ensure_vendor_libtorch_links", must_not_run)
    with pytest.raises(RuntimeError, match="bundled in decoupled mode"):
        link.setup_dcu_runtime()
    assert not (fake_env.official / "_orig_backup").exists()


def test_legacy_mode_proceeds_with_a_full_bundle(fake_env, monkeypatch):
    monkeypatch.setenv("FLAGOS_DCU_VENDOR_CORE", "1")
    for name in link._CORE_SO:
        (fake_env.bundle / name).write_bytes(b"\x7fELF fake")
    monkeypatch.setattr(link, "_bundled_dcu_lib", lambda: str(fake_env.bundle))
    monkeypatch.setattr(link, "ensure_vendor_libtorch_links", lambda *a, **k: True)
    assert link.setup_dcu_runtime() is True


def test_legacy_preloads_every_symlinked_so_from_the_bundle():
    """Anything legacy mode symlinks into torch/lib must be preloaded by bundle path.

    Once the symlink exists, torch/lib is ahead of every RUNPATH on the CI's
    LD_LIBRARY_PATH, so a transitive DT_NEEDED resolves to the symlink and glibc
    expands the loaded object's ``$ORIGIN`` to torch/lib -- where the bundle's
    auditwheel-mangled deps do not exist. Measured: libmagma.so opened that way
    dies with "libmkl_gf_lp64-<hash>.so: cannot open shared object file", while the
    same file opened from lib_dcu loads and satisfies the later resolution by inode.
    """
    assert set(link._LOAD_ORDER) == set(link._CORE_SO) | set(link._HIP_SO)


def test_legacy_preloads_magma_before_libtorch():
    """libtorch.so -> libtorch_hip.so -> libmagma.so, so libmagma must come first."""
    assert link._LOAD_ORDER.index("libmagma.so") < link._LOAD_ORDER.index("libtorch.so")


def test_core_and_device_sets_are_disjoint():
    """A name in both sets would be pruned and bundled in the same run."""
    assert not set(link._CORE_SO) & set(link._DEVICE_LOAD_ORDER)


def test_restore_is_available_for_rollback():
    assert callable(link.restore_original_libtorch)


def test_module_docstring_documents_both_modes():
    doc = link.__doc__ or ""
    assert "FLAGOS_DCU_VENDOR_CORE" in doc
    assert "before" in doc and "import torch" in doc


def test_loader_never_imports_torch():
    """The preload must run before ``import torch``; importing torch anywhere in
    this module would defeat it by letting PyTorch cache its CUDAHooks first."""
    source = (REPO / _LINK_REL).read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import torch\n", "import torch ", "import torch"))
        and not line.strip().startswith("import torch_fl")
    ]
    assert offenders == [], offenders
