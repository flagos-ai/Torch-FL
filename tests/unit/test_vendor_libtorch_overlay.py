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

"""Vendor libtorch selection must never alter the installed PyTorch wheel."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dcu_module_loader import REPO

MODULE_PATH = REPO / "torch_fl/accelerator/_vendor_libtorch.py"
CORE = ("libc10.so", "libtorch_cpu.so")


@pytest.fixture
def fake_runtime(tmp_path):
    installed = tmp_path / "site-packages" / "torch"
    stock = installed / "lib"
    stock.mkdir(parents=True)
    (installed / "__init__.py").write_text("origin = 'stock'\n")
    (installed / "version.py").write_text("__version__ = '2.10.0+cpu'\n")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "vendor_version.py").write_text("__version__ = '2.10.0+vendor'\n")
    for name in CORE:
        (stock / name).write_bytes(b"stock-" + name.encode())
        (vendor / name).write_bytes(b"vendor-" + name.encode())
    (stock / "libshm.so").write_bytes(b"stock-libshm")
    (vendor / "libtorch_cuda.so").write_bytes(b"vendor-cuda")
    return stock, vendor


def _load_module():
    spec = importlib.util.spec_from_file_location("_test_vendor_libtorch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest_tree(path):
    return {
        child.name: hashlib.sha256(child.read_bytes()).hexdigest()
        for child in path.iterdir()
    }


def test_activation_keeps_read_only_install_unchanged(
    fake_runtime, tmp_path, monkeypatch
):
    stock, vendor = fake_runtime
    module = _load_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(module, "_overlay_cache_root", lambda: str(cache))
    monkeypatch.setattr(module, "active_torch_lib", lambda: str(stock))
    monkeypatch.setattr(
        module, "discover_vendor_torch_lib", lambda *a, **k: str(vendor)
    )
    monkeypatch.setattr(module, "_preload_global", lambda *a, **k: [])
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    before = _digest_tree(stock)
    stock.chmod(0o555)
    overlay = None
    try:
        assert module.ensure_vendor_libtorch_links(
            "vendor", CORE, load_order=CORE, vendor="Test"
        )
        overlay = Path(sys.path[0])
        facade = overlay / "torch"
        assert (facade / "lib" / "libc10.so").resolve() == vendor / "libc10.so"
        assert (facade / "lib" / "libshm.so").resolve() == stock / "libshm.so"
        assert (facade / "version.py").resolve() == stock.parent / "version.py"
        assert json.loads((overlay / ".ready").read_text())["vendor_source"] == str(
            vendor
        )
        assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(overlay)
        assert module.ensure_vendor_libtorch_links("vendor", CORE)
        assert sys.path.count(str(overlay)) == 1
        module.restore_original_libtorch(CORE, bundle_dirname="vendor")
        assert _digest_tree(stock) == before
        assert not (stock / "_orig_backup").exists()
    finally:
        stock.chmod(0o755)
        if overlay is not None:
            sys.path.remove(str(overlay))


def test_imported_stock_torch_is_rejected_without_writes(fake_runtime, monkeypatch):
    stock, vendor = fake_runtime
    module = _load_module()
    monkeypatch.setattr(module, "active_torch_lib", lambda: str(stock))
    monkeypatch.setattr(
        module, "discover_vendor_torch_lib", lambda *a, **k: str(vendor)
    )
    monkeypatch.setitem(sys.modules, "torch", object())
    before = _digest_tree(stock)
    with pytest.raises(RuntimeError, match="import torch_fl before torch"):
        module.ensure_vendor_libtorch_links("vendor", CORE, vendor="Test")
    assert _digest_tree(stock) == before


def test_mismatched_vendor_version_fails_before_preload(fake_runtime, monkeypatch):
    stock, vendor = fake_runtime
    (vendor / "vendor_version.py").write_text("__version__ = '2.11.0+vendor'\n")
    module = _load_module()
    monkeypatch.setattr(module, "active_torch_lib", lambda: str(stock))
    monkeypatch.setattr(
        module, "discover_vendor_torch_lib", lambda *a, **k: str(vendor)
    )
    before = _digest_tree(stock)
    with pytest.raises(RuntimeError, match="incompatible with the installed PyTorch"):
        module.ensure_vendor_libtorch_links("vendor", CORE, vendor="Test")
    assert _digest_tree(stock) == before


def test_undeclared_vendor_abi_fails_before_preload(fake_runtime, monkeypatch):
    stock, vendor = fake_runtime
    (vendor / "vendor_version.py").unlink()
    module = _load_module()
    monkeypatch.delenv("TORCH_FL_VENDOR_TORCH_VERSION", raising=False)
    monkeypatch.setattr(module, "active_torch_lib", lambda: str(stock))
    monkeypatch.setattr(
        module, "discover_vendor_torch_lib", lambda *a, **k: str(vendor)
    )
    with pytest.raises(RuntimeError, match="ABI version metadata missing"):
        module.ensure_vendor_libtorch_links("vendor", CORE, vendor="Test")
    assert sorted(path.name for path in stock.iterdir()) == [
        "libc10.so",
        "libshm.so",
        "libtorch_cpu.so",
    ]


def test_equal_core_still_selects_missing_vendor_device_lib(
    fake_runtime, tmp_path, monkeypatch
):
    stock, vendor = fake_runtime
    for name in CORE:
        (stock / name).write_bytes((vendor / name).read_bytes())
    module = _load_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(module, "_overlay_cache_root", lambda: str(cache))
    monkeypatch.setattr(module, "active_torch_lib", lambda: str(stock))
    monkeypatch.setattr(
        module, "discover_vendor_torch_lib", lambda *a, **k: str(vendor)
    )
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    try:
        assert module.ensure_vendor_libtorch_links(
            "vendor", CORE, extra_so=("libtorch_cuda.so",)
        )
        overlay = Path(sys.path[0])
        assert (overlay / "torch/lib/libtorch_cuda.so").resolve() == (
            vendor / "libtorch_cuda.so"
        )
    finally:
        sys.path.remove(str(overlay))


def test_inherited_facade_imports_in_child(fake_runtime, tmp_path):
    stock, vendor = fake_runtime
    module = _load_module()
    module._overlay_cache_root = lambda: str(tmp_path)
    overlay = module._prepare_overlay(str(stock), str(vendor))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((overlay, env.get("PYTHONPATH", "")))
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.__file__); "
            "print(torch.__path__[0]); "
            "print(__import__('os').path.realpath(torch.__path__[0] + '/lib/libc10.so'))",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.splitlines() == [
        str(Path(overlay) / "torch" / "__init__.py"),
        str(Path(overlay) / "torch"),
        str(vendor / "libc10.so"),
    ]


def test_inherited_facade_can_preload_after_torch_import(
    fake_runtime, tmp_path, monkeypatch
):
    stock, vendor = fake_runtime
    module = _load_module()
    monkeypatch.setattr(module, "_overlay_cache_root", lambda: str(tmp_path))
    overlay = module._prepare_overlay(str(stock), str(vendor))
    active = str(Path(overlay) / "torch" / "lib")
    monkeypatch.setattr(module, "active_torch_lib", lambda: active)
    monkeypatch.setattr(
        module, "discover_vendor_torch_lib", lambda *a, **k: str(vendor)
    )
    monkeypatch.setitem(sys.modules, "torch", object())
    preloaded = []
    monkeypatch.setattr(
        module,
        "_preload_global",
        lambda *a, **k: preloaded.append(k["fallback_dir"]) or [],
    )
    assert module.ensure_vendor_libtorch_links("vendor", CORE, load_order=CORE)
    assert preloaded == [active]


def test_inherited_facade_rejects_another_vendor(fake_runtime, tmp_path, monkeypatch):
    stock, vendor = fake_runtime
    module = _load_module()
    monkeypatch.setattr(module, "_overlay_cache_root", lambda: str(tmp_path))
    overlay = module._prepare_overlay(str(stock), str(vendor))
    second = tmp_path / "other-vendor"
    second.mkdir()
    (second / "vendor_version.py").write_text("__version__ = '2.10.0+other'\n")
    for name in CORE:
        (second / name).write_bytes(b"other-" + name.encode())
    monkeypatch.setattr(
        module, "active_torch_lib", lambda: str(Path(overlay) / "torch" / "lib")
    )
    monkeypatch.setattr(
        module, "discover_vendor_torch_lib", lambda *a, **k: str(second)
    )
    with pytest.raises(RuntimeError, match="different vendor facade"):
        module.ensure_vendor_libtorch_links("vendor", CORE, vendor="Test")


def test_concurrent_publication_ignores_interrupted_stage(fake_runtime, tmp_path):
    stock, vendor = fake_runtime
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "interrupted.stage"
    (stale / "torch" / "lib").mkdir(parents=True)
    script = """\
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_test_vendor_libtorch', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._overlay_cache_root = lambda: sys.argv[4]
print(module._prepare_overlay(sys.argv[2], sys.argv[3]))
"""
    args = [
        sys.executable,
        "-c",
        script,
        str(MODULE_PATH),
        str(stock),
        str(vendor),
        str(cache),
    ]
    before = _digest_tree(stock)
    processes = [
        subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for _ in range(4)
    ]
    results = [process.communicate(timeout=20) for process in processes]
    assert all(process.returncode == 0 for process in processes), results
    paths = {out.strip() for out, _ in results}
    assert len(paths) == 1
    overlay = Path(paths.pop())
    assert (overlay / ".ready").is_file()
    assert (overlay / "torch" / "lib" / "libc10.so").resolve() == vendor / "libc10.so"
    assert _digest_tree(stock) == before
    assert stale.is_dir()
