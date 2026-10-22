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

"""The Kunlunxin libcudart preload, exercised without hardware.

FlagTree's XPU launcher is dlopen'd on the first kernel launch and resolves
``cudaMemcpy`` at that moment. The symbol lives in
``/usr/local/xcudart/lib/libcudart.so.12`` as a versioned definition
(``cudaMemcpy@@libcudart.so.12``), so it is only findable if that library was
already loaded into the global scope. Without it the launcher is left
half-resolved: the kernel is accepted and ``xpu_wait(NULL)`` never returns --
measured on a P800, where a hand-written ``@triton.jit`` kernel, every FlagGems
op and every vendor op all hang on the sync that follows, and all complete and
return correct values once this runs first.

These tests pin the selection logic: when it runs, which path it picks, and that
it is inert off Kunlunxin. Loading a real library is faked, so the suite passes
on hosts without the vendor tree.

The function is loaded out of ``torch_fl/flagos/__init__.py`` by AST extraction
rather than by importing torch_fl: the unit environment has no torch, and
``import torch_fl`` claims the PrivateUse1 dispatch key.

Run: pytest tests/unit/test_kunlunxin_libcudart_preload.py -v
"""

import ast
import ctypes
import sys
import types
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "torch_fl" / "flagos" / "__init__.py"
_UNDER_TEST = ("_preload_xcudart_for_kunlunxin", "_loaded_shared_objects")


def _load_functions():
    tree = ast.parse(_SOURCE.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _UNDER_TEST
    ]
    assert {node.name for node in wanted} == set(_UNDER_TEST), (
        f"{_SOURCE} no longer defines {_UNDER_TEST}"
    )
    module = types.ModuleType("kunlunxin_preload_under_test")
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(_SOURCE), "exec"),
        module.__dict__,
    )
    sys.modules[module.__name__] = module
    return module


@pytest.fixture()
def flagos():
    module = _load_functions()
    module._platform = lambda: "kunlunxin"
    return module


def test_loads_from_xpytorch_xcudart_path(flagos, tmp_path, monkeypatch):
    lib = tmp_path / "lib" / "libcudart.so.12"
    lib.parent.mkdir()
    lib.write_text("")
    monkeypatch.setenv("XPYTORCH_XCUDART_PATH", str(tmp_path))

    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda p, mode=None: loaded.append((p, mode)))
    monkeypatch.setattr(flagos, "_loaded_shared_objects", lambda: iter(()))

    flagos._preload_xcudart_for_kunlunxin()

    assert loaded == [(str(lib), ctypes.RTLD_GLOBAL)]


def test_uses_the_default_layout_when_no_env(flagos, monkeypatch):
    monkeypatch.delenv("XPYTORCH_XCUDART_PATH", raising=False)
    monkeypatch.delenv("XPU_CUDART_PATH", raising=False)
    monkeypatch.setattr(flagos, "_loaded_shared_objects", lambda: iter(()))

    seen = []

    class _Stat:
        @staticmethod
        def exists(path):
            return path == "/usr/local/xcudart/lib/libcudart.so.12"

    real_os = __import__("os")
    monkeypatch.setattr(real_os.path, "exists", _Stat.exists)
    monkeypatch.setattr(ctypes, "CDLL", lambda p, mode=None: seen.append((p, mode)))

    flagos._preload_xcudart_for_kunlunxin()

    assert seen == [("/usr/local/xcudart/lib/libcudart.so.12", ctypes.RTLD_GLOBAL)]


def test_skips_when_already_mapped(flagos, monkeypatch):
    monkeypatch.setattr(
        flagos,
        "_loaded_shared_objects",
        lambda: iter(["/usr/lib/x86_64-linux-gnu/libcudart.so.12"]),
    )
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda p, mode=None: loaded.append(p))

    flagos._preload_xcudart_for_kunlunxin()

    assert loaded == []


def test_noop_off_kunlunxin(flagos, monkeypatch):
    monkeypatch.setattr(flagos, "_platform", lambda: "ascend")
    monkeypatch.setattr(flagos, "_loaded_shared_objects", lambda: iter(()))
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda p, mode=None: loaded.append(p))

    flagos._preload_xcudart_for_kunlunxin()

    assert loaded == []


def test_survives_a_missing_library(flagos, monkeypatch):
    monkeypatch.setattr(flagos, "_loaded_shared_objects", lambda: iter(()))
    monkeypatch.setattr(
        ctypes, "CDLL", lambda p, mode=None: (_ for _ in ()).throw(OSError)
    )
    real_os = __import__("os")
    monkeypatch.setattr(real_os.path, "exists", lambda path: True)

    # Must not raise: a missing vendor tree is not device init's problem.
    flagos._preload_xcudart_for_kunlunxin()


def test_idempotent(flagos, monkeypatch):
    """Second call after a successful load is a no-op."""
    monkeypatch.setattr(ctypes, "CDLL", lambda p, mode=None: object())
    real_os = __import__("os")
    monkeypatch.setattr(real_os.path, "exists", lambda path: True)

    mapped = []
    monkeypatch.setattr(flagos, "_loaded_shared_objects", lambda: iter(mapped))

    flagos._preload_xcudart_for_kunlunxin()
    # A real load would put the path in /proc/self/maps; stand that in.
    mapped.append("/usr/local/xcudart/lib/libcudart.so.12")

    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda p, mode=None: loaded.append(p))
    flagos._preload_xcudart_for_kunlunxin()

    assert loaded == []
