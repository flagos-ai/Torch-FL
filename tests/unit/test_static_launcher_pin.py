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

"""The static-launcher pin, and the ordering claim its call site makes.

`torch_fl.compile.inductor_backend` sets `use_static_cuda_launcher = False` for
its own compiles, but a `torch.compile` that names no backend went to Inductor's
default instead and died with `cannot import name '_StaticCudaLauncher'` on this
build. `pin_static_cuda_launcher()` closes that gap at import; the AST test pins
that it runs before the module's other probes, which is what keeps the export
ahead of any `torch._inductor` import those probes might perform.

The behaviour tests import the backend module, so they need torch; the AST test
does not, and neither needs a device.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "torch_fl" / "compile" / "inductor_backend.py"

ENV = "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"

#: The module-level probes, in the order the call site must run them.
PROBES = [
    "pin_static_cuda_launcher",
    "_patch_native_cuda_probe",
    "_patch_native_triton_autotune",
    "_patch_native_cache_system_key",
    "_bind_musa_flagtree_runtime",
    "_patch_cuda_rng_for_cpu_torch",
]


def _backend():
    return pytest.importorskip("torch_fl.compile.inductor_backend")


# ---------------------------------------------------------------------------
# Call site
# ---------------------------------------------------------------------------


def test_the_pin_runs_first_among_the_module_level_probes():
    """The export has to precede anything that could import inductor's config.

    Order is not load-bearing for correctness -- the pin writes both places --
    but it is load-bearing for the file's own explanation, which says the export
    is in place before the probes run. Pin the claim with the code.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    called = [
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert called == PROBES


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_a_build_with_the_class_is_left_alone(monkeypatch):
    backend = _backend()
    monkeypatch.setattr(backend, "_static_cuda_launcher_available", lambda: True)
    monkeypatch.delenv(ENV, raising=False)

    assert backend.pin_static_cuda_launcher() is False
    assert ENV not in os.environ


def test_a_build_without_the_class_is_pinned_off(monkeypatch):
    backend = _backend()
    monkeypatch.setattr(backend, "_static_cuda_launcher_available", lambda: False)
    monkeypatch.delenv(ENV, raising=False)

    assert backend.pin_static_cuda_launcher() is True
    assert os.environ[ENV] == "0"


def test_an_already_imported_config_is_corrected_too(monkeypatch):
    """The export cannot reach a config that was evaluated before it.

    `config.py` computes the default once at its own import, so a config already
    in `sys.modules` keeps whatever it read then; the attribute is the only
    lever left, and the pin has to use it.
    """
    backend = _backend()
    monkeypatch.setattr(backend, "_static_cuda_launcher_available", lambda: False)
    monkeypatch.delenv(ENV, raising=False)

    config = SimpleNamespace(use_static_cuda_launcher=True)
    monkeypatch.setitem(sys.modules, "torch._inductor.config", config)

    assert backend.pin_static_cuda_launcher() is True
    assert config.use_static_cuda_launcher is False


def test_an_explicit_export_wins(monkeypatch):
    """`set_foreign` semantics, and no half-application behind them.

    A user who exports the variable has made a choice about their own build, and
    the pin neither overwrites the export nor rewrites the attribute underneath
    it -- the two would then disagree.
    """
    backend = _backend()
    monkeypatch.setattr(backend, "_static_cuda_launcher_available", lambda: False)
    monkeypatch.setenv(ENV, "1")

    config = SimpleNamespace(use_static_cuda_launcher=True)
    monkeypatch.setitem(sys.modules, "torch._inductor.config", config)

    assert backend.pin_static_cuda_launcher() is False
    assert os.environ[ENV] == "1"
    assert config.use_static_cuda_launcher is True


def test_the_config_scoped_patch_agrees_with_the_pin(monkeypatch):
    """`_resolve_config_patches` must not disagree with the import-time answer."""
    backend = _backend()
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(backend, "_static_cuda_launcher_available", lambda: False)

    patches = backend._resolve_config_patches(None, None, None)
    assert patches["use_static_cuda_launcher"] is False
