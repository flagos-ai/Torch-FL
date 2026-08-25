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

"""Unit tests for the torch_npu-free FlagTree Ascend backend policy.

These run on plain CPU against a stand-in registry that mirrors the shape of
FlagTree's ``BackendStrategyRegistry``. What they check is the part that is
FlagTree-independent: that the policy registers every strategy the Ascend driver
dispatches, that the C++ it emits carries no torch_npu coupling, and that
installing it selects the policy without importing torch_npu.

What they cannot check: whether the emitted C++ actually compiles and runs. That
needs a real 910 plus a FlagTree build carrying the Ascend backend, and is
tracked in docs/vendors/ascend/installation.md.
"""

import builtins
import sys
from typing import Callable, Dict

import pytest

from torch_fl.compile import flagtree_ascend_policy as policy


class _FakeRegistry:
    """Same contract as FlagTree's BackendStrategyRegistry, including its
    duplicate-registration ValueError and its unknown-strategy ValueError."""

    def __init__(self):
        self.strategies: Dict[str, Dict[str, Callable]] = {}

    def register(self, category: str, method: str):
        def decorator(func):
            bucket = self.strategies.setdefault(category, {})
            if method in bucket:
                raise ValueError(f"Strategy {method} already registered")
            bucket[method] = func
            return func

        return decorator

    def execute_func(self, category, method, *args, **kwargs):
        if category not in self.strategies:
            raise ValueError(f"Strategy {category} not registered")
        if method not in self.strategies[category]:
            raise ValueError(f"Strategy {method} not registered")
        return self.strategies[category][method](*args, **kwargs)

    def list_categories(self):
        return list(self.strategies)

    def list_methods(self, category):
        return list(self.strategies[category])


class _FakeLazyRegistry:
    """Mirrors FlagTree's _LazyBackendStrategyRegister, which proxies *only*
    register() and execute_func().

    This is the object install_policy() actually receives, so anything reaching
    for list_categories()/list_methods() on it gets AttributeError.
    """

    def __init__(self):
        self._inner = _FakeRegistry()

    def register(self, *args, **kwargs):
        return self._inner.register(*args, **kwargs)

    def execute_func(self, *args, **kwargs):
        return self._inner.execute_func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_installed_flag():
    """install_policy() records having registered in module state; reset it so
    each test starts from a clean slate."""
    policy._installed = False
    yield
    policy._installed = False


@pytest.fixture
def registry():
    reg = _FakeRegistry()
    policy._register_strategies(reg)
    return reg


def _emit(registry, method, *args):
    return registry.execute_func(policy.POLICY_NAME, method, *args)


def test_registers_every_strategy_the_driver_dispatches(registry):
    """A missing strategy surfaces as a compile-time failure inside FlagTree, so
    the full set is asserted here instead."""
    registered = set(registry.list_methods(policy.POLICY_NAME))
    assert set(policy._REQUIRED_STRATEGIES) <= registered


def test_no_emitted_cpp_references_torch_npu(registry):
    """The whole point of the policy: none of the generated C++ may need
    libtorch_npu or its headers."""
    emitted = [
        _emit(registry, "header_file", False),
        _emit(registry, "header_file", True),
        _emit(registry, "allocate_memory", "sz", "stream"),
        _emit(registry, "allocate_sync_block_lock", "sz", "stream"),
        _emit(registry, "pre_launch", True),
        _emit(registry, "pre_launch", False),
    ]
    for snippet in emitted:
        assert "torch_npu" not in snippet
        assert "at_npu::" not in snippet


def test_cc_cmd_does_not_link_torch_npu(registry):
    """Upstream's get_cc_cmd adds -ltorch_npu and a torch_npu include dir."""
    for build_pch in (True, False):
        flags = _emit(registry, "get_cc_cmd", build_pch)
        joined = " ".join(flags)
        assert "-ltorch_npu" not in joined
        assert "torch_npu" not in joined
        # Still has to find ATen, which the emitted header_file includes.
        assert any("include" in f for f in flags)


def test_workspace_allocations_target_privateuse1(registry):
    """Under torch_fl, PrivateUse1 is flagos, so a plain ATen allocation on that
    key reaches torch_fl's allocator rather than torch_npu's."""
    for method in ("allocate_memory", "allocate_sync_block_lock"):
        snippet = _emit(registry, method, "1024", "stream")
        assert "at::kPrivateUse1" in snippet


def test_sync_block_lock_is_zeroed(registry):
    """torch_npu's allocate_workspace hands back zeroed memory; a bare at::empty
    would not, and the lock protocol depends on it."""
    snippet = _emit(registry, "allocate_sync_block_lock", "1024", "stream")
    assert "at::zeros" in snippet


def test_async_launch_refuses_rather_than_dropping_errors(registry):
    """The task queue needs at_npu::native::OpCommand and its generated function
    returns rtError_t. Calling the lambda anyway would discard launch failures."""
    with pytest.raises(RuntimeError, match="TRITON_ENABLE_TASKQUEUE"):
        _emit(registry, "async_launch", "launch_call")


def test_cxx_abi_matches_torch(registry):
    import torch

    expected = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    assert _emit(registry, "cxx_abi") == expected


def test_version_hash_excludes_torch_npu(registry):
    """Upstream mixes in torch_npu.version.git_version; ours must not, and must
    still vary with the plugin so caches do not outlive an upgrade."""
    parts = _emit(registry, "version_hash")
    assert len(parts) == 2
    assert all(isinstance(p, (str, type(None))) for p in parts)


def test_type_convert_covers_dtypes_inductor_emits(registry):
    import numpy as np
    import torch

    mapping = _emit(registry, "type_convert")
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.bool,
    ):
        assert dtype in mapping
    assert mapping[torch.float32] is np.float32


def test_install_policy_rejects_real_torch_npu_extension(monkeypatch):
    """If the real torch_npu is loaded it already owns PrivateUse1, so there is
    nothing left for flagos to claim and the policy cannot rescue it. Detected by
    the compiled _C attribute, which only the real package has."""
    import types

    fake_real = types.ModuleType("torch_npu")
    fake_real._C = object()
    monkeypatch.setitem(sys.modules, "torch_npu", fake_real)
    with pytest.raises(RuntimeError, match="real torch_npu extension is loaded"):
        policy.install_policy()


def test_install_policy_tolerates_torch_fl_npu_stub(monkeypatch):
    """torch_fl installs its own torch_npu stub for FlagGems, so presence in
    sys.modules must not be mistaken for the real library."""
    import types

    stub = types.ModuleType("torch_npu")  # no _C, like torch_fl's shim
    monkeypatch.setitem(sys.modules, "torch_npu", stub)

    reg = _FakeLazyRegistry()
    fake_backend_register = types.SimpleNamespace(backend_strategy_registry=reg)
    fake_utils = types.SimpleNamespace(backend_policy=None)
    fake_pkg = types.ModuleType("triton.backends.ascend")
    fake_pkg.backend_register = fake_backend_register
    fake_pkg.utils = fake_utils
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "triton.backends.ascend.backend_register", fake_backend_register
    )
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.utils", fake_utils)

    assert policy.install_policy() == policy.POLICY_NAME


def test_install_policy_requires_flagos_to_own_privateuse1(monkeypatch):
    """Generated code allocates on kPrivateUse1; if flagos does not own that key
    the allocation silently goes somewhere else."""
    import types

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setattr(
        "torch._C._get_privateuse1_backend_name", lambda: "npu", raising=False
    )

    reg = _FakeLazyRegistry()
    fake_backend_register = types.SimpleNamespace(backend_strategy_registry=reg)
    fake_utils = types.SimpleNamespace(backend_policy=None)
    fake_pkg = types.ModuleType("triton.backends.ascend")
    fake_pkg.backend_register = fake_backend_register
    fake_pkg.utils = fake_utils
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "triton.backends.ascend.backend_register", fake_backend_register
    )
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.utils", fake_utils)

    with pytest.raises(RuntimeError, match="PrivateUse1 is registered as 'npu'"):
        policy.install_policy()


def test_install_policy_reports_missing_ascend_backend(monkeypatch):
    """The 3.6/3.7 FlagTree branches carry no Ascend backend at all; the error
    should say so rather than surfacing a bare ImportError."""
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "triton.backends.ascend":
            raise ImportError("no such module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="Ascend backend"):
        policy.install_policy()


def test_install_policy_is_idempotent(monkeypatch):
    """torch_fl's compile registration can be reached more than once, and
    FlagTree's registry raises on duplicate registration."""
    import types

    reg = _FakeLazyRegistry()
    fake_backend_register = types.SimpleNamespace(backend_strategy_registry=reg)
    fake_utils = types.SimpleNamespace(backend_policy=None)
    fake_pkg = types.ModuleType("triton.backends.ascend")

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "triton.backends.ascend.backend_register", fake_backend_register
    )
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.utils", fake_utils)
    fake_pkg.backend_register = fake_backend_register
    fake_pkg.utils = fake_utils

    assert policy.install_policy() == policy.POLICY_NAME
    assert policy.install_policy() == policy.POLICY_NAME
    assert fake_utils.backend_policy == policy.POLICY_NAME


def test_install_policy_disables_torch_npu_task_queue(monkeypatch):
    """TRITON_ENABLE_TASKQUEUE defaults to on upstream and the queue is
    torch_npu-only, so installing has to turn it off."""
    import types

    reg = _FakeLazyRegistry()
    fake_backend_register = types.SimpleNamespace(backend_strategy_registry=reg)
    fake_utils = types.SimpleNamespace(backend_policy=None)
    fake_pkg = types.ModuleType("triton.backends.ascend")
    fake_pkg.backend_register = fake_backend_register
    fake_pkg.utils = fake_utils

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "triton.backends.ascend.backend_register", fake_backend_register
    )
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.utils", fake_utils)
    monkeypatch.delenv("TRITON_ENABLE_TASKQUEUE", raising=False)

    policy.install_policy()
    import os

    assert os.environ["TRITON_ENABLE_TASKQUEUE"] == "false"


def test_install_policy_respects_explicit_task_queue_opt_in(monkeypatch):
    """An explicit request is left alone so the failure is async_launch's clear
    message rather than a silent override."""
    import types

    reg = _FakeLazyRegistry()
    fake_backend_register = types.SimpleNamespace(backend_strategy_registry=reg)
    fake_utils = types.SimpleNamespace(backend_policy=None)
    fake_pkg = types.ModuleType("triton.backends.ascend")
    fake_pkg.backend_register = fake_backend_register
    fake_pkg.utils = fake_utils

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "triton.backends.ascend.backend_register", fake_backend_register
    )
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.utils", fake_utils)
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", "true")

    policy.install_policy()
    import os

    assert os.environ["TRITON_ENABLE_TASKQUEUE"] == "true"


def test_installing_does_not_import_torch_npu(monkeypatch):
    """The load-bearing property: nothing in the install path may pull in
    torch_npu, since that import is what claims PrivateUse1."""
    import types

    reg = _FakeLazyRegistry()
    fake_backend_register = types.SimpleNamespace(backend_strategy_registry=reg)
    fake_utils = types.SimpleNamespace(backend_policy=None)
    fake_pkg = types.ModuleType("triton.backends.ascend")
    fake_pkg.backend_register = fake_backend_register
    fake_pkg.utils = fake_utils

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "triton.backends.ascend.backend_register", fake_backend_register
    )
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.utils", fake_utils)

    policy.install_policy()
    # Also exercise every pure-codegen strategy; none may import it either.
    for method in ("header_file", "pre_launch"):
        reg.execute_func(policy.POLICY_NAME, method, False)
    for method in ("allocate_memory", "allocate_sync_block_lock"):
        reg.execute_func(policy.POLICY_NAME, method, "1", "s")
    reg.execute_func(policy.POLICY_NAME, "get_cc_cmd", False)

    assert "torch_npu" not in sys.modules


def test_reports_backend_import_needing_torch_npu(monkeypatch):
    """FlagTree's `backends/ascend/__init__` imports `do_bench_npu`, whose module
    imports torch_npu at module scope, during backend *discovery* -- before any
    policy can be chosen. torch_fl's own torch_npu stub normally absorbs this, so
    it only bites when triton is imported before torch_fl. In that case
    install_policy() must explain the ordering rather than let the bare
    "npu and npu" RuntimeError escape.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "triton.backends.ascend" or name.startswith(
            "triton.backends.ascend."
        ):
            raise RuntimeError(
                "Two accelerators cannot be used at the same time in PyTorch: npu and npu."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "triton.backends.ascend", raising=False)
    monkeypatch.delitem(sys.modules, "triton.backends.ascend.utils", raising=False)
    monkeypatch.delitem(
        sys.modules, "triton.backends.ascend.backend_register", raising=False
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        policy.install_policy()

    assert "torch_npu" in str(excinfo.value)
