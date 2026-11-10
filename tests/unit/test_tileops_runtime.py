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

"""Unit coverage for TileOPs runtime fallback argument reconstruction."""

import sys
import types

import pytest
import torch

from torch_fl.tileops import runtime

_real_configure_device = runtime.configure_device


def test_keyword_only_reduction_fallback_preserves_schema():
    """Declined var routes must call ATen with keyword-only keepdim."""
    impl = runtime._reduce_impl(
        object,
        (torch.float16,),
        {"keepdim": True, "correction": True},
        "var.correction",
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    got = impl(x, [-1], True, correction=1)
    want = torch.ops.aten.var.correction(x, [-1], correction=1, keepdim=True)

    torch.testing.assert_close(got, want)


def test_keyword_only_reduction_fallback_handles_full_reduction():
    """Unsupported full reductions must not pass keepdim positionally."""
    impl = runtime._reduce_impl(
        object,
        (torch.float16,),
        {"keepdim": True, "correction": True},
        "std.correction",
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    got = impl(x, None, True, correction=1)
    want = torch.ops.aten.std.correction(x, correction=1, keepdim=True)

    torch.testing.assert_close(got, want)


def test_fallback_runs_nested_arguments_on_cpu(monkeypatch):
    """The Python fallback traverses container arguments and returns its result."""
    cpu_tensor = torch.tensor([0, 1, 2, 3])
    calls = []

    def aten_op(*args, **kwargs):
        calls.append((args, kwargs))
        assert all(
            value.device.type == "cpu"
            for value in torch.utils._pytree.tree_leaves((args, kwargs))
            if isinstance(value, torch.Tensor)
        )
        nested = kwargs["values"]
        return torch.stack((args[0], nested[0][0])), [nested[0][0] + 1]

    monkeypatch.setattr(runtime, "_aten_op", lambda overload: aten_op)
    result = runtime._make_fallback("relu")(cpu_tensor, values=[[cpu_tensor]])

    assert torch.equal(result[0], torch.stack((cpu_tensor, cpu_tensor)))
    assert torch.equal(result[1][0], cpu_tensor + 1)
    assert calls


@pytest.fixture(autouse=True)
def _clear_provider_state(monkeypatch):
    monkeypatch.setattr(runtime, "import_tileops", lambda: object())
    monkeypatch.setattr(runtime, "_instances", {})
    monkeypatch.setattr(runtime, "_ADAPTERS", {})
    monkeypatch.setattr(runtime, "_CONFIGURED_TARGETS", set())
    monkeypatch.setattr(runtime, "_HAS_SELECT_TARGET", None)
    monkeypatch.setattr(runtime, "configure_device", lambda target: None)
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "cuda")
    yield


def _provider(monkeypatch, targets=(), detected=None):
    dispatch = types.ModuleType("tileops.backend.dispatch")
    dispatch.detect_target = lambda device: (detected or {}).get(device.type)
    registry = types.ModuleType("tileops.backend.registry")
    registry.known_targets = lambda: set(targets)
    registry.BUILDERS = {
        (name, target): object() for name in targets for target in targets
    }
    monkeypatch.setitem(
        __import__("sys").modules, "tileops", types.ModuleType("tileops")
    )
    backend = types.ModuleType("tileops.backend")
    backend.detect_target = dispatch.detect_target
    monkeypatch.setitem(__import__("sys").modules, "tileops.backend", backend)
    monkeypatch.setitem(__import__("sys").modules, "tileops.backend.dispatch", dispatch)
    monkeypatch.setitem(__import__("sys").modules, "tileops.backend.registry", registry)


def _device(kind, index=0):
    return types.SimpleNamespace(type=kind, index=index)


def test_maca_target_uses_cuda_abi_independent_of_name():
    adapter = runtime.get_device_adapter("tileops_maca")
    assert adapter[0] == "cuda"


def test_configure_cuda_abi_needs_no_vendor_stream_lookup(monkeypatch):
    """The native CUDA/MACA stream is usable without a TileLang override."""
    tilelang_adapter = types.ModuleType("tilelang.jit.adapter.base")

    class BaseKernelAdapter:
        get_current_stream_functor = object()

    tilelang_adapter.BaseKernelAdapter = BaseKernelAdapter
    monkeypatch.setattr(runtime, "configure_device", _real_configure_device)
    monkeypatch.setitem(sys.modules, "tilelang.jit.adapter.base", tilelang_adapter)

    before = BaseKernelAdapter.get_current_stream_functor
    runtime.configure_device("tileops_maca")

    assert BaseKernelAdapter.get_current_stream_functor is before


def test_tileops_targets_are_resolved_per_op(monkeypatch):
    dispatch = types.ModuleType("tileops.backend.dispatch")
    dispatch.detect_target = lambda device: "tileops_maca"
    registry = types.ModuleType("tileops.backend.registry")
    registry.known_targets = lambda: {"cuda", "tileops_maca"}
    registry.BUILDERS = {}
    backend = types.ModuleType("tileops.backend")
    backend.detect_target = dispatch.detect_target
    for name, module in (
        ("tileops", types.ModuleType("tileops")),
        ("tileops.backend", backend),
        ("tileops.backend.dispatch", dispatch),
        ("tileops.backend.registry", registry),
    ):
        monkeypatch.setitem(__import__("sys").modules, name, module)
    monkeypatch.setattr(
        runtime, "_detection_device", lambda device: _device("cuda", device.index)
    )
    assert runtime.resolve_target(_device("cuda")) == "tileops_maca"
    assert runtime._provider_target(None, _device("cuda"), "cuda") == "tileops_maca"


def test_constructor_probe_keeps_device_out_of_target_detection(monkeypatch):
    dispatch = types.ModuleType("tileops.backend.dispatch")
    dispatch.detect_target = lambda device: (
        "tileops_maca" if device.type == "cuda" else None
    )
    registry = types.ModuleType("tileops.backend.registry")
    registry.known_targets = lambda: {"cuda", "tileops_maca"}
    backend = types.ModuleType("tileops.backend")
    backend.detect_target = dispatch.detect_target
    for name, module in (
        ("tileops", types.ModuleType("tileops")),
        ("tileops.backend", backend),
        ("tileops.backend.dispatch", dispatch),
        ("tileops.backend.registry", registry),
    ):
        monkeypatch.setitem(__import__("sys").modules, name, module)
    assert runtime.resolve_target(_device("meta"), allow_unclaimed=True) == "cuda"


def test_musa_provider_without_target_api_does_not_claim_cuda(monkeypatch):
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "musa")
    backend = types.ModuleType("tileops.backend")
    backend.register_detector = object()
    backend.register_kernel_builder = object()
    registry = types.ModuleType("tileops.backend.registry")
    registry.known_targets = lambda: {"musa"}
    registry.BUILDERS = {("ReluFwdOp", "musa"): object()}
    monkeypatch.setattr(runtime, "_provider_backend", lambda: backend)
    monkeypatch.setattr(runtime, "_provider_registry", lambda: registry)
    runtime._HAS_SELECT_TARGET = False
    assert runtime.resolve_target(_device("cuda"), allow_unclaimed=True) is None
    assert runtime.resolve_target(_device("meta"), allow_unclaimed=True) == "musa"


def test_flagos_tensor_selects_a_registered_native_provider(monkeypatch):
    """Provider target tables bridge torch_fl's PrivateUse1 device names."""
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "musa")
    _provider(monkeypatch, {"musa"}, {"musa": "musa"})
    assert runtime.resolve_target(_device("flagos")) == "musa"
    runtime.register_device_adapter(
        "musa",
        "musa",
        lambda tensor: tensor,
        lambda tensor: tensor,
        lambda index: index,
        lambda index: 0,
    )
    assert runtime.is_tileops_available(_device("flagos"))


def test_explicit_ascend_target_does_not_require_the_upstream_detector(monkeypatch):
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "ascend")
    registry = types.ModuleType("tileops.backend.registry")
    registry.known_targets = lambda: {"ascend"}
    registry.BUILDERS = {("ReluFwdOp", "ascend"): object()}
    monkeypatch.setattr(runtime, "_provider_registry", lambda: registry)
    monkeypatch.setattr(runtime, "import_tileops", lambda: object())
    assert runtime._build_target("ascend") == "ascend"


def test_explicit_metax_target_accepts_current_registry(monkeypatch):
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "metax")
    registry = types.ModuleType("tileops.backend.registry")
    registry.known_targets = lambda: {"tileops_maca"}
    monkeypatch.setattr(runtime, "_provider_registry", lambda: registry)
    monkeypatch.setattr(runtime, "import_tileops", lambda: object())
    monkeypatch.setattr(
        runtime,
        "_provider_backend",
        lambda: pytest.fail("registered MetaX target must not require a detector"),
    )
    assert runtime._build_target("metax") == "tileops_maca"


def test_flagos_call_reuses_the_build_target_after_meta_construction_probe(
    monkeypatch,
):
    """A CUDA-ABI provider must bind its meta probe to the physical flagos call."""
    monkeypatch.setattr(runtime, "resolve_target", lambda device=None: None)
    monkeypatch.setattr(runtime, "_registered_targets", lambda: {"tileops_maca"})
    assert (
        runtime._provider_target(None, _device("flagos"), "tileops_maca")
        == "tileops_maca"
    )


def test_meta_construction_probe_uses_the_generated_build_target(monkeypatch):
    """The dispatch closure must not drop the target when construction is on meta."""
    runtime.register_device_adapter(
        "tileops_maca",
        "cuda",
        lambda tensor: tensor,
        lambda tensor: tensor,
    )
    monkeypatch.setattr(runtime, "resolve_target", lambda device=None: None)
    monkeypatch.setattr(
        runtime, "_build_target", lambda *args, **kwargs: "tileops_maca"
    )
    monkeypatch.setattr(runtime, "_registered_targets", lambda: {"tileops_maca"})
    monkeypatch.setattr(runtime, "_check_registered", lambda op_cls, target: None)

    calls = []

    class Op:
        def __init__(self, *, target=None):
            self.target = target

        def __call__(self, x):
            calls.append(self.target)
            return x

    impl = runtime._unary_impl(
        Op,
        (torch.float32,),
        {},
        "relu",
        runtime.provider_device_object("generic", 0),
    )
    x = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(impl(x), x)
    assert calls == ["tileops_maca"]


def test_registry_discovery_runs_before_target_selection(monkeypatch):
    """Third-party entry points populate targets after import, not before it."""
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "metax")
    targets = set()
    registry = types.ModuleType("tileops.backend.registry")

    def ensure_loaded():
        targets.add("tileops_maca")

    registry.ensure_loaded = ensure_loaded
    registry.known_targets = lambda: set(targets)
    monkeypatch.setitem(sys.modules, "tileops.backend.registry", registry)
    monkeypatch.setattr(runtime, "import_tileops", lambda: object())
    assert runtime._registered_targets() == {"tileops_maca"}
    assert runtime._build_target("metax") == "tileops_maca"


def test_op_resolution_falls_back_to_compact_provider_layout(monkeypatch):
    class AllFwdOp:
        pass

    reduction = types.ModuleType("tileops.ops.reduction")
    reduction.AllFwdOp = AllFwdOp
    monkeypatch.setitem(sys.modules, "tileops.ops.reduction", reduction)
    assert (
        runtime._resolve_op_class("tileops.ops.reduction.all_op", "AllFwdOp")
        is AllFwdOp
    )


def test_op_resolution_keeps_generated_split_module(monkeypatch):
    class AllFwdOp:
        pass

    split = types.ModuleType("tileops.ops.reduction.all_op")
    split.AllFwdOp = AllFwdOp
    monkeypatch.setitem(sys.modules, "tileops.ops.reduction.all_op", split)
    assert (
        runtime._resolve_op_class("tileops.ops.reduction.all_op", "AllFwdOp")
        is AllFwdOp
    )


def test_provider_serving_target_supports_settled_target_name():
    op = types.SimpleNamespace(_settled_target="tileops_maca")
    assert runtime._provider_serving_target(op) == "tileops_maca"


def test_target_comes_from_provider_or_platform(monkeypatch):
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "musa")
    _provider(monkeypatch, {"musa", "cuda"}, {"musa": "musa"})
    runtime.register_device_adapter(
        "musa",
        "musa",
        lambda t: t,
        lambda t: t,
        lambda index: index,
        lambda index: 1000,
    )
    assert runtime.resolve_target(_device("musa")) == "musa"
    assert runtime.is_tileops_available(_device("musa"))


def test_unclaimed_hardware_declines_without_sm90_probe(monkeypatch):
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "musa")
    assert not runtime.is_tileops_available(types.SimpleNamespace(type="musa", index=0))
    monkeypatch.setattr(runtime, "_tileops_platform", lambda: "cuda")
    assert not runtime.is_tileops_available(types.SimpleNamespace(type="cuda", index=0))


def test_musa_current_stream_is_installed_by_adapter(monkeypatch):
    class BaseKernelAdapter:
        @staticmethod
        def get_current_stream_functor(device=None):
            return lambda: 0

    monkeypatch.setattr(
        runtime,
        "_make_stream_functor",
        lambda kind, stream, fallback: lambda device=None: lambda: 1000,
    )
    adapter = types.ModuleType("tilelang.jit.adapter.base")
    adapter.BaseKernelAdapter = BaseKernelAdapter
    monkeypatch.setattr(runtime, "configure_device", _real_configure_device)
    for name in ("tilelang", "tilelang.jit", "tilelang.jit.adapter"):
        monkeypatch.setitem(__import__("sys").modules, name, types.ModuleType(name))
    monkeypatch.setitem(__import__("sys").modules, "tilelang.jit.adapter.base", adapter)
    runtime.register_device_adapter(
        "musa",
        "musa",
        lambda t: t,
        lambda t: t,
        lambda index: index,
        lambda index: 1000,
    )
    runtime.configure_device("musa")
    assert BaseKernelAdapter.get_current_stream_functor(_device("musa", 2))() == 1000


def test_native_adapter_keeps_storage_and_provides_current_stream():
    runtime.register_device_adapter(
        "musa",
        "musa",
        lambda t: t,
        lambda t: t,
        get_device=lambda index: index,
        get_stream=lambda i: 1000 + i,
    )
    tensor = types.SimpleNamespace(device=_device("flagos", 2))
    assert runtime.prepare_tensor(tensor, "musa") is tensor
    assert runtime.restore_tensor(tensor, "musa") is tensor


def test_cache_separates_provider_targets():
    created = []

    class FakeOp:
        def __init__(self, n, dtype, target=None):
            self.target = target
            created.append(self)

    one = runtime.open_provider_op(FakeOp, "musa", 4, torch.float16)
    two = runtime.open_provider_op(FakeOp, "metax", 4, torch.float16)
    again = runtime.open_provider_op(FakeOp, "musa", 4, torch.float16)
    assert one.target == "musa"
    assert two.target == "metax"
    assert again is not one
    assert [op.target for op in created] == ["musa", "metax", "musa"]


def test_legacy_provider_uses_vendor_kernel_map():
    class LegacyOp:
        def __init__(self):
            self.kernel_map = {"relu": "cuda-kernel"}

        @property
        def default_kernel_map(self):
            return {"relu": "cuda-kernel"}

        @staticmethod
        def _maca_kernel_map():
            return {"relu": "maca-kernel"}

    op = runtime.open_provider_op(LegacyOp, "metax")
    assert op.kernel_map == {"relu": "maca-kernel"}


def test_entry_detects_shape_commitment_on_the_op_class():
    """``inspect.signature(instance)`` only exposes ``__call__``, not the ctor."""

    class LegacyOp:
        def __init__(self, N_total, dtype):
            self.N_total = N_total
            self.dtype = dtype

    entry = runtime._Entry(LegacyOp(8, torch.float16), "cuda")

    assert entry.shape_committed is True


def test_shape_committed_entry_prepares_inputs_and_restores_results(monkeypatch):
    """Legacy providers also need the CUDA-ABI boundary on both sides."""

    class LegacyOp:
        def __init__(self, N_total, dtype):
            self.N_total = N_total
            self.dtype = dtype

        def __call__(self, x):
            assert x == "provider-input"
            return "provider-output"

    prepared = types.SimpleNamespace()
    monkeypatch.setattr(
        runtime,
        "prepare_tensor",
        lambda tensor, target: (
            setattr(prepared, "seen", (tensor, target)) or "provider-input"
        ),
    )
    restored = object()
    monkeypatch.setattr(
        runtime,
        "restore_tensor",
        lambda tensor, target: (
            restored
            if (tensor, target) == ("provider-output", "tileops_maca")
            else pytest.fail("provider output did not cross the adapter boundary")
        ),
    )
    entry = runtime._Entry(LegacyOp(8, torch.float16), "tileops_maca")

    result = entry.invoke(object())

    assert prepared.seen[1] == "tileops_maca"
    assert result is restored


def test_failed_provider_construction_falls_back():
    class FakeOp:
        def __init__(self, n, dtype):
            raise ValueError("provider cannot build this kernel")

    impl = runtime._unary_impl(FakeOp, (torch.float32,), {}, "relu")
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(impl(x), torch.relu(x))


def test_new_unary_constructor_omits_committed_size(monkeypatch):
    """Current providers infer dtype and size from the first forward call."""
    calls = []

    class NewUnaryOp:
        def __init__(self, *, inplace=False, target=None):
            calls.append((inplace, target))
            assert target == "cuda"
            assert inplace is False

        def __call__(self, x):
            return torch.relu(x)

    runtime.register_device_adapter(
        "cuda",
        "cuda",
        lambda tensor: tensor,
        lambda tensor: tensor,
    )
    monkeypatch.setattr(runtime, "resolve_target", lambda device=None: "cuda")
    monkeypatch.setattr(runtime, "_check_registered", lambda op_cls, target: None)
    monkeypatch.setattr(runtime, "_make_fallback", lambda *a, **k: torch.relu)
    runtime._HAS_SELECT_TARGET = True
    impl = runtime._unary_impl(
        NewUnaryOp,
        (torch.float32,),
        {"inplace": False},
        "relu",
        device=runtime.provider_device_object("generic", 0),
    )
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(impl(x), torch.relu(x))
    assert calls == [(False, "cuda")]


def test_new_binary_constructor_omits_shapes_and_dtype(monkeypatch):
    calls = []

    class NewBinaryOp:
        def __init__(self, *, alpha=1, target=None):
            calls.append((alpha, target))
            assert target == "cuda"

        def __call__(self, a, b):
            return a + b

    runtime.register_device_adapter(
        "cuda",
        "cuda",
        lambda tensor: tensor,
        lambda tensor: tensor,
    )
    monkeypatch.setattr(runtime, "resolve_target", lambda device=None: "cuda")
    monkeypatch.setattr(runtime, "_check_registered", lambda op_cls, target: None)
    runtime._HAS_SELECT_TARGET = True
    impl = runtime._binary_impl(
        NewBinaryOp,
        (torch.float32,),
        {"alpha": True},
        "add.Tensor",
        device=runtime.provider_device_object("generic", 0),
    )
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    # The dispatcher consumes ``alpha`` while selecting the constructor
    # specialization; the fallback shim reconstructs the aten kwarg.
    torch.testing.assert_close(
        impl(a, a, alpha=2), runtime._aten_op("add.Tensor")(a, a, alpha=2)
    )
    assert calls == [(2, "cuda")]


def test_new_reduce_constructor_omits_dtype(monkeypatch):
    calls = []

    class NewReduceOp:
        def __init__(self, dim, keepdim=False, *, target=None):
            assert target == "cuda"
            calls.append((dim, keepdim, target))

        def __call__(self, x):
            return x.sum(dim=self.dim, keepdim=self.keepdim)

        @property
        def dim(self):
            return calls[-1][0]

        @property
        def keepdim(self):
            return calls[-1][1]

    runtime.register_device_adapter(
        "cuda",
        "cuda",
        lambda tensor: tensor,
        lambda tensor: tensor,
    )
    monkeypatch.setattr(runtime, "resolve_target", lambda device=None: "cuda")
    _provider(monkeypatch, {"cuda"}, {"cuda": "cuda"})
    monkeypatch.setattr(runtime, "_check_registered", lambda op_cls, target: None)
    runtime._HAS_SELECT_TARGET = True
    impl = runtime._reduce_impl(
        NewReduceOp,
        (torch.float32,),
        {"keepdim": True},
        "sum.dim_IntList",
        device=runtime.provider_device_object("generic", 0),
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.testing.assert_close(impl(x, [-1], False), x.sum(dim=-1, keepdim=False))
    assert calls == [(-1, False, "cuda")]


class _TwoInputOp:
    def __init__(self):
        self.target = None
