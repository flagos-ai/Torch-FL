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

"""Unit coverage for ``torch_fl/fast_launch.py`` (the Phase 0.5 replay shim).

The module is loaded by path, like ``test_env_registry.py``, so these tests need
neither torch nor an accelerator: everything device- and framework-specific is
injected through the module's ``Provider`` protocol.  What is pinned here is the
part that produced silent wrong numbers during the measurement campaign:

* slot classification by first appearance (never ``data_ptr`` equality with the
  returned tensor, which fails because FlagGems returns a ``view``),
* intermediates deduplicated per distinct pointer (per-occurrence allocation gave
  ``max_err`` 2312.5 on a 1M-element sum),
* capture keys that include shape/strides/contiguity (cross-shape replay gave
  ``max_err`` 31.38 and corrupted the allocator),
* intermediates persistent and never re-initialised (zeroing cost +30 us for
  nothing),
* output allocated at the result shape instead of ``.view()`` per call,
* replay patching a prebuilt argument list instead of rebuilding it,

and the fallback contract: anything the shim cannot replay keeps calling the
untouched FlagGems callable.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "torch_fl" / "fast_launch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fast_launch_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fl = _load_module()


class FakeTensor:
    """The slice of the tensor API the replay shim touches."""

    _next_ptr = 1000

    def __init__(self, shape, dtype="float32", stride=None, device="fake:0"):
        self.shape = tuple(int(dim) for dim in shape)
        total = 1
        for dim in self.shape:
            total *= dim
        self._storage = max(total, 1)
        self.dtype = dtype
        self.device = device
        self._stride = (
            tuple(stride) if stride is not None else _contiguous_stride(self.shape)
        )
        self.views = 0
        FakeTensor._next_ptr += 1
        self._ptr = FakeTensor._next_ptr
        self._storage_ptr = self._ptr

    def data_ptr(self):
        return self._ptr

    def stride(self):
        return self._stride

    def numel(self):
        return self._storage

    def is_contiguous(self):
        return self._stride == _contiguous_stride(self.shape)

    def view(self, shape):
        self.views += 1
        total = 1
        for dim in shape:
            total *= dim
        assert total == self.numel(), "view must preserve the element count"
        viewed = FakeTensor(shape, self.dtype, device=self.device)
        # A real view shares its base storage, which is how the plan pairs a
        # returned tensor with the buffer the kernel wrote.
        viewed._storage_ptr = self._storage_ptr
        return viewed


class Buffer:
    """Stand-in for FlagGems' ``StridedBuffer`` wrapper argument."""

    def __init__(self, tensor, shape, stride):
        self.tensor = tensor
        self.shape = shape
        self.stride = stride


def _contiguous_stride(shape):
    stride, acc = [], 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


class FakeProvider:
    """Records what the replay path does; the op under test drives it."""

    def __init__(self):
        self.recorded = []
        self.rebuilds = []
        self.empties = []
        self.empty_likes = []

    @property
    def launcher(self):
        def _launch(*args, **kwargs):
            self.recorded.append(
                fl.CapturedLaunch(call=_launch, args=args, kwargs=kwargs)
            )

        return _launch

    def capture(self, call):
        self.recorded.clear()
        result = call()
        return result, list(self.recorded)

    def unwrap(self, arg):
        if isinstance(arg, FakeTensor):
            return arg
        if isinstance(arg, Buffer):
            return arg.tensor
        return None

    def storage_ptr(self, tensor):
        return tensor._storage_ptr

    def rebuild(self, template, tensor):
        self.rebuilds.append(tensor)
        if isinstance(template, Buffer):
            return Buffer(tensor, template.shape, template.stride)
        return tensor

    def empty(self, shape, dtype, device=None):
        tensor = FakeTensor(shape, dtype, device=device or "fake:0")
        self.empties.append(tensor)
        return tensor

    def empty_like(self, template):
        tensor = FakeTensor(
            template.shape,
            template.dtype,
            template.stride(),
            getattr(template, "device", None),
        )
        self.empty_likes.append(tensor)
        return tensor

    def cohort(self):
        return ("fake", 1)


def _capture(provider, op, *args):
    return provider.capture(lambda: op(provider.launcher, *args))


def test_single_launch_classification():
    def op(launch, a, b):
        out = FakeTensor(a.shape, a.dtype)
        launch(a, b, out)
        return out

    provider = FakeProvider()
    a, b = FakeTensor((8, 4)), FakeTensor((8, 4))
    result, launches = _capture(provider, op, a, b)
    plan = fl.build_plan("mul", launches, [a, b], provider, [result])
    assert [slot.kind for slot in plan.launches[0].slots] == ["in", "in", "out"]
    assert plan.launch_count == 1
    assert plan.scratch == {}


def test_scratch_is_deduplicated_per_pointer():
    """The 2-launch reduce: ``mid`` appears twice in one launch, one buffer."""

    def op(launch, x):
        mid = FakeTensor((x.shape[0],))
        out = FakeTensor(())
        launch(x, mid, mid)
        launch(mid, out)
        return out

    provider = FakeProvider()
    x = FakeTensor((16, 8))
    result, launches = _capture(provider, op, x)
    plan = fl.build_plan("sum", launches, [x], provider, [result])
    assert plan.launch_count == 2
    assert len(plan.scratch) == 1, "per-occurrence scratch is the max_err 2312.5 bug"
    assert [slot.kind for slot in plan.launches[0].slots] == [
        "in",
        "scratch",
        "scratch",
    ]
    assert [slot.kind for slot in plan.launches[1].slots] == ["scratch", "out"]
    assert len(provider.empty_likes) == 1


def test_ping_pong_capture_is_unsupported():
    """sort-class: output is not introduced in the last launch."""

    def op(launch, x):
        buf = FakeTensor((x.shape[0],))
        out = FakeTensor(())
        launch(x, buf)
        for _ in range(3):
            launch(buf, buf)
        return out

    provider = FakeProvider()
    x = FakeTensor((8,))
    _, launches = _capture(provider, op, x)
    with pytest.raises(fl.Unsupported):
        fl.build_plan("sort", launches, [x], provider, [FakeTensor(())])


def test_more_launches_than_limit_is_unsupported():
    def op(launch, x):
        out = FakeTensor(())
        for _ in range(fl.MAX_LAUNCHES + 1):
            launch(x, out)
        return out

    provider = FakeProvider()
    x = FakeTensor((8,))
    _, launches = _capture(provider, op, x)
    with pytest.raises(fl.Unsupported):
        fl.build_plan("weird", launches, [x], provider, [FakeTensor(())])


def test_output_allocated_at_result_shape_without_view():
    """sum.dim-class: captured buffer (16,1), result (16,)."""

    def op(launch, x):
        buf = FakeTensor((x.shape[0], 1))
        launch(x, buf)
        return buf.view((x.shape[0],))

    provider = FakeProvider()
    x = FakeTensor((16, 4))
    result, launches = _capture(provider, op, x)
    plan = fl.build_plan("sum.dim", launches, [x], provider, [result])
    assert plan.outs[0].flat is True
    provider.empties.clear()
    provider.rebuilds.clear()
    buffers = plan.replay([x], provider)
    assert [tensor.shape for tensor in provider.empties] == [(16,)]
    assert buffers[0].views == 0, "no per-call .view() on the lean path"
    # both tensor slots are patched: the input and the fresh output
    assert len(provider.rebuilds) == 2
    assert provider.rebuilds[-1] is buffers[0]


def test_scratch_is_persistent_and_never_reinitialised():
    def op(launch, x):
        mid = FakeTensor((x.shape[0],))
        out = FakeTensor(())
        launch(x, mid)
        launch(mid, out)
        return out

    provider = FakeProvider()
    x = FakeTensor((16,))
    result, launches = _capture(provider, op, x)
    plan = fl.build_plan("sum", launches, [x], provider, [result])
    scratch = next(iter(plan.scratch.values()))
    provider.empty_likes.clear()
    plan.replay([x], provider)
    plan.replay([x], provider)
    assert provider.empty_likes == [], "scratch must not be re-allocated or zeroed"
    assert next(iter(plan.scratch.values())) is scratch


def test_replay_reuses_prebuilt_argument_list():
    def op(launch, a):
        out = FakeTensor(a.shape)
        launch(a, out)
        return out

    provider = FakeProvider()
    a = FakeTensor((4,))
    result, launches = _capture(provider, op, a)
    plan = fl.build_plan("neg", launches, [a], provider, [result])
    arg_list = plan.launches[0].arg_list
    plan.replay([a], provider)
    plan.replay([a], provider)
    assert plan.launches[0].arg_list is arg_list, "rule 6: patch, never rebuild"


def test_signature_distinguishes_strides_and_contiguity():
    provider = FakeProvider()
    base = FakeTensor((8, 8))
    transposed = FakeTensor((8, 8), stride=(1, 8))
    contiguous_sig = fl.signature_of("mul", [base], provider)
    transposed_sig = fl.signature_of("mul", [transposed], provider)
    assert contiguous_sig != transposed_sig
    assert contiguous_sig.tensors[0][3] is True
    assert transposed_sig.tensors[0][3] is False


def test_cache_is_invalidated_when_cohort_changes():
    provider = FakeProvider()

    def op(launch, a):
        out = FakeTensor(a.shape)
        launch(a, out)
        return out

    a = FakeTensor((4,))
    result, launches = _capture(provider, op, a)
    signature = fl.signature_of("neg", [a], provider)
    plan = fl.build_plan("neg", launches, [a], provider, [result])
    cache = fl.PlanCache()
    cache.put(signature, plan)
    assert cache.get(signature, plan.cohort) is plan
    assert cache.get(signature, ("different", "cohort")) is None
    assert len(cache) == 0


def test_wrap_callable_captures_once_then_replays():
    calls = {"n": 0}

    def op(launch, a):
        calls["n"] += 1
        out = FakeTensor(a.shape)
        launch(a, out)
        return out

    provider = FakeProvider()

    def original(a):
        return op(provider.launcher, a)

    stats = fl.Stats()
    wrapped = fl.wrap_callable(
        "neg", original, provider=provider, cache=fl.PlanCache(), stats=stats
    )
    tensor = FakeTensor((4,))
    # Warm-first: the cold call runs untouched, because it can still be tuning
    # (libtuner emits ~19 launches on a new layout class).
    first = wrapped(tensor)
    assert first is not None
    assert stats.warmups == 1 and stats.captures == 0
    assert calls["n"] == 1
    wrapped(tensor)  # now steady-state: capture one launch
    assert stats.captures == 1
    assert calls["n"] == 2
    wrapped(tensor)  # replay
    assert stats.hits == 1
    assert calls["n"] == 2, "the third call replays instead of running FlagGems"


def test_wrap_callable_falls_back_for_unsupported_ops():
    calls = {"n": 0}

    def original(a):
        calls["n"] += 1
        buf = FakeTensor((a.shape[0],))
        out = FakeTensor(())
        provider.launcher(a, buf)
        provider.launcher(buf, buf)
        return out

    provider = FakeProvider()
    stats = fl.Stats()
    wrapped = fl.wrap_callable(
        "sort", original, provider=provider, cache=fl.PlanCache(), stats=stats
    )
    tensor = FakeTensor((4,))
    wrapped(tensor)  # warm (untouched)
    wrapped(tensor)  # capture fails to classify -> fallback for this call
    wrapped(tensor)  # op is blacklisted -> untouched callable
    assert calls["n"] == 3, "unsupported ops keep calling the untouched callable"
    assert stats.fallbacks == 1


def test_scalar_keywords_are_captured_and_keyed(monkeypatch):
    """100 generated kernels pass ``alpha``/``dtype`` as keywords; they must not opt out.

    The scalar is part of the cache key, so a different value captures a different
    plan instead of replaying a launch that baked in the old one.
    """

    calls = {"n": 0}

    def original(a, b, *, alpha=1):
        calls["n"] += 1
        out = FakeTensor(a.shape)
        provider.launcher(a, b, out)
        return out

    provider = FakeProvider()
    stats = fl.Stats()
    cache = fl.PlanCache()
    wrapped = fl.wrap_callable(
        "add", original, provider=provider, cache=cache, stats=stats
    )

    a, b = FakeTensor((4,)), FakeTensor((4,))
    wrapped(a, b, alpha=2)  # warm
    wrapped(a, b, alpha=2)  # capture
    assert stats.captures == 1
    wrapped(a, b, alpha=2)  # replay
    assert stats.hits == 1
    assert calls["n"] == 2

    wrapped(a, b, alpha=3)  # different scalar -> its own capture, never a replay
    assert stats.captures == 2
    assert calls["n"] == 3


def test_tensor_keyword_falls_back():
    """``out=`` carries storage that moves between calls, so it opts out."""

    def original(a, out=None):
        return out

    provider = FakeProvider()
    stats = fl.Stats()
    wrapped = fl.wrap_callable(
        "mul_out", original, provider=provider, cache=fl.PlanCache(), stats=stats
    )
    target = FakeTensor((4,))
    wrapped(FakeTensor((4,)), out=target)
    assert stats.warmups == 0 and stats.captures == 0


def test_install_replaces_and_returns_original():
    class Module:
        pass

    module = Module()
    module.op = lambda value: value * 2
    original = module.op
    replacement = fl.install(module, "op", lambda value: value + 1)
    assert replacement is original
    assert module.op(1) == 2


def test_contiguous_output_is_allocated_at_result_shape():
    """mul-class: the captured output is contiguous, so the flat path applies."""

    def op(launch, a):
        out = FakeTensor(a.shape)  # contiguous, like mul's torch.empty(shape)
        launch(a, out)
        return out

    provider = FakeProvider()
    a = FakeTensor((4, 6))
    result, launches = _capture(provider, op, a)
    plan = fl.build_plan("mul", launches, [a], provider, [result])
    assert plan.outs[0].contiguous is True
    assert plan.outs[0].flat is True
    provider.empties.clear()
    provider.empty_likes.clear()
    plan.replay([a], provider)
    assert provider.empties[0].shape == (4, 6)
    assert provider.empty_likes == [], "contiguous case must not clone a template"


def test_noncontiguous_output_replays_the_captured_layout():
    """pointwise-class on a transposed input: empty_like(input) layout is captured."""

    def op(launch, a):
        out = FakeTensor(a.shape, stride=a.stride())  # preserves the input layout
        launch(a, out)
        return out

    provider = FakeProvider()
    a = FakeTensor((4, 6), stride=(1, 4))  # transposed view
    result, launches = _capture(provider, op, a)
    plan = fl.build_plan("add", launches, [a], provider, [result])
    assert plan.outs[0].contiguous is False
    assert plan.outs[0].flat is False
    provider.empty_likes.clear()
    buffers = plan.replay([a], provider)
    assert len(provider.empty_likes) == 1, "layout must be replayed, not assumed"
    assert buffers[0] is not None


def test_two_shapes_capture_twice_instead_of_reusing():
    provider = FakeProvider()
    stats = fl.Stats()
    shapes = []

    def op(launch, a):
        shapes.append(tuple(a.shape))
        out = FakeTensor(a.shape)
        launch(a, out)
        return out

    def original(a):
        return op(provider.launcher, a)

    wrapped = fl.wrap_callable(
        "neg", original, provider=provider, cache=fl.PlanCache(), stats=stats
    )
    wrapped(FakeTensor((4,)))  # warm
    wrapped(FakeTensor((4,)))  # capture shape 4
    wrapped(FakeTensor((8,)))  # different signature -> its own capture, never a replay
    assert stats.captures == 2
    wrapped(FakeTensor((4,)))
    assert stats.hits == 1, "the first shape is served from cache"


def test_aliased_inputs_are_not_collapsed_by_pointer():
    """``f(a, a)``: both slots must map to their own input index."""

    def op(launch, a, b):
        out = FakeTensor(a.shape)
        launch(a, b, out)
        return out

    provider = FakeProvider()
    a = FakeTensor((4,))
    result, launches = _capture(provider, op, a, a)
    plan = fl.build_plan("add", launches, [a, a], provider, [result])
    kinds = [slot.kind for slot in plan.launches[0].slots]
    assert kinds == ["in", "in", "out"]
    # first occurrence wins: both slots patch the same input object
    assert {slot.key for slot in plan.launches[0].slots if slot.kind == "in"} == {0}


def test_zero_element_tensor_is_refused_not_misclassified():
    """Every empty tensor reports data_ptr() == 0, so pointers cannot identify them."""

    class ZeroTensor(FakeTensor):
        def data_ptr(self):
            return 0

    def op(launch, a, b):
        out = ZeroTensor(())
        launch(a, b, out)
        return out

    provider = FakeProvider()
    zero_a, zero_b = ZeroTensor((0,)), ZeroTensor((0,))
    result, launches = _capture(provider, op, zero_a, zero_b)
    with pytest.raises(fl.Unsupported):
        fl.build_plan("add", launches, [zero_a, zero_b], provider, [result])


def test_multi_output_keeps_the_tuple_container():
    """``max.dim`` returns ``(values, indices)``; replay must not return a list."""

    def op(launch, a):
        values = FakeTensor(())
        indices = FakeTensor(())
        launch(a, values, indices)
        return values, indices

    provider = FakeProvider()
    a = FakeTensor((4, 4))
    result, launches = _capture(provider, op, a)
    assert isinstance(result, tuple)
    plan = fl.build_plan(
        "max.dim",
        launches,
        [a],
        provider,
        list(result),
        returns_tuple=True,
    )
    replayed = plan.replay([a], provider)
    assert isinstance(replayed, tuple)
    assert len(replayed) == 2


def test_replay_failure_falls_back_instead_of_raising():
    def op(launch, a):
        out = FakeTensor((a.shape[0], 1))
        launch(a, out)
        return out.view((a.shape[0],))

    provider = FakeProvider()
    a = FakeTensor((4,))
    result, launches = _capture(provider, op, a)
    plan = fl.build_plan("sum.dim", launches, [a], provider, [result])

    class ExplodingProvider(FakeProvider):
        def empty(self, shape, dtype, device=None):
            raise RuntimeError("allocator said no")

    stats = fl.Stats()
    cache = fl.PlanCache()
    signature = fl.signature_of("sum.dim", [a], provider)
    cache.put(signature, plan)
    calls = {"n": 0}

    def original(arg):
        calls["n"] += 1
        return "original"

    wrapped = fl.wrap_callable(
        "sum.dim", original, provider=ExplodingProvider(), cache=cache, stats=stats
    )
    wrapped(a)  # warm
    assert wrapped(a) == "original", "a broken replay must fall back, not raise"
    assert calls["n"] == 2
    assert stats.fallbacks == 1


def test_multi_output_order_survives_pointer_order():
    """``max.dim``: pointer order and return order are independent - do not pair them.

    ``indices`` is allocated FIRST (lowest pointer) but returned SECOND, so any
    implementation that pairs ``sorted(data_ptr)`` with the result list hands the
    caller indices where values belong - int64 for float32, no exception, and
    ``max_err`` checks on a correct-order arm cannot see it.
    """

    def op(launch, a):
        indices = FakeTensor((4, 4), dtype="int64")  # lower ptr, returned SECOND
        values = FakeTensor((4, 4), dtype="float32")  # higher ptr, returned FIRST
        launch(a, values, indices)
        return values, indices

    provider = FakeProvider()
    a = FakeTensor((4, 4))
    result, launches = _capture(provider, op, a)
    assert [tensor.dtype for tensor in result] == ["float32", "int64"]
    plan = fl.build_plan(
        "max.dim", launches, [a], provider, list(result), returns_tuple=True
    )
    assert [spec.rank for spec in plan.outs] == [0, 1]
    replayed = plan.replay([a], provider)
    assert isinstance(replayed, tuple)
    assert [tensor.dtype for tensor in replayed] == ["float32", "int64"], (
        "outputs must be returned in call order, not pointer order"
    )


def test_multi_output_with_different_shapes_is_paired_correctly():
    """Unequal element counts: a swapped pairing raises instead of returning wrong data."""

    def op(launch, a):
        indices = FakeTensor((2, 3), dtype="int64")  # lower ptr, returned second
        values = FakeTensor((4, 4), dtype="float32")  # higher ptr, returned first
        launch(a, values, indices)
        return values, indices

    provider = FakeProvider()
    a = FakeTensor((4, 4))
    result, launches = _capture(provider, op, a)
    plan = fl.build_plan(
        "topk", launches, [a], provider, list(result), returns_tuple=True
    )
    assert [tuple(spec.result_shape) for spec in plan.outs] == [(4, 4), (2, 3)]
    replayed = plan.replay([a], provider)
    assert [tuple(tensor.shape) for tensor in replayed] == [(4, 4), (2, 3)]


def test_ambiguous_output_pairing_is_refused():
    """Two returned tensors sharing one storage cannot be paired to slots."""

    def op(launch, a):
        buf = FakeTensor((4,))
        launch(a, buf, buf)  # one buffer used as both outputs
        return buf, buf.view((4,))

    provider = FakeProvider()
    a = FakeTensor((4,))
    result, launches = _capture(provider, op, a)
    with pytest.raises(fl.Unsupported):
        fl.build_plan(
            "weird", launches, [a], provider, list(result), returns_tuple=True
        )


def test_plans_reports_the_slot_census():
    """The integration assertion reads this: an output slot must exist and be ``out``."""

    def op(launch, a):
        out = FakeTensor(a.shape)
        launch(a, out)
        return out

    provider = FakeProvider()
    a = FakeTensor((4,))
    result, launches = _capture(provider, op, a)
    plan = fl.build_plan("neg", launches, [a], provider, [result])
    cache = fl.PlanCache()
    cache.put(fl.signature_of("neg", [a], provider), plan)

    described = fl.plans(cache)
    assert len(described) == 1
    entry = described[0]
    assert entry["op"] == "neg"
    assert entry["launch_count"] == 1
    assert entry["launch_arg_kinds"] == [["in", "out"]], "never ['in', 'scratch']"
    assert entry["outputs"] == 1
    assert entry["output_ranks"] == [0]
    assert entry["scratch_buffers"] == 0


def test_enable_replaces_both_package_and_submodule_bindings(monkeypatch):
    """The C++ bridge resolves ``flag_gems.<op>``; tests call ``flag_gems.ops.<op>``.

    They start as the same function object but are two module-level bindings, so
    replacing one leaves the other pointing at the original - the shim then looks
    installed and never runs on the routed path (caught on H800/H100 hardware, not
    by a unit test that only inspects one namespace).
    """
    import types

    def op_impl(a):
        return a

    package = types.ModuleType("fake_flag_gems")
    submodule = types.ModuleType("fake_flag_gems.ops")
    package.op = op_impl
    submodule.op = op_impl  # initially the same object, like the real package
    monkeypatch.setitem(sys.modules, "fake_flag_gems", package)
    monkeypatch.setitem(sys.modules, "fake_flag_gems.ops", submodule)

    installed = {}
    monkeypatch.setattr(fl, "_INSTALLED", installed, raising=False)
    monkeypatch.setattr(fl, "_PROVIDER", None, raising=False)
    monkeypatch.setattr(fl, "_CACHE", fl.PlanCache(), raising=False)
    monkeypatch.setattr(fl, "_STATS", fl.Stats(), raising=False)
    monkeypatch.setattr(fl, "_UNSUPPORTED", set(), raising=False)

    provider = FakeProvider()
    monkeypatch.setattr(fl, "FlagGemsProvider", lambda: provider)

    fl.enable(["op"], package=package)
    assert package.op is not op_impl, (
        "package binding must be replaced for the C++ route"
    )
    assert submodule.op is not op_impl, "submodule binding must be replaced too"
    assert package.op is submodule.op, "both namespaces should resolve to one wrapper"

    fl.disable()
    assert package.op is op_impl, "disable must restore the untouched callable"
    assert submodule.op is op_impl


def test_input_alias_sharing_storage_is_classified_as_input():
    """FlagGems hands the kernel a normalized view of the input, not the caller's object.

    Measured on ``max_dim``: the captured launch received a distinct tensor with the
    input's storage, shape and strides, so identity-only classification called it an
    output and the plan reported "3 output slots for 2 results".
    """

    def op(launch, a):
        alias = FakeTensor(a.shape, a.dtype, stride=a.stride())
        alias._storage_ptr = a._storage_ptr  # same storage, different object
        values = FakeTensor((a.shape[0],))
        indices = FakeTensor((a.shape[0],), dtype="int64")
        launch(alias, values, indices)
        return values, indices

    provider = FakeProvider()
    a = FakeTensor((4, 4))
    result, launches = _capture(provider, op, a)
    plan = fl.build_plan(
        "max_dim", launches, [a], provider, list(result), returns_tuple=True
    )
    kinds = [slot.kind for slot in plan.launches[0].slots]
    assert kinds == ["in", "out", "out"]
    assert [spec.rank for spec in plan.outs] == [0, 1]


def test_reshaped_view_of_input_is_refused():
    """A view with a different shape cannot be recreated around a new input."""

    def op(launch, a):
        flat = FakeTensor((4 * 4,))
        flat._storage_ptr = a._storage_ptr  # same storage, different shape
        out = FakeTensor(())
        launch(flat, out)
        return out

    provider = FakeProvider()
    a = FakeTensor((4, 4))
    result, launches = _capture(provider, op, a)
    with pytest.raises(fl.Unsupported):
        fl.build_plan("weird", launches, [a], provider, [result])
