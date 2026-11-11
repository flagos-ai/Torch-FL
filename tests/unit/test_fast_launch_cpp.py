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

"""Unit coverage for ``benchmarks/fast_launch_cpp/fast_launch_cpp.py``.

Loaded by path, like ``test_fast_launch.py``, so these tests need neither torch nor an
accelerator: the Triton capture is replaced by a fake provider and the compiled kernel by
a fake object with a ``ptx`` attribute.

What is pinned here is the part that has no runtime safety net. A wrong argument layout
does not raise; it launches the kernel with a shifted argument list and returns a
plausible wrong answer, so every refusal and every mapping rule is asserted:

* the PTX parameter list is the authority, and the two Triton scratch pointers are part
  of it even when their size is 0,
* constexprs the launcher forwards (a dtype string, a tile size, a bool) are not kernel
  parameters and must be dropped,
* the replay must return the pooled result and must not call the original callable, or
  the fast path would only add work on top of the wrapper,
* anything the layout cannot express keeps calling the untouched callable.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "fast_launch_cpp"
    / "fast_launch_cpp.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "fast_launch_cpp_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fast_launch_cpp = _load()


# The PTX entry of a pointwise Triton kernel on this Triton version: three pointers,
# one 32-bit scalar, then the global and profile scratch pointers.
PTX_POINTWISE = """
.visible .entry mul_kernel(
	.param .u64 .ptr .global .align 1 mul_kernel_param_0,
	.param .u64 .ptr .global .align 1 mul_kernel_param_1,
	.param .u64 .ptr .global .align 1 mul_kernel_param_2,
	.param .u32 mul_kernel_param_3,
	.param .u64 .ptr .global .align 1 mul_kernel_param_4,
	.param .u64 .ptr .global .align 1 mul_kernel_param_5
)
{
	.reg .b32 %r<5>;
	ret;
}
"""


class Tensor:
    """Stand-in for a torch tensor: only storage identity and dtype are needed."""

    def __init__(self, storage_id, dtype="torch.float32", shape=(1024,)):
        self._storage_id = storage_id
        self.dtype = dtype
        self.shape = shape

    def untyped_storage(self):
        return types.SimpleNamespace(data_ptr=lambda: self._storage_id)

    def is_contiguous(self):
        return True

    def numel(self):
        total = 1
        for extent in self.shape:
            total *= extent
        return total


class StridedBuffer:
    """Stand-in for a FlagGems strided operand wrapper."""

    def __init__(self, base=None):
        self._base = base


def launch_args(pointers, scalars, constexprs=()):
    """A launcher argument tuple: nine header entries, then kernel arguments."""
    header = [1024, 1, 1, 0, 0, (4, 1, 0), None, None, None]
    return tuple(header + list(pointers) + list(scalars) + list(constexprs))


# -- PTX parsing ---------------------------------------------------------------


def test_ptx_parameter_list_includes_both_scratch_pointers():
    assert fast_launch_cpp.parse_ptx_params(PTX_POINTWISE) == [
        "ptr",
        "ptr",
        "ptr",
        "i32",
        "ptr",
        "ptr",
    ]


def test_ptx_parsing_rejects_a_kernel_without_an_entry():
    with pytest.raises(fast_launch_cpp.Unsupported):
        fast_launch_cpp.parse_ptx_params("// no entry here\n")


def test_ptx_parsing_rejects_an_unexpected_parameter_type():
    ptx = """
.visible .entry odd_kernel(
	.param .f32 odd_kernel_param_0
)
{
	ret;
}
"""
    with pytest.raises(fast_launch_cpp.Unsupported):
        fast_launch_cpp.parse_ptx_params(ptx)


# -- argument mapping ----------------------------------------------------------


def test_mapping_drops_constexprs_and_fills_scratch_pointers():
    """The shape that made an earlier prototype fail: five launch ints for one param."""
    a, b, out = Tensor(1), Tensor(2), Tensor(3)
    ptx_kinds = fast_launch_cpp.parse_ptx_params(PTX_POINTWISE)
    args = launch_args(
        pointers=[a, b, out],
        scalars=[1048576, 1024],  # element count, then the BLOCK constexpr
        constexprs=["float32", True],  # dtype string and a bool flag
    )
    grid, kinds, ints = fast_launch_cpp.classify_arguments(args, [a, b], ptx_kinds)
    assert grid == 1024
    assert kinds == ["in", "in", "out", "i32", "zero", "zero"]
    assert ints == [1048576]


def test_mapping_follows_the_kernel_order_not_the_call_order():
    """A reduction that takes its output first must keep that order."""
    a, out = Tensor(1), Tensor(2)
    ptx_kinds = ["ptr", "ptr", "i32", "ptr", "ptr"]
    grid, kinds, ints = fast_launch_cpp.classify_arguments(
        launch_args(pointers=[out, a], scalars=[1024], constexprs=[512]),
        [a],
        ptx_kinds,
    )
    assert grid == 1024
    assert kinds == ["out", "in", "i32", "zero", "zero"]
    assert ints == [1024]


def test_mapping_refuses_a_strided_operand_that_is_not_contiguous():
    class NonContiguous(Tensor):
        def is_contiguous(self):
            return False

    ptx_kinds = fast_launch_cpp.parse_ptx_params(PTX_POINTWISE)
    args = launch_args(
        pointers=[StridedBuffer(NonContiguous(1)), Tensor(2), Tensor(3)],
        scalars=[1024],
        constexprs=[],
    )
    with pytest.raises(fast_launch_cpp.Unsupported, match="not contiguous"):
        fast_launch_cpp.classify_arguments(args, [Tensor(2)], ptx_kinds)


def test_mapping_accepts_a_contiguous_strided_operand():
    """A contiguous operand's wrapper is equivalent to its own pointer."""

    class Contiguous(Tensor):
        def is_contiguous(self):
            return True

    inner = Contiguous(1)
    ptx_kinds = fast_launch_cpp.parse_ptx_params(PTX_POINTWISE)
    args = launch_args(
        pointers=[StridedBuffer(inner), Tensor(2), Tensor(3)],
        scalars=[1024],
        constexprs=[],
    )
    grid, kinds, ints = fast_launch_cpp.classify_arguments(
        args, [inner, Tensor(2)], ptx_kinds
    )
    assert grid == 1024
    assert kinds == ["in", "in", "out", "i32", "zero", "zero"]


def test_mapping_refuses_a_scalar_count_mismatch():
    ptx_kinds = fast_launch_cpp.parse_ptx_params(PTX_POINTWISE)
    a, b, out = Tensor(1), Tensor(2), Tensor(3)
    args = launch_args(pointers=[a, b, out], scalars=[], constexprs=["float32"])
    with pytest.raises(fast_launch_cpp.Unsupported, match="scalar arguments"):
        fast_launch_cpp.classify_arguments(args, [a, b], ptx_kinds)


def test_mapping_refuses_when_no_input_is_recognised():
    ptx_kinds = fast_launch_cpp.parse_ptx_params(PTX_POINTWISE)
    a, b, out = Tensor(1), Tensor(2), Tensor(3)
    args = launch_args(pointers=[a, b, out], scalars=[1024], constexprs=[])
    with pytest.raises(fast_launch_cpp.Unsupported, match="no input"):
        fast_launch_cpp.classify_arguments(args, [Tensor(99)], ptx_kinds)


# -- plan census ---------------------------------------------------------------


def test_census_reports_the_layout_it_claims():
    plan = fast_launch_cpp.Plan(
        op="mul",
        grid=1024,
        threads=128,
        shared=0,
        kinds=["in", "in", "out", "i32", "zero", "zero"],
        ints=[1048576],
        shape=(1024, 1024),
        dtype="torch.float32",
    )
    census = plan.census()
    assert census["launches"] == 1
    assert census["outputs"] == 1
    assert census["scratch_buffers"] == 2
    assert census["launch_arg_kinds"] == [["in", "in", "out", "i32", "zero", "zero"]]


# -- replay contract -----------------------------------------------------------


class FakeExtension:
    def __init__(self):
        self.registered = []
        self.calls = 0
        self.result = "pooled-result"

    def register_kernel(self, *args):
        self.registered.append(args)

    def launch(self, key, inputs):
        self.calls += 1
        return self.result


class FakeCompiled:
    name = "mul_kernel"
    asm = {"cubin": b"cubin", "ptx": PTX_POINTWISE}
    metadata = types.SimpleNamespace(num_warps=4, shared=0)


class FakeCaptured:
    def __init__(self, launches, results, returns_tuple=False):
        self.launches = launches
        self.compiled = FakeCompiled()
        self.results = results
        self.returns_tuple = returns_tuple

    def close(self):
        pass


class FakeProvider:
    """Calls the operator and reports it, like the real capture does."""

    def __init__(self, captured):
        self.captured = captured
        self.installed = False

    def install(self):
        self.installed = True

    def uninstall(self):
        self.installed = False

    def capture(self, callable_, args):
        self.captured.results[0] = callable_(*args)
        return self.captured


class FakePackage:
    def __init__(self, name, attribute):
        setattr(self, attribute, name)


def make_backend(captured):
    return fast_launch_cpp.Backend(FakeExtension(), FakeProvider(captured))


def test_replay_returns_the_pooled_result_and_does_not_call_the_original():
    """The regression that matters: a replay that also calls the original is pure cost."""
    a, b, out = Tensor(1), Tensor(2), Tensor(3)
    captured = FakeCaptured(
        launches=[launch_args([a, b, out], [1024], ["float32"])], results=[Tensor(4)]
    )
    backend = make_backend(captured)
    calls = []
    wrapper_result = Tensor(9)

    def original(*args, **kwargs):
        calls.append(args)
        return wrapper_result

    wrapped = backend.wrap("mul", original)
    assert wrapped(a, b) is wrapper_result  # cold: compiles and autotunes
    assert wrapped(a, b) is wrapper_result  # warm: captured, still the original
    assert backend.plans["mul"].kinds == ["in", "in", "out", "i32", "zero", "zero"]

    assert wrapped(a, b) == "pooled-result"  # ready: replayed from C++
    assert len(calls) == 2
    assert backend.extension.calls == 1
    assert backend.stats["hits"] == 1


def test_replay_falls_back_when_the_launch_fails():
    a, b, out = Tensor(1), Tensor(2), Tensor(3)
    captured = FakeCaptured(
        launches=[launch_args([a, b, out], [1024], ["float32"])], results=[Tensor(4)]
    )
    backend = make_backend(captured)

    def boom(self, key, inputs):
        raise RuntimeError("CUDA_ERROR_INVALID_VALUE")

    backend.extension.launch = types.MethodType(boom, backend.extension)
    wrapper_result = Tensor(7)

    def original(*args, **kwargs):
        return wrapper_result

    wrapped = backend.wrap("mul", original)
    assert wrapped(a, b) is wrapper_result
    assert wrapped(a, b) is wrapper_result
    assert wrapped(a, b) is wrapper_result  # falls back instead of raising
    assert backend.stats["fallbacks"] == 1
    assert "replay failed" in backend.reasons["mul"]


def test_multi_launch_operator_is_declined():
    a, b, out = Tensor(1), Tensor(2), Tensor(3)
    captured = FakeCaptured(
        launches=[
            launch_args([a, b, out], [1024], ["float32"]),
            launch_args([a, b, out], [1024], ["float32"]),
        ],
        results=[Tensor(4)],
    )
    backend = make_backend(captured)

    def original(*args, **kwargs):
        return "wrapper-result"

    wrapped = backend.wrap("sum", original)
    wrapped(a, b)
    assert wrapped(a, b) == "wrapper-result"
    assert "launches per call" in backend.reasons["sum"]
    assert "sum" not in backend.plans


def test_multi_output_operator_is_declined():
    a, b, out = Tensor(1), Tensor(2), Tensor(3)
    captured = FakeCaptured(
        launches=[launch_args([a, b, out], [1024], ["float32"])],
        results=[Tensor(4), Tensor(5)],
        returns_tuple=True,
    )
    backend = make_backend(captured)

    def original(*args, **kwargs):
        return ("wrapper-result",)

    wrapped = backend.wrap("max_dim", original)
    wrapped(a, b)
    assert wrapped(a, b) == ("wrapper-result",)
    assert "multi-output" in backend.reasons["max_dim"]


def test_unverified_operator_is_declined_and_keeps_the_binding():
    """A layout that has not been proven must never replace the operator."""
    calls = []

    def original(*args, **kwargs):
        calls.append(args)
        return "wrapper-result"

    package = FakePackage(original, "relu")
    backend = make_backend(FakeCaptured(launches=[], results=[Tensor(4)]))
    backend.enable(["relu"], package)
    assert package.relu is original
    assert backend.reasons["relu"] == "operator is not in the verified set"
    assert backend.plans == {}


def test_disabled_operator_keeps_the_original_binding():
    calls = []

    def original(*args, **kwargs):
        calls.append(args)
        return "wrapper-result"

    package = FakePackage(original, "mul")
    captured = FakeCaptured(
        launches=[launch_args([Tensor(1), Tensor(2), Tensor(3)], [1024], ["float32"])],
        results=[Tensor(4)],
    )
    backend = make_backend(captured)
    backend.enable(["mul"], package)
    assert package.mul is not original
    package.mul(Tensor(1), Tensor(2))
    backend.disable()
    assert package.mul is original
