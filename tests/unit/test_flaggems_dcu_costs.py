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

"""Unit coverage for the DCU FlagGems cost patches in ``torch_fl.flagos``.

``_patch_flaggems_pow_scalar_square``, ``_patch_flaggems_mean_last_dim`` and
``_prewarm_triton_key`` all replace one backend-resolved callable with a cheaper
one, and in every case the replacement is only correct inside a narrow envelope
-- an exponent of exactly 2, a contiguous last-axis row mean, a DCU build on a
conf that routes to FlagGems. Outside it the *original* has to be reached, and
nothing about the wrong branch is loud: the wrong branch of the first two is a
different answer, not an exception, and the wrong branch of the third is a
thread that runs on a platform nobody measured.

So these tests cover the decision rather than the arithmetic. They run on any
host: the flag_gems module those wrappers call into is a stub in ``sys.modules``,
and the triton key is a function they never call. That is deliberate -- the host
python segfaults when this tree's ``torch`` and the real ``flag_gems`` are
imported into one process, and none of this needs either to be real.

The on-hardware measurements behind the patches, and what each one is worth, are
in ``torch_fl/flagos/__init__.py`` at each patch and summarised in
``docs/reference/operator-support.md``.
"""

import contextlib
import sys
import types

import pytest
import torch

from torch_fl import flagos


SENTINEL = -12345.0


# ---------------------------------------------------------------------------
# _rebind_flag_gems_name
# ---------------------------------------------------------------------------


class TestRebindFlagGemsName:
    """The sweep has to find every holder of a name and nothing else.

    ``flag_gems`` re-exports each op at the top level and in ``flag_gems.ops``,
    and the C++ bridge resolves one of those strings once and caches what it
    finds, so a rebind that misses a re-export leaves the bridge calling the
    original. The failure is silent in both directions: a miss keeps the old
    callable, and an over-eager match would patch an unrelated function that
    happens to share the name.
    """

    @staticmethod
    def _install(monkeypatch, *modules):
        for mod in modules:
            monkeypatch.setitem(sys.modules, mod.__name__, mod)

    def test_every_holder_moves(self, monkeypatch):
        def original(*args, **kwargs):
            return "original"

        def wrapper(*args, **kwargs):
            return "wrapper"

        holders = [types.ModuleType(f"_torch_fl_holder_{i}") for i in range(3)]
        for mod in holders:
            mod.target = original
        self._install(monkeypatch, *holders)

        flagos._rebind_flag_gems_name("target", original, wrapper)

        assert all(mod.target is wrapper for mod in holders)

    def test_a_different_callable_under_the_same_name_is_left_alone(self, monkeypatch):
        def original(*args, **kwargs):
            return "original"

        def wrapper(*args, **kwargs):
            return "wrapper"

        def other(*args, **kwargs):
            return "other"

        holder = types.ModuleType("_torch_fl_holder_same")
        holder.target = original
        decoy = types.ModuleType("_torch_fl_holder_decoy")
        decoy.target = other
        self._install(monkeypatch, holder, decoy)

        flagos._rebind_flag_gems_name("target", original, wrapper)

        assert holder.target is wrapper
        assert decoy.target is other

    def test_a_module_that_computes_the_name_is_not_touched(self, monkeypatch):
        """``__dict__`` membership, not ``getattr``.

        ``transformers`` ships a ``_LazyModule`` per model, and ``getattr`` on
        one of those imports the submodule behind the name. Asking those modules
        costs seconds of load time and materialises modules nothing asked for,
        and a module that only computes the name was never holding the reference
        that has to move.
        """

        def original(*args, **kwargs):
            return "original"

        def wrapper(*args, **kwargs):
            return "wrapper"

        asked = []
        lazy = types.ModuleType("_torch_fl_holder_lazy")

        def _getattr(name):
            asked.append(name)
            return original

        lazy.__getattr__ = _getattr
        self._install(monkeypatch, lazy)

        flagos._rebind_flag_gems_name("target", original, wrapper)

        assert asked == []

    def test_an_entry_that_cannot_be_read_is_skipped(self, monkeypatch):
        """``sys.modules`` is not all modules, and a bad entry is not fatal.

        A ``None`` entry is what a half-finished import leaves behind, and it has
        no ``__dict__`` at all. The sweep has to step over it and keep going: the
        alternative is that the module holding the op is the one after it.
        """

        def original(*args, **kwargs):
            return "original"

        def wrapper(*args, **kwargs):
            return "wrapper"

        holder = types.ModuleType("_torch_fl_holder_after")
        holder.target = original
        monkeypatch.setitem(sys.modules, "_torch_fl_holder_none", None)
        self._install(monkeypatch, holder)

        flagos._rebind_flag_gems_name("target", original, wrapper)

        assert holder.target is wrapper

    def test_a_name_nothing_holds_is_a_no_op(self, monkeypatch):
        holder = types.ModuleType("_torch_fl_holder_unrelated")
        self._install(monkeypatch, holder)

        flagos._rebind_flag_gems_name(
            "target", lambda *a, **k: None, lambda *a, **k: None
        )

        assert not hasattr(holder, "target")


# ---------------------------------------------------------------------------
# pow.Tensor_Scalar
# ---------------------------------------------------------------------------


@pytest.fixture
def square_calls(monkeypatch):
    """A stub ``flag_gems`` whose ``mul`` records its calls."""
    calls = []

    def mul(a, b):
        calls.append((a, b))
        return "mul"

    stub = types.ModuleType("flag_gems")
    stub.mul = mul
    monkeypatch.setitem(sys.modules, "flag_gems", stub)
    return calls


def _original_pow(tag="pow"):
    def pow_tensor_scalar(A, exponent):
        return tag

    return pow_tensor_scalar


class TestPowScalarSquare:
    """``x.pow(2)`` becomes ``x * x``, and only ``x.pow(2)`` does."""

    @pytest.mark.parametrize("exponent", [2, 2.0])
    def test_the_exponent_is_read_off_any_scalar_it_can_be_converted_to(
        self, exponent, square_calls
    ):
        assert flagos._pow_exponent_is_two(exponent) is True

    @pytest.mark.parametrize(
        "exponent",
        [2.5, 3, 1, 0, -2, "two", None, [2], {"two": 2}, object()],
        ids=["2.5", "3", "1", "0", "-2", "str", "none", "list", "dict", "object"],
    )
    def test_anything_that_is_not_two_is_refused(self, exponent):
        """The exponent is an aten ``Scalar``, so it can be anything at all.

        Converting is the only way to ask the question, and a value that cannot
        be converted is not 2 -- it must not raise out of the wrapper, which
        runs on the dispatch path.
        """
        assert flagos._pow_exponent_is_two(exponent) is False

    def test_an_integer_exponent_of_two_is_a_multiply(self, square_calls):
        """``pow.Tensor_Scalar`` passes an int for ``x.pow(2)``."""
        wrapped = flagos._flaggems_pow_scalar_square_wrapper(_original_pow(), None)
        x = torch.randn(4, 8)

        assert wrapped(x, 2) == "mul"
        assert square_calls == [(x, x)], "the multiply is not x * x"

    def test_a_float_exponent_of_two_is_a_multiply(self, square_calls):
        wrapped = flagos._flaggems_pow_scalar_square_wrapper(_original_pow(), None)

        assert wrapped(torch.randn(4, 8), 2.0) == "mul"
        assert len(square_calls) == 1

    @pytest.mark.parametrize("exponent", [3, 2.5, 0.5, -1])
    def test_every_other_exponent_reaches_the_original(self, exponent, square_calls):
        wrapped = flagos._flaggems_pow_scalar_square_wrapper(_original_pow(), None)

        assert wrapped(torch.randn(4, 8), exponent) == "pow"
        assert square_calls == []

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64, torch.bool], ids=["int32", "int64", "bool"]
    )
    def test_a_non_floating_input_reaches_the_original(self, dtype, square_calls):
        """``x * x`` overflows where ``pow`` saturates, so ints keep ``pow``."""
        x = torch.ones(4, 8, dtype=dtype)
        wrapped = flagos._flaggems_pow_scalar_square_wrapper(_original_pow(), None)

        assert wrapped(x, 2) == "pow"
        assert square_calls == []

    def test_the_captured_mul_stands_in_when_the_stub_has_none(
        self, monkeypatch, square_calls
    ):
        """The patch can install itself mid-import, before ``flag_gems.mul`` exists.

        ``triton`` asks for the active device while ``flag_gems/__init__.py`` is
        still importing, that query comes back through device init into the
        patch, and the top-level names are not bound yet. So the fallback is
        whatever ``mul`` was at patch time, and the call-time lookup is the
        preference.
        """
        del sys.modules["flag_gems"].mul

        def square(a, b):
            square_calls.append((a, b))
            return "square"

        wrapped = flagos._flaggems_pow_scalar_square_wrapper(_original_pow(), square)
        x = torch.randn(4, 8)

        assert wrapped(x, 2) == "square"
        assert square_calls == [(x, x)]

    def test_with_neither_a_stub_nor_a_captured_mul_the_original_is_reached(
        self, monkeypatch, square_calls
    ):
        del sys.modules["flag_gems"].mul
        wrapped = flagos._flaggems_pow_scalar_square_wrapper(_original_pow(), None)

        assert wrapped(torch.randn(4, 8), 2) == "pow"
        assert square_calls == []

    def test_the_wrapper_keeps_a_handle_on_what_it_replaced(self, square_calls):
        original = _original_pow()
        wrapped = flagos._flaggems_pow_scalar_square_wrapper(original, None)

        assert wrapped.__wrapped__ is original
        assert wrapped._torch_fl_square_via_mul is True


# ---------------------------------------------------------------------------
# mean.dim
# ---------------------------------------------------------------------------


def _cdiv(value, divisor):
    return -(-value // divisor)


class _TiledKernel:
    """Stands in for ``flag_gems.ops.mean.mean_dim_kernel``.

    Records the launch like the real thing is called -- ``kernel[grid](args)``
    with a grid that is a function of the meta-parameters -- and fills the
    output with a sentinel, so a test reads the *decision* rather than
    re-deriving the arithmetic in a second implementation.
    """

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(inp, out, m, n):
            self.launches.append(
                (grid({"BLOCK_M": 64}), tuple(inp.shape), tuple(out.shape), m, n)
            )
            out.fill_(SENTINEL)

        return launch


@pytest.fixture
def tiled():
    return _TiledKernel()


@pytest.fixture
def wrapped_mean(tiled):
    def mean_dim(inp, dim=None, keepdim=False, *, dtype=None):
        return "original"

    return (
        flagos._flaggems_mean_last_dim_wrapper(
            mean_dim, tiled, _cdiv, contextlib.nullcontext
        ),
        tiled,
    )


class TestMeanIsLastDim:
    """The tiled kernel is only right for a contiguous last-axis row mean."""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((4, 8), -1),
            ((4, 8), 1),
            ((4, 8), [1]),
            ((1, 4122, 32, 128), -1),
            ((1, 4122, 32, 128), 3),
            ((4, 1024), -1),
        ],
        ids=[
            "2d-neg",
            "2d-pos",
            "2d-list",
            "model-shape",
            "model-shape-pos",
            "wide-1k",
        ],
    )
    def test_a_last_axis_row_mean_is_taken(self, shape, dim):
        assert flagos._mean_is_last_dim(torch.randn(*shape), dim) is True

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((4, 8), 0),
            ((4, 8), -2),
            ((4, 8), [0]),
            ((1, 4, 8), 1),
            ((4, 8), [0, 1]),
            ((1, 144, 2048, 2048), 1),
            ((4, 2048), -1),
        ],
        ids=[
            "leading",
            "leading-neg",
            "leading-list",
            "middle",
            "two-dims",
            "vae-shape",
            "wide-2k",
        ],
    )
    def test_any_other_reduction_is_refused(self, shape, dim):
        assert flagos._mean_is_last_dim(torch.randn(*shape), dim) is False

    @pytest.mark.parametrize("dim", [2, -3, 7])
    def test_an_out_of_range_dim_is_refused(self, dim):
        """``%`` normalises a negative dim; it does not reject a large one.

        Without the range test, ``dim=2`` on a 2-D input normalises to the last
        axis and the wrapper returns a plausible mean over the wrong axis where
        the op it replaces raises.
        """
        assert flagos._mean_is_last_dim(torch.randn(4, 8), dim) is False

    def test_a_strided_input_is_refused(self):
        """A transposed last axis is not contiguous, so the row is not a row."""
        assert flagos._mean_is_last_dim(torch.randn(8, 4).t(), -1) is False

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
    )
    def test_a_non_floating_input_is_refused(self, dtype):
        assert flagos._mean_is_last_dim(torch.ones(4, 8, dtype=dtype), -1) is False

    @pytest.mark.parametrize(
        "inp",
        [torch.randn(0, 8), torch.randn(3, 0, 8)],
        ids=["empty", "empty-row"],
    )
    def test_a_reduction_with_no_row_to_speak_of_is_refused(self, inp):
        assert flagos._mean_is_last_dim(inp, -1) is False

    def test_a_non_tensor_is_refused(self):
        """The wrapper is reached from dispatch, but not only with tensors."""
        assert flagos._mean_is_last_dim([[1.0, 2.0]], -1) is False

    def test_no_dim_is_refused(self):
        assert flagos._mean_is_last_dim(torch.randn(4, 8), None) is False


class TestMeanLastDimWrapper:
    """The wrapper's shape handling, not the kernel's arithmetic."""

    def test_a_row_mean_reaches_the_tiled_kernel(self, wrapped_mean):
        wrapped, kernel = wrapped_mean
        x = torch.randn(1, 4122, 32, 128)

        out = wrapped(x, -1, True)

        assert len(kernel.launches) == 1, "the tiled kernel was not reached"
        grid, in_shape, out_shape, m, n = kernel.launches[0]
        assert in_shape == (1, 4122, 32, 128)
        # mean_dim_kernel writes one value per row whatever the input's rank.
        assert out_shape == (131904, 1)
        assert (m, n) == (131904, 128)
        assert grid == (_cdiv(131904, 64),)
        assert out.shape == (1, 4122, 32, 1)
        assert bool((out == SENTINEL).all())

    def test_keepdim_false_squeezes_the_axis_the_kernel_wrote(self, wrapped_mean):
        wrapped, kernel = wrapped_mean
        x = torch.randn(4, 8)

        out = wrapped(x, -1, False)

        assert len(kernel.launches) == 1
        assert out.shape == (4,)

    def test_a_refused_call_goes_to_the_original(self, wrapped_mean):
        wrapped, kernel = wrapped_mean
        x = torch.randn(4, 8)

        assert wrapped(x, 0, True) == "original"
        assert kernel.launches == []

    def test_an_explicit_dtype_keeps_the_original(self, wrapped_mean):
        """``dtype`` is what the kernel's accumulators would have to follow."""
        wrapped, kernel = wrapped_mean

        assert wrapped(torch.randn(4, 8), -1, True, dtype=torch.float64) == "original"
        assert kernel.launches == []

    def test_a_non_contiguous_call_keeps_the_original(self, wrapped_mean):
        wrapped, kernel = wrapped_mean

        assert wrapped(torch.randn(8, 4).t(), -1, True) == "original"
        assert kernel.launches == []

    def test_a_wide_row_keeps_the_original(self, wrapped_mean):
        wrapped, kernel = wrapped_mean

        assert wrapped(torch.randn(4, 2048), -1, True) == "original"
        assert kernel.launches == []

    def test_the_wrapper_keeps_a_handle_on_what_it_replaced(self, wrapped_mean):
        wrapped, _ = wrapped_mean

        assert wrapped._torch_fl_tiled_row_mean is True
        assert wrapped.__wrapped__(torch.randn(4, 8), -1, True) == "original"


# ---------------------------------------------------------------------------
# the triton key prewarm
# ---------------------------------------------------------------------------


class _RecordingThread:
    """A ``threading.Thread`` that runs its target in ``start()``."""

    instances = []

    def __init__(self, target=None, name=None, daemon=None):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        _RecordingThread.instances.append(self)

    def start(self):
        self.started = True
        self.target()


@pytest.fixture
def tkey(monkeypatch):
    """A stub ``triton.runtime.cache`` whose ``triton_key`` records its calls."""
    calls = []

    def triton_key():
        calls.append(1)
        return "key"

    triton = types.ModuleType("triton")
    runtime = types.ModuleType("triton.runtime")
    cache = types.ModuleType("triton.runtime.cache")
    cache.triton_key = triton_key
    triton.runtime = runtime
    runtime.cache = cache
    for mod in (triton, runtime, cache):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr("threading.Thread", _RecordingThread)
    _RecordingThread.instances = []
    return calls


def _gate(monkeypatch, accelerator, routed):
    monkeypatch.setattr("torch_fl._build_accelerator", lambda: accelerator)
    monkeypatch.setattr("torch_fl._conf_routes_to_flaggems", lambda: routed)


class TestPrewarmTritonKey:
    """The key is hashed on a side thread, on the builds it was measured on."""

    def test_the_key_is_computed_on_a_daemon_thread(self, monkeypatch, tkey):
        _gate(monkeypatch, "dcu", True)

        flagos._prewarm_triton_key()

        assert len(_RecordingThread.instances) == 1
        thread = _RecordingThread.instances[0]
        assert thread.started, "the thread was created but never started"
        assert thread.daemon, "a non-daemon thread would hold the process open"
        assert thread.name == "torch_fl-triton-key"
        # ``start()`` runs the target, so by here the lru_cache is populated.
        assert tkey == [1]

    @pytest.mark.parametrize(
        "accelerator,routed",
        [("cuda", True), ("dcu", False), ("ascend", False), ("cuda", False)],
        ids=["not-dcu", "not-routed", "ascend", "neither"],
    )
    def test_nothing_starts_off_the_measured_configuration(
        self, monkeypatch, tkey, accelerator, routed
    ):
        _gate(monkeypatch, accelerator, routed)

        flagos._prewarm_triton_key()

        assert _RecordingThread.instances == []
        assert tkey == []

    def test_a_key_that_cannot_be_computed_is_swallowed(self, monkeypatch, tkey):
        """Best-effort: an unwarmed cache is slow, not broken."""
        _gate(monkeypatch, "dcu", True)

        def boom():
            raise RuntimeError("no key for you")

        sys.modules["triton.runtime.cache"].triton_key = boom

        flagos._prewarm_triton_key()

        assert len(_RecordingThread.instances) == 1
        assert _RecordingThread.instances[0].started

    def test_a_triton_without_the_function_is_swallowed(self, monkeypatch, tkey):
        """The module can be importable and still not have the name."""
        _gate(monkeypatch, "dcu", True)
        del sys.modules["triton.runtime.cache"].triton_key

        flagos._prewarm_triton_key()

        assert _RecordingThread.instances == []
