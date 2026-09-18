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

"""Unit coverage for the FlagGems pointwise dispatch patch installed on DCU.

``_patch_flaggems_pointwise_dispatch`` replaces the launch path every FlagGems
elementwise op goes through, so the risk it carries is not "is it faster" --
that is measured on hardware and recorded in ``docs/reference/operator-support.md``
-- but "does it still launch the same kernel with the same arguments". The tests
here pin the three pieces that decide that:

* the per-entry classification (``_libentry_run_plan``), which decides which
  argument list each positional argument belongs to,
* the fast key (``_fast_libentry_key``), which has to produce the stock
  ``LibEntry.key`` tuple exactly, because that tuple selects the compiled kernel,
* the two pure wrappers (``_hoisted_descriptor_cache_key``,
  ``_memoized_broadcast_shapes``), which have to be indistinguishable from the
  functions they replace, cache hit or miss.

The fast ``run`` itself cannot be exercised off a device -- it launches a
kernel -- so its own agreement with the stock path is asserted on hardware, and
what is checked here is that every input it cannot express is handed back to the
stock ``run`` rather than guessed at.
"""

import importlib

import pytest
import torch

from torch_fl import _build_accelerator, flagos

libentry = None
try:
    # ``import flag_gems.utils.libentry as le`` would bind the *function* that
    # flag_gems' package __init__ re-exports under that name; import_module
    # hands back the module itself.
    libentry = importlib.import_module("flag_gems.utils.libentry")
except Exception:  # noqa: BLE001 - any import failure means "not installed here"
    libentry = None

pytestmark = pytest.mark.skipif(
    libentry is None, reason="flag_gems is not installed in this environment"
)


@pytest.fixture
def entry():
    """A real ``LibEntry``: the classification is what is under test, not a stub."""
    triton = pytest.importorskip("triton")
    import triton.language as tl

    @libentry.libentry()
    @triton.jit
    def _copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    return _copy_kernel


class TestRunPlan:
    """Which positional argument goes to which of the stock paths."""

    def test_partitions_every_position(self, entry):
        n = len(entry.jit_function.params)
        support, plain, constexpr = flagos._libentry_run_plan(entry, n)
        assert sorted(support + plain + constexpr) == list(range(n))

    def test_membership_follows_the_entry(self, entry):
        n = len(entry.jit_function.params)
        support, plain, constexpr = flagos._libentry_run_plan(entry, n)
        assert set(support) == {i for i in range(n) if i in entry.specialize_indices}
        assert set(plain) == {
            i for i in range(n) if i in entry.do_not_specialize_indices
        }
        assert set(constexpr) == set(range(n)) - set(support) - set(plain)

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_partial_launches_take_the_stock_path(self, entry, delta):
        # A launch that does not supply every parameter would need the stock
        # path's default handling, so no plan may be produced for it.
        n = len(entry.jit_function.params)
        assert flagos._libentry_run_plan(entry, n + delta) is None

    def test_a_tuned_entry_takes_the_stock_path(self, entry):
        # A tuner-produced constant is not in the argument list this path
        # forwards, so an entry that has one has to fall back.
        missing = object()
        original = entry.__dict__.get("_has_flagtune_tuner", missing)
        entry.__dict__["_has_flagtune_tuner"] = True
        try:
            n = len(entry.jit_function.params)
            assert flagos._libentry_run_plan(entry, n) is None
        finally:
            if original is missing:
                entry.__dict__.pop("_has_flagtune_tuner", None)
            else:
                entry.__dict__["_has_flagtune_tuner"] = original


class TestFastKeyMatchesStock:
    """The key selects the compiled kernel, so it has to be the stock key."""

    @staticmethod
    def _stock_key(entry, spec_args, dns_args, const_args):
        # Importing torch_fl installs the patch on DCU, so ``LibEntry.key`` is
        # already the fast one here; the stock function is the one it wrapped.
        stock = getattr(libentry.LibEntry.key, "__wrapped__", libentry.LibEntry.key)
        return stock(entry, spec_args, dns_args, const_args)

    @staticmethod
    def _specialization():
        gems_runtime = importlib.import_module("flag_gems.runtime")
        return flagos._hygon_tensor_specialization(gems_runtime.device)

    def test_same_tuple_for_a_typical_launch(self, entry):
        x = torch.randn(64)
        y = torch.randn(64)
        spec = [x, y] if 0 in entry.specialize_indices else []
        fast_key = flagos._fast_libentry_key(
            entry, libentry._descriptor_cache_key, self._specialization()
        )
        assert fast_key(spec, [64], [128]) == self._stock_key(entry, spec, [64], [128])

    def test_same_tuple_when_nothing_is_specialised(self, entry):
        fast_key = flagos._fast_libentry_key(
            entry, libentry._descriptor_cache_key, self._specialization()
        )
        assert fast_key([], [], []) == self._stock_key(entry, [], [], [])

    def test_specialization_hook_lands_in_the_key(self, entry):
        # The hygon branch appends the vendor's specialization to a tensor's
        # key; driven directly so it is covered on a host that takes the other
        # branch too.
        x = torch.randn(64)
        marker = object()
        key = flagos._fast_libentry_key(
            entry, libentry._descriptor_cache_key, lambda _t: marker
        )([x], [], [])
        aligned = x.data_ptr() % entry.divisibility == 0
        assert key == ((x.dtype, aligned, marker),)

    def test_hook_is_none_off_hygon(self):
        class _Device:
            vendor_name = "not-hygon"

        assert flagos._hygon_tensor_specialization(_Device()) is None

    def test_hook_resolves_to_a_callable_or_none(self):
        hook = self._specialization()
        assert hook is None or callable(hook)

    def test_cached_key_is_built_once_per_entry(self, entry):
        calls = []

        def key_for_entry(e):
            calls.append(e)
            return flagos._fast_libentry_key(e, libentry._descriptor_cache_key, None)

        first = flagos._cached_libentry_key(entry, key_for_entry)
        second = flagos._cached_libentry_key(entry, key_for_entry)
        try:
            assert first is second
            assert calls == [entry]
        finally:
            entry.__dict__.pop("_torch_fl_key_fn", None)


class TestHoistedDescriptorCacheKey:
    """A passthrough for everything that is not a tensor descriptor."""

    @staticmethod
    def _stock_descriptor_key():
        # Importing torch_fl installs the patch on DCU; the stock function is
        # whatever the installed one wrapped.
        return getattr(
            libentry._descriptor_cache_key,
            "__wrapped__",
            libentry._descriptor_cache_key,
        )

    @pytest.mark.parametrize("obj", [7, 2.5, "i32", None])
    def test_non_descriptors_pass_through(self, obj):
        dck = flagos._hoisted_descriptor_cache_key(None)
        assert dck(obj) is obj

    def test_tensors_pass_through_on_this_platform(self):
        dck = flagos._hoisted_descriptor_cache_key(None)
        x = torch.randn(4)
        assert dck(x) is x

    def test_stock_agrees_on_non_descriptors(self):
        dck = flagos._hoisted_descriptor_cache_key(None)
        stock = self._stock_descriptor_key()
        x = torch.randn(4)
        for obj in (x, 3, "i32"):
            assert repr(dck(obj)) == repr(stock(obj))


class TestBroadcastShapesMemo:
    """``torch.broadcast_shapes`` is pure, so the memo is invisible."""

    @staticmethod
    def _unpatched():
        return getattr(torch.broadcast_shapes, "__wrapped__", torch.broadcast_shapes)

    def test_matches_torch(self):
        original = self._unpatched()
        memo = flagos._memoized_broadcast_shapes(original)
        shapes = [(1, 4096, 1), (1, 1, 4096)]
        assert tuple(memo(*shapes)) == tuple(original(*shapes))
        assert isinstance(memo(*shapes), type(original(*shapes)))

    def test_repeat_returns_the_cached_object(self):
        memo = flagos._memoized_broadcast_shapes(self._unpatched())
        first = memo((2, 3), (3,))
        second = memo((2, 3), (3,))
        assert first is second

    def test_lists_fall_back_instead_of_raising(self):
        memo = flagos._memoized_broadcast_shapes(self._unpatched())
        assert tuple(memo([2, 3], [3])) == (2, 3)

    def test_cache_size_is_bounded(self):
        memo = flagos._memoized_broadcast_shapes(self._unpatched())
        limit = flagos._BROADCAST_SHAPES_LIMIT
        first = memo(
            (7, 11),
        )
        for i in range(limit + 8):
            memo(
                (i, 1),
            )
        # Past the limit the cache is dropped, and the answer is still right.
        assert tuple(
            memo(
                (7, 11),
            )
        ) == tuple(first)


class TestGate:
    """The patch is scoped to the build and conf it was measured on."""

    def _restore(self, run, key, descriptor_key, broadcast_shapes):
        libentry.LibEntry.run = run
        libentry.LibEntry.key = key
        libentry._descriptor_cache_key = descriptor_key
        torch.broadcast_shapes = broadcast_shapes

    def test_does_not_install_off_dcu(self):
        if _build_accelerator() == "dcu":
            pytest.skip("DCU build -- this is the configuration the gate opens on")
        before = (
            libentry.LibEntry.run,
            libentry.LibEntry.key,
            libentry._descriptor_cache_key,
            torch.broadcast_shapes,
        )
        flagos._patch_flaggems_pointwise_dispatch()
        assert (
            libentry.LibEntry.run,
            libentry.LibEntry.key,
            libentry._descriptor_cache_key,
            torch.broadcast_shapes,
        ) == before

    def test_installs_on_dcu_and_is_idempotent(self):
        if _build_accelerator() != "dcu":
            pytest.skip("not a DCU build")
        saved = (
            libentry.LibEntry.run,
            libentry.LibEntry.key,
            libentry._descriptor_cache_key,
            torch.broadcast_shapes,
        )
        already = getattr(libentry.LibEntry.run, "_torch_fl_fast", False)
        try:
            flagos._patch_flaggems_pointwise_dispatch()
            first = (
                libentry.LibEntry.run,
                libentry.LibEntry.key,
                libentry._descriptor_cache_key,
                torch.broadcast_shapes,
            )
            if not already:
                # A second install must not wrap the wrappers: the key and the
                # stock ``run`` a fallback returns to both have to stay the ones
                # the first install captured.
                flagos._patch_flaggems_pointwise_dispatch()
                assert libentry.LibEntry.run is first[0]
                assert libentry.LibEntry.key is first[1]
                assert libentry.LibEntry.run.__wrapped__ is saved[0]
                assert libentry.LibEntry.key.__wrapped__ is saved[1]
            assert getattr(libentry.LibEntry.run, "_torch_fl_fast", False)
            assert getattr(libentry.LibEntry.key, "_torch_fl_fast", False)
        finally:
            if not already:
                self._restore(*saved)
