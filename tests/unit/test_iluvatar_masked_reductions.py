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

"""Unit coverage for the Iluvatar non-inner reduction workaround.

The defect the workaround avoids needs the Iluvatar backend to observe, so it
cannot be reproduced here. What can be checked is the part that would silently
be wrong if it drifted: which calls take the workaround at all, which axis is
moved, and that the result comes back in the shape the caller asked for. A
wrapper that redirected an inner-dim reduction, or that dropped ``keepdim``,
would be wrong on every platform while looking like a correct fix.

The module is loaded by path rather than imported: importing ``torch_fl``
executes the package ``__init__`` and needs a built accelerator, and the module
under test itself only needs ``torch``. The path resolves through the installed
``torch_fl`` first so the same file runs in a wheel-only workspace and in a
source checkout.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _module_path() -> Path:
    """Locate torch_fl/compile/iluvatar_masked_reductions.py without importing the package.

    From wherever torch_fl resolves: the installed wheel in CI's wheel-only test
    workspace (which has no source tree), or the repo copy in a source checkout.
    Same arrangement as tests/integration/ops/conftest.py's `_platform_path`,
    because this file runs in both.
    """
    spec = importlib.util.find_spec("torch_fl")
    if spec is not None and spec.origin:
        candidate = (
            Path(spec.origin).resolve().parent
            / "compile"
            / "iluvatar_masked_reductions.py"
        )
        if candidate.is_file():
            return candidate
    return (
        Path(__file__).resolve().parents[2]
        / "torch_fl"
        / "compile"
        / "iluvatar_masked_reductions.py"
    )


def _load():
    spec = importlib.util.spec_from_file_location(
        "iluvatar_masked_reductions", _module_path()
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mr = _load()


@pytest.mark.parametrize(
    "shape, dim, expected",
    [
        ((2, 3, 4), 1, (1, False)),
        ((2, 3, 4), -2, (1, False)),
        ((2, 3, 4), [1], (1, True)),
        ((2, 3, 4), (1,), (1, True)),
        ((2, 3, 4), 2, (None, False)),  # trailing axis: inner kernel
        ((2, 3, 4), -1, (None, False)),
        ((2, 3, 4), 0, (0, False)),
        ((2, 3, 4), None, (None, False)),  # full reduction
        ((2, 3, 4), [1, 2], (None, False)),  # multi-dim: left alone
        ((2, 3, 4), [], (None, False)),
        ((), 0, (None, False)),  # 0-d
    ],
)
def test_single_non_inner_dim(shape, dim, expected):
    x = torch.empty(shape)
    assert mr._single_non_inner_dim(x, dim) == expected


def _reference_sum(x, dim, keepdim, *, dtype=None):
    """Stand-in for flag_gems.sum_dim: records what it was asked for."""
    _reference_sum.calls.append((tuple(x.shape), dim, keepdim))
    return x.sum(dim=dim if isinstance(dim, int) else tuple(dim), keepdim=keepdim)


_reference_sum.calls = []


def test_reduce_moves_axis_and_restores_keepdim():
    _reference_sum.calls.clear()
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    patched = mr.wrap_reduction(_reference_sum, "reduce")

    out = patched(x, 1, True)
    assert out.shape == (2, 1, 4, 5)
    assert torch.equal(out, x.sum(dim=1, keepdim=True))
    # The inner call got a permuted, contiguous view whose last axis is the
    # reduced one, and keepdim off (it is re-applied by the reshape).
    shape, dim, keepdim = _reference_sum.calls[-1]
    assert shape == (2, 4, 5, 3) and dim == -1 and keepdim is False

    out = patched(x, 1, False)
    assert out.shape == (2, 4, 5)
    assert torch.equal(out, x.sum(dim=1))


def test_reduce_preserves_sequence_dim_form():
    """sum.dim_IntList passes a list; flag_gems rejects an int there."""
    _reference_sum.calls.clear()
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    patched = mr.wrap_reduction(_reference_sum, "reduce")
    out = patched(x, [1], False)
    shape, dim, keepdim = _reference_sum.calls[-1]
    assert isinstance(dim, list) and dim == [-1]
    assert torch.equal(out, x.sum(dim=1))


def test_inner_dim_calls_the_original_unchanged():
    _reference_sum.calls.clear()
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    patched = mr.wrap_reduction(_reference_sum, "reduce")
    out = patched(x, -1, False)
    assert _reference_sum.calls[-1] == ((2, 3, 4), -1, False)
    assert out.shape == (2, 3)
    # ... and an unknown dim form is forwarded, not guessed at.
    _reference_sum.calls.clear()
    patched(x, [0, 1], False)
    assert _reference_sum.calls[-1] == ((2, 3, 4), [0, 1], False)


def test_softmax_permutes_back():
    def fake_softmax(x, dim, *args, **kwargs):
        fake_softmax.calls.append((tuple(x.shape), dim))
        return torch.softmax(x, dim)

    fake_softmax.calls = []
    x = torch.randn(2, 3, 4, 5)
    patched = mr.wrap_reduction(fake_softmax, "softmax")

    out = patched(x, 1)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.softmax(x, dim=1), atol=1e-6)
    shape, dim = fake_softmax.calls[-1]
    assert shape == (2, 4, 5, 3) and dim == -1

    # trailing axis goes straight through
    fake_softmax.calls.clear()
    patched(x, -1)
    assert fake_softmax.calls[-1] == ((2, 3, 4, 5), -1)


def test_std_kind_keeps_keepdim_keyword_only():
    """std's keepdim is keyword-only, so it cannot be passed positionally."""

    def fake_std(x, dim=None, *, correction=None, keepdim=False):
        fake_std.calls.append((tuple(x.shape), dim, correction, keepdim))
        return torch.std(x, dim=dim, correction=correction, keepdim=keepdim)

    fake_std.calls = []
    x = torch.randn(2, 3, 4)
    patched = mr.wrap_reduction(fake_std, "reduce_kwonly")
    out = patched(x, 1, correction=1, keepdim=True)
    assert out.shape == (2, 1, 4)
    shape, dim, correction, keepdim = fake_std.calls[-1]
    assert shape == (2, 4, 3) and dim == -1 and correction == 1 and keepdim is False


def test_out_entry_points_are_never_rewired(monkeypatch):
    """`out=` entry points keep their own argument behaviour.

    flag_gems' `sum_dim_out` is not `sum_dim`: with no `dtype` it passes the
    caller's `keepdim` through and reads the result dtype off the input, which
    is how `torch.sum(bool_input, dim=1, out=int64_out)` lands in an int64
    buffer. The wrapper forces `keepdim=False` and returns a fresh tensor, so
    wrapping that binding would silently change both. Alias sweeps keyed on
    object identity are exactly where that mistake hides: a build that binds
    `sum_dim_out` to the same function as `sum_dim` would be rewired with it.
    """

    class FakeGems:
        pass

    def sum_dim(inp, dim=None, keepdim=False, **kw):
        return inp

    def sum_dim_out(inp, dim=None, keepdim=False, **kw):
        return inp

    fake = FakeGems()
    fake.sum_dim = sum_dim
    fake.sum_dim_out = sum_dim_out
    monkeypatch.setitem(sys.modules, "flag_gems", fake)

    assert mr.patch_iluvatar_masked_reductions() is True
    assert fake.sum_dim.__code__.co_name == "patched"
    assert fake.sum_dim_out is sum_dim_out

    # The identity sweep is what makes this worth pinning: a build that aliases
    # the two entry points must still leave the `out=` name alone.
    del sys.modules["flag_gems"]
    aliased = FakeGems()
    aliased.sum_dim = aliased.sum_dim_out = sum_dim
    monkeypatch.setitem(sys.modules, "flag_gems", aliased)
    assert mr.patch_iluvatar_masked_reductions() is True
    assert aliased.sum_dim.__code__.co_name == "patched"
    assert aliased.sum_dim_out is sum_dim


def test_bool_sum_out_dtype_survives_the_patch(monkeypatch, tmp_path):
    """`torch.sum(bool, dim=1, out=...)` still writes the int64 the op promises.

    Mirrors flag_gems' `sum_dim_comm` dtype rule: an unspecified dtype is read
    off the input, and a bool input becomes int64. The patched binding is the
    one `sum.IntList_out` resolves, so if the sweep had wrapped it the caller
    would get a bool result back and the `out=` buffer would stay untouched.
    """

    class FakeGems:
        pass

    def sum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
        if dtype is None:
            dtype = torch.int64 if inp.dtype is torch.bool else inp.dtype
        result = torch.sum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    fake = FakeGems()
    fake.sum_dim = lambda inp, dim=None, keepdim=False, **kw: inp
    fake.sum_dim_out = sum_dim_comm
    monkeypatch.setitem(sys.modules, "flag_gems", fake)
    mr.patch_iluvatar_masked_reductions()

    mask = torch.tensor([[True, False, True], [False, True, True]])
    out = torch.empty(2, dtype=torch.int64)
    got = fake.sum_dim_out(mask, 1, out=out)
    assert got is out
    assert out.dtype == torch.int64
    assert out.tolist() == [2, 2]


def test_patch_is_idempotent_and_tolerates_missing_flag_gems(monkeypatch):
    class FakeGems:
        pass

    fake = FakeGems()
    fake.sum_dim = lambda inp, dim=None, keepdim=False, **kw: inp
    fake.softmax = lambda inp, dim, *a, **kw: inp
    monkeypatch.setitem(sys.modules, "flag_gems", fake)
    before = fake.sum_dim

    assert mr.patch_iluvatar_masked_reductions() is True
    assert fake.sum_dim is not before
    assert fake.sum_dim.__code__.co_name == "patched"
    # A second install must not wrap the wrapper again.
    assert mr.patch_iluvatar_masked_reductions() is False
    assert fake.sum_dim.__code__.co_name == "patched"

    monkeypatch.setitem(sys.modules, "flag_gems", None)
    assert mr.patch_iluvatar_masked_reductions() is False
