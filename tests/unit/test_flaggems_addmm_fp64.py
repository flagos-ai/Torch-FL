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

"""Unit coverage for the MUSA float64 addmm repair installed at device init.

``_patch_flaggems_addmm_fp64`` replaces the float64 case of five FlagGems addmm
entrypoints -- ``addmm``, ``addmm.out``, ``addmm_``, ``addmm.dtype`` and
``addmm.dtype_out`` -- with a composition over ``mm``, because the mthreads
kernel rounds both dot operands to float32 inside its K loop while ``mm`` keeps
float64 -- except for a single-column product, where ``mm`` is float32 too. Five
separate things can be wrong here, and none of them needs a MUSA device to check:

* the gate, which has to accept exactly the call the composition can serve and
  decline everything else, including the mixed dtypes this route accepts
  silently;
* the composition's arithmetic, which is not a plain ``addmm`` in two places:
  ``alpha``/``beta`` scaling and the ``beta == 0`` case, where PyTorch ignores
  the bias rather than scaling it, NaN and Inf included;
* the product's shape: ``mm`` is float64 everywhere except a single-column
  result, where it is float32, so that one shape is not asked of it;
* the ordering that makes an in-place ``addmm_`` work, whose bias *is* its
  destination and so must be read before it is written;
* the installer's bookkeeping: which five names it wraps, that the defining
  module is written after the identity sweep, and the marker that makes a second
  call a no-op rather than a double wrap.

That the composition is *accurate on this hardware* -- every previously wrong
case at 1e-16 relative against a CPU float64 reference, where the kernel it
replaces is at 1e-7 -- is a measured claim, and it is asserted in
``tests/integration/ops/test_musa_flaggems.py`` instead.
"""

import sys
import types

import pytest
import torch

import torch_fl
from torch_fl import flagos


DT = torch.float64
SHAPE = (4, 6, 5)
# (alpha, beta) pairs the device harness measures: the identity, the one that
# drops the bias, and one that scales both terms and negates the second.
ALPHA_BETA = [(1, 1), (2.5, 0), (0.25, -1.75)]


def _operands(dtype=DT):
    m, k, n = SHAPE
    return (
        torch.randn(m, n, dtype=dtype),
        torch.randn(m, k, dtype=dtype),
        torch.randn(k, n, dtype=dtype),
    )


def _biases(dtype=DT):
    """Every bias shape an ``addmm`` accepts, which is what the composition sees.

    Nothing is tested of the bias's shape by the gate: 2-D of the output's
    shape, 1-D of length N, 0-D, and both broadcastable row/column forms are all
    one ``add`` for itself.
    """
    m, _, n = SHAPE
    return [
        torch.randn(m, n, dtype=dtype),
        torch.randn(n, dtype=dtype),
        torch.randn((), dtype=dtype),
        torch.randn(1, n, dtype=dtype),
        torch.randn(m, 1, dtype=dtype),
    ]


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_the_gate_accepts_three_strided_float64_operands():
    bias, mat1, mat2 = _operands()
    assert flagos._addmm_fp64_terms(bias, mat1, mat2, 2.5, -1.75) == (2.5, -1.75)
    # The ATen schema carries alpha and beta as Scalar, so a call arrives with
    # ints as often as floats and both have to leave as floats.
    assert flagos._addmm_fp64_terms(bias, mat1, mat2, 1, 1) == (1.0, 1.0)
    # A 0-d bias is still a strided float64 tensor.
    assert flagos._addmm_fp64_terms(torch.randn((), dtype=DT), mat1, mat2, 1, 1) == (
        1.0,
        1.0,
    )


def test_the_gate_declines_every_operand_it_cannot_serve():
    bias, mat1, mat2 = _operands()
    deviations = {
        "a float32 bias": _operands(dtype=torch.float32),
        "a float32 bias beside float64 matrices": (
            torch.randn(*SHAPE[:1], *SHAPE[2:], dtype=torch.float32),
            mat1,
            mat2,
        ),
        "a float32 mat1": (bias, torch.randn(*SHAPE[:2], dtype=torch.float32), mat2),
        "a float32 mat2": (bias, mat1, torch.randn(*SHAPE[1:], dtype=torch.float32)),
        # `layout`, not contiguity: a transposed tensor is still strided, and the
        # composition is built out of `mm`, which serves dense matrices.
        "a sparse mat1": (bias, torch.randn(*SHAPE[:2], dtype=DT).to_sparse(), mat2),
        "a non-tensor operand": (bias, mat1, [[1.0, 2.0]]),
    }
    for label, operands in deviations.items():
        assert flagos._addmm_fp64_terms(*operands, 1, 1) is None, label


# --------------------------------------------------------------------------
# The composition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,beta", ALPHA_BETA)
def test_the_composition_matches_addmm_for_every_bias_shape(alpha, beta):
    _, mat1, mat2 = _operands()
    for bias in _biases():
        got = flagos._fp64_addmm_composite(
            torch.mm, bias, mat1, mat2, alpha, beta, None
        )
        want = torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)
        assert got.dtype == DT
        torch.testing.assert_close(got, want, rtol=1e-13, atol=1e-13)


def test_a_bias_that_cannot_broadcast_raises_the_broadcasts_error():
    """The one error-surface difference the gate's breadth buys.

    A bias the composition's ``add`` cannot broadcast fails with the broadcast's
    ``RuntimeError``, where the FlagGems kernel raised an ``AssertionError``.
    Neither returns a result, and neither is a call that has one; this is here so
    the difference is recorded rather than discovered.
    """
    _, mat1, mat2 = _operands()
    with pytest.raises(RuntimeError):
        flagos._fp64_addmm_composite(
            torch.mm, torch.randn(3, 3, dtype=DT), mat1, mat2, 1.0, 1.0, None
        )


def test_a_nan_bias_is_ignored_when_beta_is_zero():
    """``beta == 0`` drops the bias instead of scaling it, NaN included.

    PyTorch's rule, and the FlagGems kernel's: with a zero beta the bias does not
    reach the result at all. The composition has to skip the term rather than
    multiply it, because ``NaN * 0.0`` is NaN and would put a NaN into a result
    the reference leaves finite.
    """
    bias = torch.full((*SHAPE[:1], *SHAPE[2:]), float("nan"), dtype=DT)
    _, mat1, mat2 = _operands()
    got = flagos._fp64_addmm_composite(torch.mm, bias, mat1, mat2, 1.0, 0.0, None)
    want = torch.addmm(bias, mat1, mat2, beta=0.0, alpha=1.0)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-13, atol=1e-13)
    # The spelling the skip exists to avoid.
    assert not torch.isfinite(bias * 0.0).any()


def test_out_is_written_after_the_bias_is_read():
    """``addmm_`` arrives with its bias *as* its destination.

    The composition writes ``out`` last, so a bias that is also the output buffer
    contributes its pre-call value. Writing the product into ``out`` first would
    lose it, which is the whole reason the in-place entrypoint needs its own
    ``in_place`` flag.
    """
    bias, mat1, mat2 = _operands()
    want = torch.addmm(bias.clone(), mat1, mat2)
    got = flagos._fp64_addmm_composite(torch.mm, bias, mat1, mat2, 1.0, 1.0, out=bias)
    assert got is bias
    torch.testing.assert_close(bias, want, rtol=1e-13, atol=1e-13)


def test_an_out_buffer_is_written_and_returned():
    bias, mat1, mat2 = _operands()
    out = torch.empty(*SHAPE[:1], *SHAPE[2:], dtype=DT)
    got = flagos._fp64_addmm_composite(torch.mm, bias, mat1, mat2, 1.5, 0.5, out=out)
    assert got is out
    torch.testing.assert_close(
        out,
        torch.addmm(bias, mat1, mat2, beta=0.5, alpha=1.5),
        rtol=1e-13,
        atol=1e-13,
    )


# --------------------------------------------------------------------------
# The product's shape
# --------------------------------------------------------------------------


class _MMRecorder:
    """A stand-in for ``mm``: exact, and recording the shapes it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, mat1, mat2):
        self.calls.append((tuple(mat1.shape), tuple(mat2.shape)))
        return torch.mm(mat1, mat2)


def test_a_single_column_product_is_never_asked_of_mm_that_way():
    """``mm`` is float32 on this device exactly when its product is one column
    wide -- it dispatches to a GEMV kernel that rounds both operands -- so the
    product is widened to two columns and sliced, and the sliced column is the
    exact one.
    """
    recorder = _MMRecorder()
    mat1 = torch.randn(*SHAPE[:2], dtype=DT)
    mat2 = torch.randn(SHAPE[1], 1, dtype=DT)

    got = flagos._fp64_addmm_product(recorder, mat1, mat2)

    assert recorder.calls == [(tuple(mat1.shape), (SHAPE[1], 2))]
    assert got.shape == (SHAPE[0], 1)
    torch.testing.assert_close(got, torch.mm(mat1, mat2), rtol=1e-13, atol=1e-13)


def test_every_wider_product_goes_to_mm_unchanged():
    recorder = _MMRecorder()
    _, mat1, mat2 = _operands()

    got = flagos._fp64_addmm_product(recorder, mat1, mat2)

    assert recorder.calls == [(tuple(mat1.shape), tuple(mat2.shape))]
    torch.testing.assert_close(got, torch.mm(mat1, mat2), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("alpha,beta", ALPHA_BETA)
def test_the_composition_is_exact_for_a_single_column_result(alpha, beta):
    """The whole path at N == 1, bias shapes included.

    A single-column ``addmm`` is where the composition's product and its element
    terms meet the broadcast rules at their narrowest, and it is the one shape
    the device harness measures that the wider cases cannot stand in for.
    """
    mat1 = torch.randn(*SHAPE[:2], dtype=DT)
    mat2 = torch.randn(SHAPE[1], 1, dtype=DT)
    for bias in (torch.randn(SHAPE[0], 1, dtype=DT), torch.randn(1, dtype=DT)):
        got = flagos._fp64_addmm_composite(
            _MMRecorder(), bias, mat1, mat2, alpha, beta, None
        )
        want = torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=1e-13, atol=1e-13)


# --------------------------------------------------------------------------
# The wrappers
# --------------------------------------------------------------------------


class _Recorder:
    """Stands in for a FlagGems entrypoint and records how it was called."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else object()

    def __call__(self, bias, mat1, mat2, *, beta=1, alpha=1, out=None):
        self.calls.append((bias, mat1, mat2, beta, alpha, out))
        return self.result


def test_every_call_the_gate_declines_reaches_the_original_untouched():
    original = _Recorder()
    wrapper = flagos._flaggems_addmm_fp64_wrapper(original, torch.mm)
    bias32, mat132, mat232 = _operands(dtype=torch.float32)

    assert wrapper(bias32, mat132, mat232) is original.result
    assert original.calls[-1] == (bias32, mat132, mat232, 1, 1, None)

    # An fp64 call with an out buffer of a dtype the composition cannot fill is
    # the kernel's to serve, or to refuse.
    bias, mat1, mat2 = _operands()
    short_out = torch.empty(*SHAPE[:1], *SHAPE[2:], dtype=torch.float32)
    assert wrapper(bias, mat1, mat2, out=short_out) is original.result
    assert original.calls[-1] == (bias, mat1, mat2, 1, 1, short_out)

    assert wrapper(bias32, mat132, mat232, beta=2.5, alpha=-1.75) is original.result
    assert original.calls[-1] == (bias32, mat132, mat232, 2.5, -1.75, None)


def test_the_float64_call_does_not_reach_the_original():
    original = _Recorder()
    wrapper = flagos._flaggems_addmm_fp64_wrapper(original, torch.mm)
    bias, mat1, mat2 = _operands()

    got = wrapper(bias, mat1, mat2, beta=0.25, alpha=-1.75)
    assert original.calls == []
    torch.testing.assert_close(
        got,
        torch.addmm(bias, mat1, mat2, beta=0.25, alpha=-1.75),
        rtol=1e-13,
        atol=1e-13,
    )


def test_the_in_place_wrapper_writes_its_own_bias():
    original = _Recorder()
    wrapper = flagos._flaggems_addmm_fp64_wrapper(original, torch.mm, in_place=True)
    bias, mat1, mat2 = _operands()

    want = torch.addmm(bias.clone(), mat1, mat2)
    assert wrapper(bias, mat1, mat2) is bias
    assert original.calls == []
    torch.testing.assert_close(bias, want, rtol=1e-13, atol=1e-13)


class _DtypeRecorder(_Recorder):
    """The ``addmm.dtype`` signature, which carries ``out_dtype`` positionally."""

    def __call__(self, bias, mat1, mat2, out_dtype=None, *, beta=1, alpha=1, out=None):
        self.calls.append((bias, mat1, mat2, out_dtype, beta, alpha, out))
        return self.result


def test_the_dtype_wrapper_gates_on_out_dtype():
    original = _DtypeRecorder()
    wrapper = flagos._flaggems_addmm_dtype_fp64_wrapper(original, torch.mm)
    bias, mat1, mat2 = _operands()

    # Every mixed-precision pair a compiler emits stays with FlagGems' kernel.
    for out_dtype in (torch.float32, torch.float16):
        assert wrapper(bias, mat1, mat2, out_dtype) is original.result
        assert original.calls[-1] == (bias, mat1, mat2, out_dtype, 1, 1, None)

    got = wrapper(bias, mat1, mat2, torch.float64)
    assert len(original.calls) == 2
    assert got.dtype == DT
    torch.testing.assert_close(
        got, torch.addmm(bias, mat1, mat2), rtol=1e-13, atol=1e-13
    )


def test_the_marker_is_what_makes_the_wrapper_recognisable():
    """The installer skips an entrypoint already carrying the marker."""
    original = _Recorder()
    for factory in (
        flagos._flaggems_addmm_fp64_wrapper,
        flagos._flaggems_addmm_dtype_fp64_wrapper,
    ):
        wrapper = factory(original, torch.mm)
        assert wrapper.__wrapped__ is original
        assert getattr(wrapper, flagos._ADDMM_FP64_PATCHED) is True
        assert not getattr(original, flagos._ADDMM_FP64_PATCHED, False)


# --------------------------------------------------------------------------
# The installer
# --------------------------------------------------------------------------


def _stub_flag_gems(monkeypatch):
    """Stand up the FlagGems tree the installer walks, without FlagGems.

    The installer asks for the vendor directory by name
    (``_{vendor_name}.ops``) and resolves each entrypoint by importing the
    defining submodule, so a handful of module objects and the two package-level
    predicates are the whole of what it reads.
    """

    def make(label):
        def entrypoint(*args, **kwargs):
            return (label, args, kwargs)

        return entrypoint

    functions = {
        name: make(name)
        for name in (
            "mm",
            "addmm",
            "addmm_out",
            "addmm_",
            "addmm_dtype",
            "addmm_dtype_out",
        )
    }
    modules = {}

    def module(name, **attributes):
        mod = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(mod, key, value)
        modules[name] = mod
        return mod

    module("_mthreads.ops.mm", mm=functions["mm"])
    module(
        "_mthreads.ops.addmm",
        addmm=functions["addmm"],
        addmm_out=functions["addmm_out"],
        addmm_dtype=functions["addmm_dtype"],
        addmm_dtype_out=functions["addmm_dtype_out"],
    )
    module("_mthreads.ops.addmm_", addmm_=functions["addmm_"])

    flag_gems = types.ModuleType("flag_gems")
    flag_gems.runtime = types.SimpleNamespace(
        device=types.SimpleNamespace(vendor_name="mthreads")
    )
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setattr(torch_fl, "_build_accelerator", lambda: "musa")
    monkeypatch.setattr(torch_fl, "_conf_routes_to_flaggems", lambda: True)

    rebound = []

    def resolve(module_names, attribute):
        for module_name in module_names:
            mod = modules.get(module_name)
            if mod is not None and attribute in vars(mod):
                return mod, vars(mod)[attribute]
        return None, None

    monkeypatch.setattr(flagos, "_flag_gems_callable", resolve)
    monkeypatch.setattr(
        flagos,
        "_rebind_flag_gems_name",
        lambda name, original, wrapper: rebound.append((name, original, wrapper)),
    )
    return modules, functions, rebound


def test_the_installer_wraps_all_five_entrypoints_in_order(monkeypatch):
    modules, functions, rebound = _stub_flag_gems(monkeypatch)
    flagos._patch_flaggems_addmm_fp64()

    assert [name for name, _, _ in rebound] == [
        "addmm",
        "addmm_out",
        "addmm_",
        "addmm_dtype",
        "addmm_dtype_out",
    ]
    for name, original, wrapper in rebound:
        assert original is functions[name]
        assert wrapper.__wrapped__ is original, name
        assert getattr(wrapper, flagos._ADDMM_FP64_PATCHED) is True, name

    # Each name is wrapped by its own call of the factory, and the defining
    # module is written as well as the sweep -- the package namespace is still
    # importing when this runs, so the sweep is what reaches it, but a second
    # import of the defining module must not hand back the unwrapped callable.
    assert len({id(wrapper) for _, _, wrapper in rebound}) == 5
    assert vars(modules["_mthreads.ops.addmm"])["addmm"] is rebound[0][2]
    assert vars(modules["_mthreads.ops.addmm"])["addmm_dtype"] is rebound[3][2]
    assert vars(modules["_mthreads.ops.addmm_"])["addmm_"] is rebound[2][2]
    assert vars(modules["_mthreads.ops.mm"])["mm"] is functions["mm"]


def test_a_second_install_is_a_no_op(monkeypatch):
    _, _, rebound = _stub_flag_gems(monkeypatch)
    flagos._patch_flaggems_addmm_fp64()
    assert len(rebound) == 5
    flagos._patch_flaggems_addmm_fp64()
    assert len(rebound) == 5


def test_the_installer_installs_nothing_without_mm(monkeypatch):
    """No ``mm`` means no float64-exact product to compose over."""
    modules, _, rebound = _stub_flag_gems(monkeypatch)
    del modules["_mthreads.ops.mm"]
    flagos._patch_flaggems_addmm_fp64()
    assert rebound == []


def test_the_installer_declines_a_conf_that_does_not_route_to_flaggems(monkeypatch):
    _, _, rebound = _stub_flag_gems(monkeypatch)
    monkeypatch.setattr(torch_fl, "_conf_routes_to_flaggems", lambda: False)
    flagos._patch_flaggems_addmm_fp64()
    assert rebound == []
