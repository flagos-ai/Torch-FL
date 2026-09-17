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

"""
Real MetaX coverage for the MetaX FlagGems hybrid path.

``backends_metax.conf`` is FlagGems-first: 12 of the ops torch_fl can route go
to ``flaggems_cpp`` (the C++ path, ``Backend::kFlagGemsCpp``), 591 go to
``flaggems`` (the FlagGems Python/Triton path, ``Backend::kFlagGems``), and only
the ops measured not to work there fall back to the cuda boxing kernel (maca
``libtorch_cuda``). This file pins both halves of that decision on hardware, the
way ``test_musa_flaggems.py`` does for the MThreats hybrid path:

1. **Routing** -- an op the conf assigns to a FlagGems key really dispatches
   there. Read from the dispatch log in a subprocess rather than from the conf,
   so the assertion covers the table the runtime builds and not just the text on
   disk. The log distinguishes the two FlagGems paths: conf ``flaggems`` prints
   as ``flagos_python`` and conf ``flaggems_cpp`` prints as ``flagos``.
2. **Execution** -- the same ops return the CPU answer, on inputs built once on
   the host and moved, so the two arms are compared on identical values. A
   ``flagos_python`` route that cannot resolve its ``flag_gems.<name>`` entry
   point raises (``AttributeError``) or, worse, runs the wrong kernel; only
   running it catches that.
3. **Exclusions** -- the ops measured to fail on flagos tensors stay off the
   FlagGems path. Most are a literal ``device.type == "cuda"`` guard in the
   KernelBench code, so no PrivateUse1 tensor can satisfy it and the op fails
   identically on every flagos platform -- see the ``flaggems_runtime_broken``
   entries in ``scripts/codegen_ops.py`` for the per-op diagnosis. The rest fail
   for a reason specific to their gems kernel -- a wrapper that raises, or a
   result that differs from ATen's -- and are listed per op in the
   ``metax_triton_fallback`` literal.

The dispatch log is the oracle for (1) instead of monkeypatching a FlagGems
callable: ``PythonOpCache::GetFunc`` memoizes the resolved ``py::object`` per
qualname, so a patch installed after the first call in the same session is not
observed. A subprocess starts with an empty cache.

Measured on MetaX C550 with flagtree 0.6.1+metax3.6 and FlagGems
5.4.0rc2.post1+g5a58df410 (master @ 5a58df410, 2026-09-15).

Seven of the ops measured here were added to the conf by that re-measurement,
not by the shared generator: they sit in ``flaggems_runtime_broken`` group (5)
in ``scripts/codegen_ops.py`` -- a hold that was about the generation cohort not
defining their ``flag_gems.<name>`` entry point, which the pinned cohort does --
and ``METAX_FLAGGEMS_MEASURED`` in ``scripts/gen_vendor_confs.py`` lifts them for
this platform only. Two of the seven were failing outright on the cuda boxing
route before this (``special_bessel_j1`` raises ``cudaErrorMemoryValueTooLarge``
through maca, ``linalg_ldl_solve`` needs a ``cusolverDnXsytrs_bufferSize`` maca
does not provide), so their execution cases below are a fix rather than a
preference.

An eighth op, ``igammac_``, was promoted by the same re-measurement and then
withdrawn from it: that probe fed strictly positive inputs, and on the domain
where the arguments go negative the gems kernel returns finite values where ATen
returns NaN (``a=-1.1524, b=+0.9200`` -> ``0.0275``; ``a=+0.8487, b=-1.4782`` and
``a=+0.3223, b=-1.6293`` -> ``1.0``; 36 of 64 elements of a ``torch.randn(8, 8)``
pair, every disagreement one-sided). The boxing route reproduces the host NaN
mask exactly, so it keeps the op and the exclusion below guards the withdrawal.

Sixteen more ops were withdrawn the same day, by a differential survey of the 166
ops MetaX gained when the FlagGems cohort was widened. Each was run through the
same ``2d-f32`` profile on both routes -- one input pair built on the host and
moved with ``.to("flagos")``, so both arms see identical values, and the cuda arm
reached with ``FLAGOS_OP_<op>=cuda`` -- and each one **passes on cuda while
failing on flaggems**, which is what makes the withdrawal a correction rather
than a preference. The failures fall in three groups, spelled out per op in the
``metax_triton_fallback`` literal in ``scripts/codegen_ops.py``: four where the
gems kernel asserts its input is a real CUDA tensor and aborts, four where the
gems wrapper raises on its own argument handling (``nansum.out``,
``lu_unpack.out``, ``linalg_matrix_exp.out`` and ``_cdist_forward``), and eight
that run to completion and return a result ATen does not. Five of the twenty-one
ops the survey flagged are **not** withdrawn -- they fail on both routes
(``_native_batch_norm_legit.no_stats`` segfaults on each, ``linalg_lstsq`` and
``log_sigmoid_backward``/``.grad_input`` return wrong values on each, and
``linalg_eig`` is *better* on flaggems: the boxing route raises "MAGMA requires
compiling PyTorch" while the gems eigenvalues match the host), so holding them
would not have fixed anything.

A seventeenth withdrawal, ``slice.Tensor``, came from a model rather than from
the survey. Qwen-Image-2512's transformer slices complex rotary frequencies
(diffusers' ``_compute_video_freqs``, ``freqs_pos[0][idx : idx + frame]``), and
``flag_gems/ops/slice.py`` asserts against ``complex64`` and ``complex128``
although its body is a ``torch.as_strided`` view that never reads the dtype. It
is the fourth failure mode -- the gems op refuses a dtype its own implementation
does not need -- and, unlike the sixteen, it is not MetaX-specific: the
assertion rejects complex on every device, so it is held here only to keep the
change local. Reported upstream as FlagGems issue #6356.

Usage:
    pytest tests/integration/ops/test_metax_flaggems.py -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest
import torch
import torch_fl


DEVICE = "flagos:0"

pytestmark = pytest.mark.flaggems_python

# Inputs every call form below builds on. Kept in one block so a call's shape
# requirements stay visible next to it: `src` has as many rows as `idx` has
# entries because `index_add_` asserts on that, and `p`/`q` are in [0, 1]
# because `binary_cross_entropy`'s CUDA kernel asserts on its input's domain
# (the CPU kernel only warns, so a wrong-domain input passes on the host and
# aborts the device -- `torch.rand(...) + 0.5` reaches 1.5 and traps).
_PRELUDE = (
    "a = torch.randn(4, 4, device=DEVICE)\n"
    "b = torch.randn(4, 4, device=DEVICE)\n"
    "p = torch.rand(4, 4, device=DEVICE)\n"
    "q = torch.rand(4, 4, device=DEVICE)\n"
    "zero = torch.zeros((), device=DEVICE)\n"
    "w = torch.randn(3, 4, device=DEVICE)\n"
    "idx = torch.tensor([0, 2, 1], device=DEVICE)\n"
    "src = torch.randn(3, 4, device=DEVICE)\n"
)

# Conf key -> (the call this test makes, the backend its dispatch line must
# name). The log prints the conf key verbatim -- `sub.Tensor` stays
# `sub.Tensor` -- so the keys here are also the names asserted against.
#
# The expected backend is per entry because "on FlagGems" is two different log
# names: `flagos_python` is the Python/Triton path (``flag_gems.<fn>``, conf key
# `flaggems`) and `flagos` is the C++ path (conf key `flaggems_cpp`). Embedding
# is in `_ROUTED_TO_FLAGGEMS` on the C++ slot; the other 30 are on the Python
# one, which is the path whose 591 routes this file exists to guard. Seven of
# the 30 -- the ones below the tensor-factory group -- come from
# METAX_FLAGGEMS_MEASURED rather than the shared coverage set; see the module
# docstring.
#
# Each call is asserted on *its own* op rather than on "nothing here ran on
# cuda": every one of these calls also emits incidental routes -- `lift_fresh`,
# `empty_like`, `permute`, `squeeze.dims`, `mul.Tensor` -- that the conf sends
# to the boxing kernel and that have nothing to do with the op under test.
_ROUTED_TO_FLAGGEMS = {
    "neg": ("torch.neg(a)", "flagos_python"),
    "abs": ("torch.abs(a)", "flagos_python"),
    "exp": ("torch.exp(a)", "flagos_python"),
    "log": ("torch.log(a)", "flagos_python"),
    "sqrt": ("torch.sqrt(a)", "flagos_python"),
    "rsqrt": ("torch.rsqrt(a)", "flagos_python"),
    "tanh": ("torch.tanh(a)", "flagos_python"),
    "sigmoid": ("torch.sigmoid(a)", "flagos_python"),
    "gelu": ("torch.nn.functional.gelu(a)", "flagos_python"),
    "silu": ("torch.nn.functional.silu(a)", "flagos_python"),
    "all": ("torch.all(a)", "flagos_python"),
    "any": ("torch.any(a)", "flagos_python"),
    "amax": ("torch.amax(a, 1)", "flagos_python"),
    "amin": ("torch.amin(a, 1)", "flagos_python"),
    "min.dim": ("torch.min(a, 1)", "flagos_python"),
    "sub.Tensor": ("a - b", "flagos_python"),
    "pow.Tensor_Tensor": ("a ** b", "flagos_python"),
    "where.self": ("torch.where(a > 0, a, b)", "flagos_python"),
    # A 0-d value: the FlagGems kernel asserts on that
    # (`masked_fill_ only supports a 0-dimensional value tensor`), and passing a
    # Python float instead selects `masked_fill.Scalar`, a different conf key.
    "masked_fill.Tensor": ("a.masked_fill(a > 0, zero)", "flagos_python"),
    # `slice.Tensor` used to be here. It moved to `cuda` when the widened cohort
    # put it on the FlagGems path and Qwen-Image-2512's complex rotary-frequency
    # slice hit `flag_gems/ops/slice.py`'s complex64 assertion; see
    # `_FORCED_OFF_FLAGGEMS` group 4 and `_FORCED_OFF_DISPATCH`.
    "embedding": ("torch.nn.functional.embedding(idx, w)", "flagos"),
    "index_add_": ("x = a.clone()\nx.index_add_(0, idx, src)", "flagos_python"),
    "linspace": ("torch.linspace(0.0, 1.0, 8, device=DEVICE)", "flagos_python"),
    "full": ("torch.full((4, 4), 3.0, device=DEVICE)", "flagos_python"),
    "ones": ("torch.ones((4, 4), device=DEVICE)", "flagos_python"),
    # The seven METAX_FLAGGEMS_MEASURED ops. `a[None]` is deliberately the call
    # for `unsqueeze`: that is the form a model actually writes, and the one the
    # stale-cohort hold was recorded as breaking.
    "_embedding_bag_per_sample_weights_backward": (
        "g = torch.randn(2, 4, device=DEVICE)\n"
        "ci = torch.tensor([0, 1, 2, 0], device=DEVICE)\n"
        "co = torch.tensor([0, 2], device=DEVICE)\n"
        "o2b = torch.tensor([0, 0, 1, 1], device=DEVICE)\n"
        "torch.ops.aten._embedding_bag_per_sample_weights_backward(g, w, ci, co, o2b, 0)",
        "flagos_python",
    ),
    "_native_batch_norm_legit_functional": (
        "torch.ops.aten._native_batch_norm_legit_functional(\n"
        "    a.view(2, 4, 1, 2), None, None, torch.zeros(4, device=DEVICE),\n"
        "    torch.ones(4, device=DEVICE), True, 0.1, 1e-5)",
        "flagos_python",
    ),
    "binary_cross_entropy_with_logits": (
        "torch.ops.aten.binary_cross_entropy_with_logits(a, q)",
        "flagos_python",
    ),
    "linalg_ldl_solve": (
        "m = (a @ a.T + 4 * torch.eye(4, device=DEVICE)).cpu()\n"
        "LD, pivots, _ = torch.linalg.ldl_factor_ex(m, hermitian=True)\n"
        "torch.ops.aten.linalg_ldl_solve(LD.to(DEVICE), pivots.to(DEVICE), b,\n"
        "                                hermitian=True)",
        "flagos_python",
    ),
    "special_bessel_j1": ("torch.ops.aten.special_bessel_j1(a)", "flagos_python"),
    "unsqueeze": ("a[None]", "flagos_python"),
    "unsqueeze_": ("torch.ops.aten.unsqueeze_(a.clone(), 1)", "flagos_python"),
}

# Ops the conf must KEEP off the FlagGems path, as conf key -> the call whose
# dispatch must name the boxing kernel. Most raise on a flagos tensor inside
# KernelBench -- a literal `device.type == "cuda"` guard in the emitted code, so
# no PrivateUse1 tensor can satisfy it -- which makes routing them to flaggems a
# hard failure rather than a slowdown. See the `flaggems_runtime_broken` entries
# in scripts/codegen_ops.py for the per-op diagnosis. `hardtanh_backward` and
# `mul_.Tensor` are here because their generated call would be wrong, not
# because they trap: both were re-measured to dispatch and compute correctly on
# the boxing route (grad sum 6.0 for a 4x4 hardtanh, matching the host).
_FORCED_OFF_FLAGGEMS = (
    "binary_cross_entropy",
    "binary_cross_entropy.out",
    "hardtanh_backward",
    "igammac",
    "igammac.out",
    "special_modified_bessel_i0",
    "special_modified_bessel_i0.out",
    "_fake_quantize_learnable_per_tensor_affine_backward",
    "upsample_bilinear2d",
    "mul_.Tensor",
    # Withdrawn from METAX_FLAGGEMS_MEASURED after re-measuring on the domain its
    # first probe excluded. `igammac_` on the gems path returns finite values
    # (0.0275, 1.0) where ATen's CPU and the cuda boxing route both return NaN,
    # so it is pinned back to boxing; the host NaN mask is reproduced exactly
    # there (51/51 NaN over a `torch.randn(8, 8)` pair, 0 one-sided).
    "igammac_",
    # The two of the ten stale-cohort ops that the MetaX re-measurement did NOT
    # promote. `max_unpool3d`: the gems signature is
    # `(input, indices, kernel_size, stride, padding, output_size)` while ATen's
    # is `(self, indices, output_size, stride, padding)`, so the generated call
    # passes 4 as the kernel size and returns (1, 2, 6, 6, 6) where boxing
    # returns (1, 2, 4, 4, 4) -- an arity-compatible, meaning-incompatible
    # signature that discovery's static checks cannot see.
    # `cudnn_batch_norm_backward` was not re-measured (its ATen schema needs
    # save_mean/save_var and an opaque reserveSpace tensor, and it has no CPU
    # kernel to compare against), so it stays held.
    "max_unpool3d",
    "cudnn_batch_norm_backward",
    # The sixteen withdrawn by the differential survey of the 166 ops the widened
    # FlagGems cohort added (see the module docstring). Every one of them PASSES
    # on the cuda boxing route and fails on flaggems, so the hold is a
    # correction; the per-op diagnosis is grouped by cause in the
    # `metax_triton_fallback` literal in scripts/codegen_ops.py. The groups below
    # state the *measured* gems verdict, which is what the survey recorded.
    #
    # Group 1 -- the gems kernel asserts its input is a real CUDA tensor, so it
    # aborts before computing anything:
    "special_bessel_j0",
    "special_i1e",
    "special_i1e.out",
    "special_chebyshev_polynomial_w.out",
    # Group 2 -- the gems wrapper raises on its own argument handling, an `out=`
    # variant it cannot serve:
    "nansum.out",
    "lu_unpack.out",
    "linalg_matrix_exp.out",
    "_cdist_forward",
    # Group 3 -- the gems kernel runs to completion and returns the wrong result,
    # so routing is the only symptom a caller would see without comparing values:
    "sum.out",
    "_compute_linear_combination",
    "_compute_linear_combination.out",
    "_fused_rms_norm",
    "igamma",
    "igamma_",
    "logit_backward",
    "special_shifted_chebyshev_polynomial_t",
    # Group 4 -- the gems op refuses a dtype its own implementation does not
    # need. flag_gems' `slice` asserts against complex64/complex128, but it
    # builds the result with `torch.as_strided` from the input's shape, strides
    # and storage offset and never consults the dtype, so the assertion is
    # vestigial. Qwen-Image-2512's transformer slices complex rotary
    # frequencies (diffusers' `_compute_video_freqs`), which is what surfaced
    # it; the assertion rejects complex on every device, so this is not
    # MetaX-specific and is held in MetaX only to keep the change local. Filed
    # upstream as FlagGems issue #6356 (the remainder of #6049 / #6061, which
    # trimmed the same assertion for `bool`).
    "slice.Tensor",
)

# The five of those whose dispatch is checked on hardware, one per failure mode:
# forward with a device guard on the input domain, an in-place op that would
# otherwise mutate through a FlagGems kernel, and a backward op reached only
# through autograd. `max_unpool3d` is a fourth mode -- the kernel runs and
# returns the wrong shape -- so its call is checked to reach the boxing kernel
# rather than to compute anything. `igammac_` is a fifth: a wrong *value* rather
# than a wrong shape or an abort, checkable only by routing, since its in-place
# call is the one whose flagos result disagrees with the host. Three more, at the
# end, cover the sixteen-op survey withdrawal -- one per cause group -- and a
# fourth covers the seventeenth withdrawal, `slice.Tensor`, whose failure mode is
# a dtype refusal rather than a kernel fault.
_FORCED_OFF_DISPATCH = {
    "binary_cross_entropy": "torch.nn.functional.binary_cross_entropy(p, q)",
    "mul_.Tensor": "x = a.clone()\nx.mul_(b)",
    "hardtanh_backward": (
        "x = a.clone().requires_grad_(True)\n"
        "torch.nn.functional.hardtanh(x).sum().backward()"
    ),
    # `b` here spans both signs, which is the domain the flaggems kernel gets
    # wrong; routing is what is asserted, so the call only has to be the one
    # that reaches `aten::igammac_`.
    "igammac_": (
        "y = torch.randn(4, 4, device=DEVICE)\ntorch.ops.aten.igammac_(y.clone(), y)"
    ),
    # The pooling input is built here rather than reusing `_PRELUDE`'s 2-D `a`:
    # the boxing kernel rejects a pooled output with an empty non-batch
    # dimension, which a 4x4 spatial input pooled by 2 collapses to.
    "max_unpool3d": (
        "u = torch.randn(1, 2, 4, 4, 4, device=DEVICE)\n"
        "pooled, pool_idx = torch.nn.functional.max_pool3d(\n"
        "    u, 2, stride=2, return_indices=True)\n"
        "torch.ops.aten.max_unpool3d(pooled, pool_idx, [4, 4, 4], [2, 2, 2], [0, 0, 0])"
    ),
    # One representative per cause group of the sixteen withdrawn by the
    # differential survey, so each group's *reason* is re-checked on hardware and
    # not only its conf line. The half of the A/B that can be asserted here is the
    # cuda arm: `special_i1e` raises on the gems path on a device-type assert,
    # `_cdist_forward` raises inside the gems wrapper, and `sum.out` -- which is
    # the one worth running rather than merely routing -- does not raise at all on
    # gems; it returns the `out` buffer's `(32, 32)` where the reduction is `()`.
    # The gems arm needs a route override and is recorded in
    # `docs/reference/operator-support.md`. The remaining thirteen are pinned by
    # the conf-text guard above and by that same survey run.
    "special_i1e": "torch.ops.aten.special_i1e(a)",
    "sum.out": ("torch.ops.aten.sum.out(a, dtype=None, out=torch.empty_like(a))"),
    "_cdist_forward": "torch.ops.aten._cdist_forward(a, b, 2.0, None)",
    # Group 4's only member, and the call is the one that failed: a complex
    # tensor sliced along its last axis, which is how diffusers' Qwen-Image
    # rotary embedding reaches `aten::slice.Tensor`. The `[0]` select keeps the
    # slice on a 1-D input, matching `freqs_pos[0][idx:idx + frame]`; slicing a
    # 2-D complex tensor would take the same route.
    "slice.Tensor": (
        "z = torch.complex(torch.randn(8, 16, device=DEVICE),\n"
        "                   torch.randn(8, 16, device=DEVICE))\n"
        "z[0][2:6]"
    ),
}

# Measured number of `flaggems` routes in backends_metax.conf. A regression guard
# rather than an exact contract: the count only moves when an op is added,
# removed, or re-measured, and each of those is a deliberate change.
_MEASURED_FLAGGEMS_ROUTES = 591

_DISPATCH_LINE = re.compile(r"\[flagos dispatch\] (\S+) -> (\S+)")


def _active_conf() -> str:
    return torch_fl.backend_config_path()


def _require_metax_flaggems() -> None:
    """Skip unless this is a MetaX boxing build with a usable FlagGems runtime."""
    if not _active_conf().endswith("backends_metax.conf"):
        pytest.skip(
            "the MetaX FlagGems routes are stated in backends_metax.conf; this "
            f"run selected {_active_conf() or '<no conf>'}"
        )
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("MetaX device is unavailable")
    try:
        import triton

        import flag_gems
    except Exception as exc:  # noqa: BLE001 - any import failure means skip
        pytest.skip(f"FlagGems MetaX runtime is unavailable: {exc}")
    if "metax" not in triton.backends.backends:
        pytest.skip("the installed Triton does not provide the MetaX backend")
    print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")


def _run_logged(body: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter with the dispatch log on.

    A fresh process is not incidental: the routing assertions read the dispatch
    log, and the operator under test must not have been reached earlier in this
    session (PythonOpCache memoizes the resolved FlagGems callable per qualname,
    and the conf is parsed once per process).
    """
    code = (
        "import torch, torch_fl\n"
        "from torch import nn\n"
        f"DEVICE = {DEVICE!r}\n"
        f"{_PRELUDE}"
        f"{body}\n"
    )
    env = os.environ.copy()
    env["FLAGOS_LOG"] = "dispatch"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _dispatch_routes(stderr: str) -> dict[str, str]:
    return {op: backend for op, backend in _DISPATCH_LINE.findall(stderr)}


def _ldl_solve_inputs() -> tuple[torch.Tensor, ...]:
    """(LD, pivots, B) for ``linalg_ldl_solve``, factored on the host.

    The factorisation is a composite op, not the kernel under test, and it is
    the reason this case does not just call ``torch.linalg.solve``.
    """
    a = torch.randn(4, 4)
    ld, pivots, _ = torch.linalg.ldl_factor_ex(
        a @ a.T + 4 * torch.eye(4), hermitian=True
    )
    return ld, pivots, torch.randn(4, 3)


# Cases compared at 1e-3 rather than 1e-4: the special-function and batch-norm
# kernels are not the same formulation as their CPU counterparts (series
# expansions, a different reduction order), so the two answers agree to roughly
# single-precision rather than to the last few bits.
_LOOSE_TOLERANCE = {
    "_native_batch_norm_legit_functional",
    "special_bessel_j1",
}


class TestMetaXFlaggemsRouting:
    """Every conf route the platform claims for FlagGems is the one dispatched."""

    @pytest.mark.parametrize(
        "op,call,backend",
        sorted(
            (op, call, backend) for op, (call, backend) in _ROUTED_TO_FLAGGEMS.items()
        ),
    )
    def test_op_dispatches_to_flaggems(self, op, call, backend):
        _require_metax_flaggems()
        result = _run_logged(call)
        assert result.returncode == 0, f"{op}: {call} failed\n{result.stderr}"
        routes = _dispatch_routes(result.stderr)
        assert op in routes, (
            f"{op}: no dispatch line for {call}; the conf no longer routes it "
            f"through the dispatcher at all\n{result.stderr}"
        )
        assert routes[op] == backend, (
            f"{op}: {call} dispatched to {routes[op]}, but backends_metax.conf "
            f"routes it to FlagGems ({backend}). The op is in _ROUTED_TO_FLAGGEMS "
            "because it was measured to work on that path"
        )

    def test_measured_route_count_holds(self):
        """The conf still routes the measured number of ops to FlagGems.

        Read from the conf the runtime selected, not from the repo path, so this
        reports the table actually in use.
        """
        _require_metax_flaggems()
        conf = _active_conf()
        routes = {}
        with open(conf) as f:
            for line in f:
                name, sep, value = line.split("#")[0].partition("=")
                if sep:
                    routes[name.strip()] = value.strip()
        on_flaggems = [op for op, backend in routes.items() if backend == "flaggems"]
        assert len(on_flaggems) == _MEASURED_FLAGGEMS_ROUTES, (
            f"{conf} routes {len(on_flaggems)} ops to FlagGems, measured "
            f"{_MEASURED_FLAGGEMS_ROUTES}"
        )


class TestMetaXFlaggemsExecution:
    """The ops routed to FlagGems compute the CPU answer on flagos tensors."""

    @pytest.mark.parametrize(
        "tag,build,run",
        [
            # Each case builds its inputs on the host and moves them, so the two
            # arms consume identical values. Building twice with
            # `torch.randn(..., device=dev)` per arm -- the obvious shape -- makes
            # the comparison meaningless: it reports the difference between two
            # random draws, which is how a set of correct ops (nll_loss_forward,
            # _log_softmax, the upsample-aa backward, ...) were once recorded as
            # wrong numerics.
            ("neg", lambda: (torch.randn(64, 64),), lambda x: torch.neg(x)),
            ("abs", lambda: (torch.randn(64, 64),), lambda x: torch.abs(x)),
            ("exp", lambda: (torch.randn(64, 64),), lambda x: torch.exp(x)),
            ("rsqrt", lambda: (torch.rand(64, 64) + 0.5,), lambda x: torch.rsqrt(x)),
            ("tanh", lambda: (torch.randn(64, 64),), lambda x: torch.tanh(x)),
            (
                "silu",
                lambda: (torch.randn(64, 64),),
                lambda x: torch.nn.functional.silu(x),
            ),
            (
                "gelu",
                lambda: (torch.randn(64, 64),),
                lambda x: torch.nn.functional.gelu(x),
            ),
            (
                "amax",
                lambda: (torch.randn(64, 64),),
                lambda x: torch.amax(x, 1),
            ),
            (
                "amin",
                lambda: (torch.randn(64, 64),),
                lambda x: torch.amin(x, 1),
            ),
            (
                "sub.Tensor",
                lambda: (torch.randn(64, 64), torch.randn(64, 64)),
                lambda x, y: x - y,
            ),
            (
                "where.self",
                lambda: (torch.randn(64, 64), torch.randn(64, 64)),
                lambda x, y: torch.where(x > 0, x, y),
            ),
            (
                "embedding",
                lambda: (torch.randint(0, 3, (4, 8)), torch.randn(3, 16)),
                lambda idx, w: torch.nn.functional.embedding(idx, w),
            ),
            # The seven ops METAX_FLAGGEMS_MEASURED lifted out of the
            # flaggems_runtime_broken group (5) hold. Their execution cases are
            # the evidence for promoting them: two of the seven
            # (`linalg_ldl_solve`, `special_bessel_j1`) raised on the boxing
            # route, so for those this is the only route that works at all.
            (
                "_embedding_bag_per_sample_weights_backward",
                lambda: (
                    torch.randn(2, 4),
                    torch.randn(3, 4),
                    torch.tensor([0, 1, 2, 0]),
                    torch.tensor([0, 2]),
                    torch.tensor([0, 0, 1, 1]),
                ),
                lambda g, w, i, o, b: (
                    torch.ops.aten._embedding_bag_per_sample_weights_backward(
                        g, w, i, o, b, 0
                    )
                ),
            ),
            (
                "_native_batch_norm_legit_functional",
                lambda: (
                    torch.randn(2, 4, 4, 4),
                    torch.rand(4) + 0.5,
                    torch.randn(4),
                    torch.zeros(4),
                    torch.ones(4),
                ),
                lambda x, w, b, rm, rv: (
                    torch.ops.aten._native_batch_norm_legit_functional(
                        x, w, b, rm, rv, True, 0.1, 1e-5
                    )[0]
                ),
            ),
            (
                "binary_cross_entropy_with_logits",
                lambda: (torch.randn(6, 5), torch.rand(6, 5)),
                lambda x, t: torch.ops.aten.binary_cross_entropy_with_logits(x, t),
            ),
            (
                "linalg_ldl_solve",
                _ldl_solve_inputs,
                lambda ld, pivots, b: torch.ops.aten.linalg_ldl_solve(
                    ld, pivots, b, hermitian=True
                ),
            ),
            (
                "special_bessel_j1",
                lambda: (torch.randn(8, 6),),
                lambda x: torch.ops.aten.special_bessel_j1(x),
            ),
            (
                "unsqueeze",
                lambda: (torch.randn(4, 4),),
                lambda x: torch.ops.aten.unsqueeze(x, 1),
            ),
            (
                "unsqueeze_",
                lambda: (torch.randn(4, 4),),
                lambda x: torch.ops.aten.unsqueeze_(x.clone(), 1),
            ),
        ],
    )
    def test_values_match_cpu(self, tag, build, run):
        _require_metax_flaggems()
        torch.manual_seed(0)
        host = build()
        on_device = [t.to(DEVICE) for t in host]
        expected = run(*host)
        actual = run(*on_device)
        assert actual.device.type == "flagos"
        tol = 1e-3 if tag in _LOOSE_TOLERANCE else 1e-4
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=tol, atol=tol, msg=f"{tag} on flagos"
        )

    def test_log_softmax_backward_uses_the_vendor_override(self):
        """The generated call names the package-level entry point, not a module.

        ``_log_softmax_backward_data`` is the case that proved the point: the
        MetaX kernel supplies ``BLOCK_N`` through a ``triton.heuristics``
        decorator that the generic ``flag_gems.ops.log_softmax`` module does not
        carry, so a generated call frozen to that module path raised
        ``TypeError: dynamic_func() missing 1 required positional argument:
        'BLOCK_N'``. ``flag_gems.log_softmax_backward`` is the name the MetaX
        backend rebound at import, and it works.
        """
        _require_metax_flaggems()
        torch.manual_seed(0)
        host = torch.randn(64, 128)
        ref = host.clone().requires_grad_(True)
        torch.nn.functional.log_softmax(ref, dim=1).sum().backward()

        dev = host.to(DEVICE).requires_grad_(True)
        torch.nn.functional.log_softmax(dev, dim=1).sum().backward()

        torch.testing.assert_close(dev.grad.cpu(), ref.grad, rtol=1e-4, atol=1e-4)


class TestMetaXFlaggemsExclusions:
    """Ops measured to fail on flagos tensors must stay off the FlagGems path."""

    @pytest.mark.parametrize("op", _FORCED_OFF_FLAGGEMS)
    def test_broken_op_is_not_routed_to_flaggems(self, op):
        _require_metax_flaggems()
        with open(_active_conf()) as f:
            for line in f:
                name, sep, value = line.split("#")[0].partition("=")
                if sep and name.strip() == op:
                    assert value.strip() != "flaggems", (
                        f"{op} is routed to FlagGems in {_active_conf()}, but it "
                        "raises on a flagos tensor -- see the flaggems_runtime_broken "
                        "entry for it in scripts/codegen_ops.py"
                    )
                    return
        pytest.fail(f"{op} is not listed in {_active_conf()}")

    @pytest.mark.parametrize("op,call", sorted(_FORCED_OFF_DISPATCH.items()))
    def test_broken_op_dispatches_to_cuda(self, op, call):
        """Each of these must reach the boxing kernel, never the FlagGems one.

        Most carry a literal device-type guard -- ``special_i1e`` is one -- and
        fail with an ``AssertionError`` / ``ValueError`` naming CUDA on any
        PrivateUse1 tensor. Two of the withdrawn sixteen fail the other way: they
        raise from inside the FlagGems wrapper, or, for ``sum.out``, return a
        wrong result without raising. ``slice.Tensor`` is the exception to the
        wording rather than to the guard: its ``AssertionError`` names a dtype,
        not a device, so the call completes here only because the boxing kernel
        is the route. The assertion below is therefore the one that holds for all
        of them -- the call completes on this route, and the dispatch log names
        this route -- and the per-op failure mode of the gems arm is recorded in
        ``docs/reference/operator-support.md``.

        On MetaX the boxing kernel is ``cuda``, reached through maca's
        ``libtorch_cuda.so``.
        """
        _require_metax_flaggems()
        result = _run_logged(call)
        assert result.returncode == 0, f"{op}: {call} failed\n{result.stderr}"
        routes = _dispatch_routes(result.stderr)
        assert op in routes, f"{op}: no dispatch line for {call}\n{result.stderr}"
        assert routes[op] == "cuda", (
            f"{op} dispatched to {routes[op]}; it is pinned to the boxing kernel "
            "because the FlagGems kernel raises on a flagos tensor"
        )
