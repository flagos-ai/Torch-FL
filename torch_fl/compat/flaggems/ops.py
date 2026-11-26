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

"""Compatibility patches for FlagGems operator wrappers."""

import contextlib
import importlib
import sys

import torch

from ..._platform import platform_for
from .launch import launch_patches_apply
from . import runtime


def align_device_identity():
    """Make FlagGems' device name agree with the one torch_fl registered.

    FlagGems compares the device type of its inputs against the device string its
    own vendor backend declares before it runs any kernel, falling back to ATen
    when the two differ. On a generic NVIDIA box those are ``flagos`` (torch_fl's
    registered backend) and ``cuda`` (what FlagGems' nvidia descriptor says), so
    every op that makes that comparison fell back and never ran its kernel under
    the ``backends_nvidia.conf`` routing.

    No platform gate here: the remedy is decided by which vendor descriptor
    FlagGems resolved, and of the CUDA-compatible ones only nvidia's and hygon's
    name this accelerator differently from torch_fl while also being measured
    against the realignment. DCU needs it as much as CUDA does -- its guarded
    modules raise rather than fall back, which is the ``ValueError: i0: input
    tensor must be on cuda device`` class in the DCU survey. On every other
    vendor (Ascend, GCU, ordinary MetaX, MUSA, and other descriptors that say
    ``cuda``) the call returns immediately -- see
    ``torch_fl.accelerator.nvidia._cuda_compat.patch_flaggems_device_name``, which
    GCU mirrors in ``torch_fl.accelerator.enflame._gcu_compat``. MetaX native
    hosts are admitted separately by the wheel's build record.

    Best-effort -- the name check is a dispatch guard, so failing to realign it
    costs the routed ops their Triton kernel but must not take down device init.
    """
    try:
        from torch_fl.accelerator.nvidia._cuda_compat import patch_flaggems_device_name
        from torch_fl import _build_config

        patch_flaggems_device_name(
            native_hosts=getattr(_build_config, "NATIVE_HOSTS", False)
        )
    except Exception:
        pass


def _vector_norm_permutes(x, dims) -> bool:
    """Whether FlagGems' ``dim_compress`` would materialise a transposed copy.

    ``dim_compress`` (flag_gems/utils/shape_utils.py) builds
    ``order = batch_dims + sorted(reduced_dims, key=stride, reverse=True)`` and
    returns ``permute(order).contiguous()``. That only stays a view when ``order``
    is already the identity, which is the case for a reduction over the
    innermost dim of a contiguous tensor -- and for a full-tensor reduction,
    where every dim is in the list.
    """
    order = [i for i in range(x.ndim) if i not in dims]
    order += sorted(dims, key=lambda d: x.stride(d), reverse=True)
    return order != list(range(x.ndim))


def _flaggems_vector_norm_wrapper(original):
    """Wrap FlagGems' ``vector_norm`` so single-dim L2 avoids the transpose.

    See ``patch_vector_norm`` for why. Everything the fast path does
    not cover -- any other ``ord``, a multi-dim or full reduction, an explicit
    ``dtype``, a half input, and every reduction whose dim is already innermost --
    goes to ``original`` unchanged.
    """

    def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
        dims = None
        if dim is not None and x.ndim:
            try:
                # dim is an int for torch.linalg.vector_norm calls that name a
                # single axis, and a sequence otherwise; the aten schema
                # (OptionalIntArrayRef) does not say which.
                dims = [dim] if isinstance(dim, int) else [int(d) for d in dim]
                dims = [d % x.ndim for d in dims]
            except Exception:
                dims = None
        if (
            dims is None
            or len(dims) != 1
            or ord != 2
            or dtype is not None
            # x*x is evaluated in the input dtype, so a half input would round
            # (and, for fp16 above 256, overflow) before FlagGems' fp32
            # accumulator ever sees it. FlagGems' own kernel promotes first.
            or x.dtype not in (torch.float32, torch.float64)
            or x.numel() == 0
            or not _vector_norm_permutes(x, dims)
        ):
            return original(x, ord, dim, keepdim, dtype)
        # aten-level spellings, so the conf still decides each sub-op's route:
        # sum.dim_IntList and sqrt are both FlagGems, and sum's single-dim path
        # reduces the strided axis in place instead of permuting it.
        out = torch.sqrt(torch.sum(x * x, dim=dims, keepdim=True))
        return out if keepdim else out.squeeze(dim=dims)

    vector_norm.__wrapped__ = original
    vector_norm._torch_fl_composite_l2 = True
    return vector_norm


def _rebind_flag_gems_name(name, original, wrapper):
    """Point every module that *holds* ``name`` at ``wrapper`` instead.

    The C++ bridge resolves ``"flag_gems.<name>"`` once and caches the callable,
    and each re-export on the way down (``flag_gems``, ``flag_gems.ops``, and
    the submodule that defines it) holds its own reference, so all of them have
    to move. Matched by identity, so a module that happens to export a
    different callable under the same name is left alone.

    The test is ``__dict__`` membership rather than ``getattr``. On this
    interpreter the two are not the same question: ``transformers`` installs a
    ``_LazyModule`` for every model it ships, and ``getattr`` on one of those
    imports the submodule behind the name and warns about the alias it hands
    back. Sweeping several thousand of them costs seconds of load time and
    materialises hundreds of image-processing modules that nothing asked for.
    A module that only computes the name in ``__getattr__`` was never holding
    the reference that has to move, so ``__dict__`` is also the right test.

    ``sys`` is a module-level import here, not one of the function-local ones the
    patch functions use for their own imports: a local ``import sys`` binds a
    local name and leaves this function's ``sys`` unbound, which is a
    ``NameError`` on the dispatch path -- and every caller of this helper wraps
    it in ``except Exception: pass``, so the patch installs nothing and says
    nothing.
    """
    for mod in list(sys.modules.values()):
        try:
            if mod.__dict__.get(name) is original:
                mod.__dict__[name] = wrapper
        except Exception:
            continue


def patch_vector_norm(accelerator: str):
    """Stop FlagGems' L2 ``vector_norm`` from materialising a transposed copy.

    ``flag_gems.ops.vector_norm`` sends every partial reduction through
    ``dim_compress``, which permutes the reduced dim to innermost and then calls
    ``.contiguous()`` on the result. On a strided reduction that permute is not
    the identity, so the tensor is gathered into a fresh buffer a few hundred
    bytes at a time. Measured on a DCU bw1000 with ``(1, 144, 1, 2048, 2048)``
    fp32 (2.42 GB), against the same device's 1.34 TB/s on a plain contiguous
    copy of the same buffer:

        dim_compress(x, [1])          354869 us    13.6 GB/s
        x.clone()                       3606 us  1339.8 GB/s
        flag_gems.vector_norm(x,2,[1]) 357307 us
        sqrt(sum_dim(x*x, [1]))           5801 us

    So 99.3% of the op is the copy, and the ``l2_norm_kernel`` it was copied for
    is ~2.4 ms. FlagGems' own ``sum`` does not do this -- its single-dim path
    reduces the strided axis in place over a plain ``contiguous()`` -- so the
    same reduction spelled ``sqrt(sum(x*x, dim=dim, keepdim=True))`` is 61x
    faster and keeps every sub-op on the route the conf already gives it
    (``sum.dim_IntList`` and ``sqrt`` are both FlagGems). Nothing is rerouted to
    the CUDA-boxing path: the op still runs FlagGems Triton kernels.

    Scoped to the configuration that was measured -- DCU, L2, one dim -- and
    gated on the route actually being FlagGems. Upstream workaround rather than
    a design: remove it once FlagGems' vector_norm stops permuting. Best-effort,
    like ``align_device_identity``: an unpatched FlagGems is slow here,
    not broken, so this must not be able to take down device init.
    """
    try:
        # The build record names the vendor ("hygon"); the platform is "dcu".
        if platform_for(accelerator) != "dcu":
            return
        if not runtime.routes_to_flaggems():
            return
        import flag_gems.ops as _gems_ops

        # ``flag_gems.ops.__init__`` re-exports the function under the
        # submodule's own name, so ``from flag_gems.ops import vector_norm``
        # yields the *function* and ``getattr``-ing ``.vector_norm`` off it
        # finds nothing. Take the defining module out of sys.modules instead.
        _gems_module = sys.modules.get("flag_gems.ops.vector_norm")
        original = getattr(_gems_module, "vector_norm", None)
        if original is None:
            original = getattr(_gems_ops, "vector_norm", None)
        if original is None or getattr(original, "_torch_fl_composite_l2", False):
            return
        wrapper = _flaggems_vector_norm_wrapper(original)
        # The C++ bridge resolves "flag_gems.vector_norm" once and caches the
        # callable, and every re-export above holds its own reference, so rebind
        # every module that currently points at the original.
        _rebind_flag_gems_name("vector_norm", original, wrapper)
        if _gems_module is not None:
            _gems_module.vector_norm = wrapper
    except Exception:
        pass


def _pow_exponent_is_two(exponent) -> bool:
    """Whether a ``Scalar`` exponent is exactly 2, without trusting its type."""
    try:
        return float(exponent) == 2.0
    except Exception:
        return False


def _flaggems_pow_scalar_square_wrapper(original, square, native_square=None):
    """Wrap FlagGems' ``pow_tensor_scalar`` so an exponent of 2 is a multiply.

    See ``patch_pow_scalar_square`` for why. Every other exponent, and
    every input that is not floating point, goes to ``original`` unchanged.
    ``square`` is the value of ``flag_gems.mul`` seen when the patch was
    installed; it is only a fallback, because the patch can run before that name
    exists (see below).
    """

    def pow_tensor_scalar(A, exponent):
        try:
            fast = A.dtype.is_floating_point and _pow_exponent_is_two(exponent)
        except Exception:
            fast = False
        if fast:
            if native_square is not None:
                return native_square(A, A)
            import flag_gems

            mul = getattr(flag_gems, "mul", None) or square
            if mul is not None:
                return mul(A, A)
        return original(A, exponent)

    pow_tensor_scalar.__wrapped__ = original
    pow_tensor_scalar._torch_fl_square_via_mul = True
    return pow_tensor_scalar


def patch_pow_scalar_square(accelerator: str):
    """Compute ``x.pow(2)`` as ``x * x`` instead of one ``powf`` each.

    ``pow.Tensor_Scalar`` reaches ``flag_gems.pow_tensor_scalar``, which on this
    backend is not upstream's ``flag_gems/ops/pow.py`` at all: the backend
    registrar swaps in ``flag_gems/runtime/backend/_hygon/ops/pow.py``, and that
    copy evaluates ``_pow(x.to(tl.float64), exponent.to(tl.float64))`` for every
    non-half input -- an fp64 libdevice ``powf`` per element. The exponent is a
    runtime operand, so the kernel is the same whether it was handed 2.0 or
    1.37. Measured on a DCU bw1000, fp32 ``(1, 4122, 32, 128)`` (16.9M elements,
    67.6 MB), pair of device events around a back-to-back launch loop against
    the same loop's host time:

        pow_tensor_scalar(z, 2.0)   host  185.6 us   device  418.3 us
        flag_gems.mul(z, z)         host   20.1 us   device  104.8 us

    and on bf16, where the same two ops are 384.6/169.7 and 56.5/20.1. The
    replacement is also at least as accurate: ``x * x`` is correctly rounded,
    whereas ``exp2(2 * log2(x))`` is not, and the two agree bit for bit at 0,
    -0, inf and nan.

    It is spelled with FlagGems' own ``mul`` rather than ``torch.mul`` so the
    whole op stays on FlagGems Triton kernels. That is a deliberate ~50 us: the
    conf routes ``mul.Tensor`` to the CUDA-boxing path, whose kernel is about
    twice as fast on fp32, and taking it would move a routed op off FlagGems,
    which is what this patch exists to avoid.

    ``flag_gems.mul`` is read at call time rather than captured here, because
    this function can run in the middle of ``import flag_gems``. Triton asks for
    the active device while ``flag_gems/__init__.py`` is still importing
    ``flag_gems.fused``, that query goes through the PrivateUse1 lazy init back
    into ``_lazy_init``, and the top-level names -- ``mul`` among them -- are not
    bound until ``from flag_gems.ops import *`` a few lines later, so on that
    first pass there is nothing to capture. By the time anything is dispatched
    ``flag_gems.mul`` is the function the bridge would have called, so binding it
    late costs a dict lookup and removes the ordering question entirely.

    Reached through ``flag_gems.pow_tensor_scalar`` rather than through the
    defining module, because those are two different functions here: the
    backend override is what the C++ bridge resolves, and upstream's copy is
    left alone only if nothing else calls it. Both are rebound if both are
    present.

    Scoped to the configuration that was measured -- DCU, on a conf that routes
    to FlagGems, exponent exactly 2, floating-point input -- and gated on the
    route actually being FlagGems. Best-effort, like
    ``align_device_identity``: an unpatched ``pow`` is slow here, not
    broken, so this must not be able to take down device init.
    """
    try:
        # The build record names the vendor ("hygon"); the platform is "dcu".
        if platform_for(accelerator) != "dcu":
            return
        if not runtime.routes_to_flaggems():
            return
        import flag_gems
        import flag_gems.ops as _gems_ops

        # The backend-resolved name first: that is the one the bridge calls.
        # The defining module can hold a different function, the way
        # ``_hygon/ops/pow.py`` shadows ``flag_gems/ops/pow.py``, so take it
        # too and let the identity sweep below find every reference to each.
        candidates = []
        resolved = getattr(flag_gems, "pow_tensor_scalar", None)
        if resolved is not None:
            candidates.append(resolved)
        _gems_module = sys.modules.get("flag_gems.ops.pow")
        upstream = getattr(_gems_module, "pow_tensor_scalar", None)
        if upstream is None:
            upstream = getattr(_gems_ops, "pow_tensor_scalar", None)
        if upstream is not None and upstream is not resolved:
            candidates.append(upstream)

        # And the vendor's own copy, which is the one that ends up in the
        # registry: this runs from inside ``import flag_gems``, while
        # ``SpecOpRegistrar.apply`` is a few lines further down the same file,
        # and that registrar reads its functions out of ``_{vendor}.ops`` with
        # ``inspect.getmembers`` at that later moment. Rebinding the vendor
        # package now is therefore what decides what gets published -- the name
        # the sweep above fixes on ``flag_gems`` is overwritten either way.
        try:
            from flag_gems import runtime as _gems_runtime

            vendor = _gems_runtime.device.vendor_name
        except Exception:
            vendor = None
        if vendor:
            try:
                vendor_ops = importlib.import_module(f"_{vendor}.ops")
            except Exception:
                vendor_ops = None
            vendor_pow = getattr(vendor_ops, "pow_tensor_scalar", None)
            if vendor_pow is not None and all(vendor_pow is not c for c in candidates):
                candidates.append(vendor_pow)

        if not candidates:
            return
        square = getattr(flag_gems, "mul", None)
        from ... import _build_config

        # Native source selection belongs to the ATen route. Calling the
        # registry callable directly bypasses its compiled pointwise entry.
        native_square = (
            torch.mul if getattr(_build_config, "NATIVE_HOSTS", False) else None
        )
        # The C++ bridge resolves "flag_gems.pow_tensor_scalar" once and caches
        # the callable, and every re-export above holds its own reference, so
        # rebind every module that currently points at an original -- matched
        # by identity, the way patch_vector_norm does it.
        for original in candidates:
            if getattr(original, "_torch_fl_square_via_mul", False):
                continue
            wrapper = _flaggems_pow_scalar_square_wrapper(
                original, square, native_square
            )
            _rebind_flag_gems_name("pow_tensor_scalar", original, wrapper)
    except Exception:
        pass


# A contiguous last-dim reduction at or below this width is short enough that
# one CTA per row is the wrong shape for it; see the patch for the measurement.
_MEAN_TILED_MAX_N = 1024


def _mean_is_last_dim(inp, dim) -> bool:
    """Whether ``mean_dim`` is reducing a contiguous tensor's last axis."""
    if not isinstance(inp, torch.Tensor) or inp.ndim < 1:
        return False
    if not inp.dtype.is_floating_point or not inp.is_contiguous() or inp.numel() == 0:
        return False
    if dim is None:
        return False
    try:
        # dim is an int for calls that name a single axis and a sequence
        # otherwise; the aten schema (OptionalIntArrayRef) does not say which.
        dims = [dim] if isinstance(dim, int) else [int(d) for d in dim]
    except Exception:
        return False
    if len(dims) != 1 or not -inp.ndim <= dims[0] < inp.ndim:
        # The range test is not redundant with the modulo below: ``%`` normalises
        # a negative dim, it does not reject an out-of-range one, and an
        # out-of-range dim would otherwise read as the last axis (dim 3 on a 2-D
        # input) and return a mean over the wrong axis instead of raising the way
        # the composite it replaces does.
        return False
    return dims[0] % inp.ndim == inp.ndim - 1 and inp.shape[-1] <= _MEAN_TILED_MAX_N


def _flaggems_mean_last_dim_wrapper(original, kernel, cdiv, device_ctx):
    """Wrap FlagGems' ``mean_dim`` so a row mean goes to the tiled kernel.

    See ``patch_mean_last_dim`` for why. Everything the fast path does
    not cover -- an explicit ``dtype``, a non-float or non-contiguous input, a
    reduction that is not over the last axis, a wide row -- goes to ``original``
    unchanged.
    """

    def mean_dim(inp, dim=None, keepdim=False, *, dtype=None):
        if inp.numel() == 0:
            output_dtype = inp.dtype if dtype is None else dtype
            if not (output_dtype.is_floating_point or output_dtype.is_complex):
                raise RuntimeError(
                    "mean(): input or output dtype must be floating point or complex"
                )
            dims = (
                list(range(inp.ndim))
                if dim is None
                else [dim]
                if isinstance(dim, int)
                else list(dim)
            )
            if not dims:
                dims = list(range(inp.ndim))
            normalized = []
            for axis in dims:
                if not -inp.ndim <= axis < inp.ndim:
                    raise IndexError("mean(): dimension out of range")
                axis %= inp.ndim
                if axis in normalized:
                    raise RuntimeError("mean(): dimension appears multiple times")
                normalized.append(axis)
            shape = [
                1 if keepdim and i in normalized else n
                for i, n in enumerate(inp.shape)
                if keepdim or i not in normalized
            ]
            return torch.full(
                shape, float("nan"), dtype=output_dtype, device=inp.device
            )
        if dtype is None and _mean_is_last_dim(inp, dim):
            n = inp.shape[-1]
            m = inp.numel() // n
            # mean_dim_kernel writes one value per row, so its out is (M, 1)
            # regardless of the input's rank.
            out = torch.empty((m, 1), dtype=inp.dtype, device=inp.device)
            grid = lambda meta: (cdiv(m, meta["BLOCK_M"]),)  # noqa: E731
            with device_ctx(inp.device):
                kernel[grid](inp, out, m, n)
            out = out.view(list(inp.shape[:-1]) + [1])
            return out if keepdim else out.squeeze(-1)
        return original(inp, dim, keepdim, dtype=dtype)

    mean_dim.__wrapped__ = original
    mean_dim._torch_fl_tiled_row_mean = True
    return mean_dim


def patch_mean_last_dim(accelerator: str):
    """Send a short last-dim ``mean`` to FlagGems' tiled reduction kernel.

    ``flag_gems.ops.mean.mean_dim_comm`` picks its kernel by ``K = numel/M/N``,
    the length of the *unreduced* slice: ``K >= 1024`` gets the vectorised
    kernel, ``K > 1`` the tiled one, and ``K == 1`` -- a reduction over the last
    axis of a contiguous tensor -- gets ``mean_dim_kernel_inner``, which is one
    CTA per row. On a short row that is a CTA per 512 bytes, and the launch cost
    of 131904 of them swamps the reduction. Measured on a DCU bw1000, fp32
    ``(1, 4122, 32, 128)`` (131904 rows of 128), the same event-pair split as
    the pow patch:

        mean_dim(z, -1, True)   [shipped]        host   77.7 us   device  443.5 us
        mean_dim_kernel tiled                    host   47.8 us   device   59.8 us

    and on bf16 the shipped kernel is the same 443.5 us. Both kernels sum in
    fp32 and divide by N, so bf16 agrees bit for bit; fp32 differs by fp
    reassociation only (max |delta| 1.5e-8 on unit-variance input, which is
    below the bf16 rounding the model then applies). The tiled kernel is the
    one ``mean_dim_comm`` already uses for multi-dim reductions and for
    ``sum``/``amax``/``prod``, so this is a dispatch fix, not a new kernel.

    Measured flat across M -- 46 us against 76 us at M=1 as well as at M=16384
    -- so it needs no lower bound on the row count, only an upper bound on the
    row width. Scoped to the configuration that was measured -- DCU, on a conf
    that routes to FlagGems -- and gated on the route actually being FlagGems.
    Best-effort, like ``align_device_identity``: an unpatched ``mean``
    is slow here, not broken, so this must not be able to take down device init.
    """
    try:
        # The build record names the vendor ("hygon"); the platform is "dcu".
        if platform_for(accelerator) != "dcu":
            return
        if not runtime.routes_to_flaggems():
            return
        import flag_gems
        import flag_gems.ops as _gems_ops

        # Resolve the way the bridge does -- off the top level -- so a backend
        # that ships its own mean_dim is patched where it actually is, and take
        # the kernel and helpers from whichever module defines that function.
        original = getattr(flag_gems, "mean_dim", None)
        if original is None:
            original = getattr(_gems_ops, "mean_dim", None)
        if original is None:
            return
        if getattr(original, "_torch_fl_tiled_row_mean", False):
            return
        _gems_module = sys.modules.get(getattr(original, "__module__", ""))
        kernel = getattr(_gems_module, "mean_dim_kernel", None)
        if kernel is None:
            # Same re-export trap as the vector_norm patch: ``flag_gems.ops``
            # binds the *function* under the submodule's name, so the defining
            # module has to come out of sys.modules.
            _gems_module = sys.modules.get("flag_gems.ops.mean")
            kernel = getattr(_gems_module, "mean_dim_kernel", None)
        if kernel is None:
            return
        # mean_dim_comm wraps its launches in this context; keep the same
        # ordering against any pending device switch. cdiv comes off the same
        # module so the wrapper does not need its own triton import.
        device_ctx = getattr(_gems_module, "torch_device_fn", None)
        device_ctx = getattr(device_ctx, "device", None) or contextlib.nullcontext
        cdiv = getattr(getattr(_gems_module, "triton", None), "cdiv", None)
        if cdiv is None:
            import triton

            cdiv = triton.cdiv
        wrapper = _flaggems_mean_last_dim_wrapper(original, kernel, cdiv, device_ctx)
        _rebind_flag_gems_name("mean_dim", original, wrapper)
        if _gems_module is not None:
            _gems_module.mean_dim = wrapper
    except Exception:
        pass


# Dispatcher kernels registered from this module are held by the torch.library
# Library object that registered them, so the objects are kept for the life of
# the process rather than left to the garbage collector.
_PATCH_LIBS = []


def patch_normal(accelerator: str) -> None:
    """Forward the generator and layout when calling the FlagGems normal kernel."""
    if accelerator != "nvidia":
        return
    from flag_gems.ops.normal import normal_distribution, transform_func_float_float

    from torch_fl import native_host

    def normal_(self, mean=0.0, std=1.0, *, generator=None):
        if not std >= 0:
            raise RuntimeError("normal_ expects a nonnegative standard deviation")
        if not self.numel():
            return self
        sample = normal_distribution
        transform = transform_func_float_float
        if native_host._hosts is not None:
            sample = native_host._hosts.resolve(sample)
            transform = native_host._hosts.resolve(transform)
        with torch.flagos.device(self.device):
            output = self
            if not self.is_contiguous():
                output = torch.empty_like(self, memory_format=torch.contiguous_format)
            sample(output.shape, output.device, generator=generator, out=output)
            transform(output, std, mean, out0=output)
            return self if output is self else self.copy_(output)

    library = torch.library.Library("aten", "IMPL", "PrivateUse1")
    library.impl("normal_", normal_)
    _PATCH_LIBS.append(library)


def _flagos_index_put_roundtrip(self, indices, values, accumulate):
    """Scatter on the CPU when the caller permits the legacy fallback."""
    self_cpu = self.cpu()
    values_cpu = values.cpu()
    indices_cpu = [i.cpu() if isinstance(i, torch.Tensor) else i for i in indices]
    # The Tensor method rejects the None entry produced by x[:, mask] = v.
    torch.ops.aten.index_put_(self_cpu, indices_cpu, values_cpu, accumulate)
    self.copy_(self_cpu)
    return self


def patch_index_put(accelerator: str):
    """Run ``index_put_`` on FlagGems' Triton kernel, not on a CPU round-trip.

    The handwritten C++ registration retains a legacy CPU route. This patch
    prefers the published FlagGems kernel on supported platforms and checks
    the fallback policy before attempting a host round trip.

    Measured on a DCU bw1000, in the Qwen-Image-2.1 denoise loop, where the op
    is the largest exclusive entry in the steady-state census (24 calls over
    four steps at 15501.54 us each, ahead of every FlagGems op), against
    FlagGems' kernel on the same shapes with fresh operands:

        shape                                       round-trip   FlagGems
        (4122,) <- (4096,)              int            0.140 ms    0.155 ms
        (4122,) <- (4096,)              bool mask      0.173 ms    0.507 ms
        (4122,) <- scalar                              0.150 ms    0.153 ms
        (1,4122) <- (26,)                              0.085 ms    0.180 ms
        (1,4122,4096) [:, i] <- (4096,4096)      f32   68.583 ms    0.291 ms
        (1,4122,4096) [:, m] <- (4096,4096)      f32   72.360 ms    0.638 ms
        (1,4122,4096) [:, m] <- (4096,4096)      bf16  43.442 ms    0.578 ms

    The census mean is over that bimodal set: the model issues six calls a step
    -- five narrow ones that are a wash either way, and one wide
    ``joint_hidden_states[:, image_pad_mask] = hidden_states`` that is 70 of
    the step's ~93 ms. The win is on the wide shapes, and it is 75x-236x there.
    All of them agree bitwise with the round-trip (``equal=True``,
    ``max|delta|=0``, the same for ``accumulate=True``); the operand transfers
    and the CPU scatter are each tens of microseconds when timed on their own,
    so what the wide shapes cost is the round-trip, not the data.

    End to end, on the same six-step driver, against the run that differs only
    by this patch: the op falls from 15501.54 to 388.79 us per call on the
    steady block and from 16720.84 to 381.17 on the last step, the steady step
    wall from 1.272 to 1.180 s, and the last step's exclusive sum by 0.093 s --
    against 0.098 s for the op alone, so nothing else moved. The rendered image
    is byte-identical to the unpatched one (md5 6fc5c62c451ecf5772f11bb0e1652cfd,
    1875231 bytes, both).

    Measured again on an H100, same op, same shape: the round-trip is 7.96 ms
    against FlagGems' 0.22 ms -- 36x -- on ``(1, 4160, 3072)`` bf16 assigned
    through a ``(4160,)`` bool mask, which is the shape the 2.1 forward issues
    before its block stack. The H100 figure for the same assignment on the
    vendor route is 0.10 ms, so this is the round-trip's 26 MB host copy each
    way and not a kernel difference.

    Narrows to FlagGems rather than away from it: the op keeps Triton kernels
    and never takes the CUDA-boxing route. Scoped by the shared
    ``launch_patches_apply`` gate -- see
    ``_FLAGGEMS_LAUNCH_PATCH_BUILDS`` -- and installed through ``torch.library``
    rather than in C++ because the route cannot be stated in the conf: the conf
    enumerates the generated registration list, which this hand-registered op is
    deliberately not in. Upstream workaround rather than a design; remove it once
    the wrapper stops round-tripping. Best-effort, like
    ``align_device_identity``: an unpatched ``index_put_`` is slow
    here, not broken, so this must not be able to take down device init.
    """
    try:
        if accelerator == "nvidia" and hasattr(
            torch.ops.flagos, "_activate_native_nvidia"
        ):
            return
        if not launch_patches_apply(accelerator):
            return
        import flag_gems

        gems_index_put = getattr(flag_gems, "index_put_", None)
        if gems_index_put is None:
            return

        def _index_put_(self, indices, values, accumulate=False):
            try:
                gems_index_put(self, list(indices), values, accumulate)
            except Exception:
                if not torch.ops.flagos.cpu_fallback_allowed():
                    raise
                return _flagos_index_put_roundtrip(self, indices, values, accumulate)
            return self

        def _index_put_impl_(self, indices, values, accumulate=False, unsafe=False):
            # FlagGems' index_put_ is the same scatter without the
            # duplicate-index guard, so it is the implementation for both. Its
            # _index_put_impl_ is not used: it takes a separate path for the
            # single-bool-mask case that was never measured here.
            return _index_put_(self, indices, values, accumulate)

        lib = torch.library.Library("aten", "IMPL")
        lib.impl("index_put_", _index_put_, "PrivateUse1")
        lib.impl("_index_put_impl_", _index_put_impl_, "PrivateUse1")
        _PATCH_LIBS.append(lib)
    except Exception:
        pass


def patch_layer_norm_backward(accelerator: str) -> None:
    """Correct flag_gems' layer-norm backward for inputs above 2-D.

    flag_gems' wrapper derives the reduced-row count from ``input.shape[0]``:

        M = input.shape[0]
        N = input.numel() // M

    That is only the row count for a 2-D input. On a 3-D transformer tensor
    ``shape[0]`` is the batch size, so the dX kernel's grid is short by the
    sequence length and the weight/bias reduction sums the wrong number of
    rows. Measured on A100 (flag_gems 5.4.0, no boxing kernels compiled, so
    ``native_layer_norm_backward`` can only route to the FlagGems Python path):
    for a (13, 7, 32) input with a ones/zeros affine the bias gradient must be
    the constant 91.0 and comes back with max 13.0 -- exactly ``shape[0]``.
    The error is large, not a rounding artifact (dX off by 4.96, dbeta by 96.5)
    and it makes the result non-reproducible: 40 identical forward/backward
    passes over one frozen BERT gave 23 distinct losses, while vendor CUDA
    torch on the same box gave 1.

    The kernels are correct for a 2-D input -- the 2-D case is bit-exact -- so
    the fix is to feed the existing wrapper a contiguous (M, N) view, which
    makes its own M/N derivation right, and reshape dX back. That relies only
    on the 2-D contract the wrapper already satisfies, rather than restating
    its kernel launch code here, so it keeps working as flag_gems evolves.

    Cross-platform: this is a flag_gems defect, not a CUDA one, but it is
    deliberately not applied to every platform. Only the routing decides
    whether the op reaches this wrapper at all, and a platform whose conf keeps
    ``native_layer_norm_backward`` on its own kernels (metax, dcu, ppu, ascend)
    never calls it. Installing it unconditionally would replace a correct
    vendor kernel with the patched gems one on those platforms, which is a
    behaviour change this evidence does not cover. The gate is therefore the
    same one the other flag_gems patches use -- some op in the selected conf
    routes to a FlagGems path -- and it is a no-op wherever the op is not
    routed there.

    Remove this once the wrapper computes M from ``normalized_shape`` itself.
    """
    if not runtime.routes_to_flaggems():
        return
    if accelerator not in ("nvidia", "", "thead", "mthreads", "tsingmicro"):
        return

    try:
        import math

        import flag_gems
        import flag_gems.ops.layernorm as _layernorm

        _orig = flag_gems.layer_norm_backward

        def _patched(
            grad_out,
            input,
            normalized_shape,
            mean,
            rstd,
            weight=None,
            bias=None,
            output_mask=None,
        ):
            if input.dim() <= 2:
                return _orig(
                    grad_out,
                    input,
                    normalized_shape,
                    mean,
                    rstd,
                    weight,
                    bias,
                    output_mask,
                )
            shape = tuple(input.shape)
            n = math.prod(normalized_shape)
            m = input.numel() // n
            dx, dweight, dbias = _orig(
                grad_out.reshape(m, n),
                input.reshape(m, n),
                [n],
                mean,
                rstd,
                weight,
                bias,
                output_mask,
            )
            if dx is not None:
                dx = dx.reshape(shape)
            return dx, dweight, dbias

        # The generated kernel calls the package-level name
        # ("flag_gems.layer_norm_backward" in flaggems_python_kernels.cc) and
        # resolves it on first use, so patch both bindings before any op runs.
        flag_gems.layer_norm_backward = _patched
        _layernorm.layer_norm_backward = _patched
    except Exception:
        # FlagGems remains optional; a missing or renamed entry point must not
        # make torch_fl unimportable.
        pass
