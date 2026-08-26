#!/usr/bin/env python3
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
Codegen for torch_fl CUDA operators.

Reads:
  - torch_fl/configs/backends_cuda.conf (op list + backend mapping)
  - PyTorch native_functions.yaml (via torchgen)

Generates (into csrc/aten/generated/):
  - ops.h            shared dispatcher declarations (typedef + DECLARE_DISPATCHER)
  - ops.cc           dispatcher definitions (ADD_IMPL_TO_DISPATCHER)
  - cuda_kernels.cc  CUDA boxing kernels + REGISTER_IMPL_TO_DISPATCHER
  - register.inc     wrapper functions + m.impl() lines for register.cc

Authoritative signature source: torchgen's faithful C++ signature
(exploded TensorOptions form, required for PrivateUse1 boxing dispatch).
We never hand-roll a type table; torchgen produces the exact `at::` signature.

Naming convention (schema-driven, PyTorch-aligned):
  "add.Tensor"           -> AddTensorFn                 / add_tensor_dispatcher
  "add_.Tensor"          -> AddInplaceTensorFn          / add_inplace_tensor_dispatcher
  "mm.out"               -> MmOutFn                     / mm_out_dispatcher
  "_foreach_add_.Scalar" -> ForeachAddInplaceScalarFn   / foreach_add_inplace_scalar_dispatcher

Categories (detected from torchgen metadata):
  functional_pure    standard DeviceBoxingGuard + at::op call
  inplace            box + at::op_ inplace + unbox (returns Tensor& or void)
  out_variant        box(self.., out) + at::op_out(out, ...) + return out
  tuple_return       box + call + unbox each tuple element
  foreach_tensorlist TensorListBoxingGuard + call (+ unbox vec for non-inplace)
  factory            create device tensor directly (no input boxing)
  special_optlist    box self + each optional<Tensor> in the list + call + unbox
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


# TensorList C++ spelling (IListRef vs ArrayRef) must match what PyTorch itself
# registered for the op, or dispatcher registration crashes at import with
# "Mismatch in kernel C++ signatures".
#
# The authoritative rule is torchgen's own (torchgen/context.py):
#     use_ilistref_for_tensor_lists = f.part_of_structured_group
# i.e. an op that is part of a structured group uses IListRef (e.g. aten::cat);
# everything else (all _foreach_*, stack, _amp_foreach_*, ...) uses ArrayRef.
# Using this predicate instead of a hand-maintained set classifies ALL TensorList
# ops correctly in one shot, across every torch version.
def should_use_arrayref(func):
    """True -> emit ArrayRef; False -> emit IListRef. Mirrors torchgen's rule
    use_ilistref_for_tensor_lists = func.part_of_structured_group."""
    return not func.part_of_structured_group


# ============================================================================
# Full-CUDA enumeration mode (--all-cuda)
# ============================================================================
#
# Instead of reading a hand-maintained op list from backends_cuda.conf, the
# full mode enumerates EVERY leaf CUDA operator from native_functions.yaml and
# generates a boxing kernel for each. Ops the current templates cannot safely
# express are skipped and fall through to the existing cpu_fallback (functional,
# just slower), so coverage strictly grows and nothing regresses.
#
# Ops already registered by hand in csrc/aten/register.cc. Generating these
# would collide at registration (duplicate m.impl) -> MUST be excluded.
MANUAL_REGISTERED_OPS = {
    "empty.memory_format",
    "empty_strided",
    "as_strided",
    "resize_",
    "_reshape_alias",
    "_copy_from",
    "_copy_from_and_resize",
    "copy_",
    "_local_scalar_dense",
    "set_.source_Tensor",
    "set_.source_Storage",
    "set_.source_Storage_storage_offset",
    "view",
    "contiguous",
    "clone",
    "_to_copy",
    "index_put_",
    "_index_put_impl_",
    "record_stream",
}


# Some backend CUDA kernels require a plain-contiguous input even though ATen's
# reference impl (and the native CUDA path) would silently normalize the layout
# first. When a boxed op receives a channels-last / strided tensor, the backend
# kernel asserts (e.g. native_group_norm: "Expected X.is_contiguous(memory_format)
# to be true"). This map names, per op, which Tensor args must be forced to
# contiguous layout *before* DeviceBoxingGuard runs -- boxing rewrites the tensor's
# device metadata, so the copy has to happen while the tensor still lives on flagos.
# is_contiguous() short-circuits to a no-op when the input is already contiguous,
# so the only cost is on the exact non-contiguous inputs that would otherwise crash.
_CONTIGUOUS_TENSOR_ARGS_BY_OP = {
    "native_group_norm": ("input",),
}


def _contiguous_prelude(op, args):
    """Return (prelude_lines, rename_map) for op args listed in
    _CONTIGUOUS_TENSOR_ARGS_BY_OP. prelude_lines declares `<arg>_contiguous`
    locals (emitted before the DeviceBoxingGuard); rename_map maps the original
    arg name to the contiguous local so the guard and the at:: call use it."""
    targets = _CONTIGUOUS_TENSOR_ARGS_BY_OP.get(op)
    if not targets:
        return "", {}
    arg_names = {n for _, n in args}
    lines = ""
    rename = {}
    for name in targets:
        if name not in arg_names:  # guard against schema drift
            continue
        local = f"{name}_contiguous"
        lines += f"  at::Tensor {local} = {name}.is_contiguous() ? {name} : {name}.contiguous();\n"
        rename[name] = local
    return lines, rename


def _delegate_target_has_cuda(func, funcs, cuda_index):
    """A structured_delegate op routes its CUDA kernel through the named target
    (e.g. add.Tensor -> add.out). Return True if that target has a CUDA kernel."""
    deleg = getattr(func, "structured_delegate", None)
    if deleg is None:
        return False
    target = funcs.get(str(deleg))
    return target is not None and cuda_index.has_kernel(target)


def cuda_supported(func, funcs, cuda_index):
    """
    True if this op can be executed on CUDA (directly or by decomposition),
    matching the union that makes the enumeration a superset of the existing
    hand-written 71-op conf:
      1. direct CUDA kernel (functional/out leaf), OR
      2. structured_delegate whose target has a CUDA kernel (add/mm/silu_backward
         route their kernel through the .out / .grad_input variant), OR
      3. CompositeExplicitAutograd kernel (sort/abs/div.Scalar decompose into
         CUDA sub-ops).
    structured_delegate is checked BEFORE the composite_implicit exclusion so
    ops that carry both flags (e.g. silu_backward) are not dropped.
    """
    if cuda_index.has_kernel(func):
        return True
    if _delegate_target_has_cuda(func, funcs, cuda_index):
        return True
    if func.has_composite_explicit_autograd_kernel:
        return True
    return False


# Ops that are CompositeImplicitAutograd (so normally decomposed above our
# dispatch key and skipped) but that we WANT to intercept with a fused backend
# kernel. Registering a PrivateUse1 kernel for these overrides the composite
# decomposition (verified: F.rms_norm and aten._fused_rms_norm both land on the
# PrivateUse1 impl). The backend kernel is hand-written (aclnnRmsNorm) since the
# tuple(output, rstd) + normalized_shape semantics no codegen category expresses.
# Without this, HF's Qwen3RMSNorm decomposes into ~6 elementwise ops + 2 dtype
# casts per layer (the eager decode hot path).
FORCE_INCLUDE_OPS = {
    "_fused_rms_norm",
}


def enumerate_all_cuda_ops(nf, funcs, cuda_index):
    """
    Returns (kept_ops, skipped) where kept_ops is the list of op-name strings to
    generate and skipped is {reason: [op, ...]}.

    Skip reasons:
      manual      already registered by hand in register.cc
    composite_implicit ops are excluded up front: PyTorch decomposes them ABOVE
    our dispatch key into leaf ops we already box, so registering them is both
    unnecessary and risky. structured_delegate ops survive that exclusion.
    Ops in FORCE_INCLUDE_OPS bypass both the cuda_supported and composite checks
    so a hand-written backend kernel can intercept them.
    """
    kept = []
    skipped = defaultdict(list)
    for func in nf.native_functions:
        op = str(func.func.name)

        if op in FORCE_INCLUDE_OPS:
            if op not in MANUAL_REGISTERED_OPS:
                kept.append(op)
            continue

        if not cuda_supported(func, funcs, cuda_index):
            continue

        # composite_implicit (and NOT structured_delegate) -> decomposed above us
        if (
            func.has_composite_implicit_autograd_kernel
            and getattr(func, "structured_delegate", None) is None
            and not func.has_composite_explicit_autograd_kernel
        ):
            continue

        if op in MANUAL_REGISTERED_OPS:
            skipped["manual"].append(op)
            continue

        # Multi-output out-variants are now handled: gen_out_variant calls
        # at::<base>_outf(<all args faithful order>) and unboxes each mutable
        # Tensor& out, so no per-out-count skip is needed.

        kept.append(op)
    return kept, skipped


try:
    import torchgen
    from torchgen.gen import parse_native_yaml
    from torchgen import local
except ImportError:
    print("Error: torchgen not found. Install torch>=2.0 and pyyaml.", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Naming conventions (schema-driven, PyTorch-aligned)
# ============================================================================


def schema_to_cpp_name(op_name: str) -> Tuple[str, str]:
    """
    (fn_type PascalCase, dispatcher_name snake_case) from a schema op name.

      "add.Tensor"           -> ("AddTensorFn", "add_tensor_dispatcher")
      "add_.Tensor"          -> ("AddInplaceTensorFn", "add_inplace_tensor_dispatcher")
      "mm.out"               -> ("MmOutFn", "mm_out_dispatcher")
      "_foreach_add_.Scalar" -> ("ForeachAddInplaceScalarFn", "foreach_add_inplace_scalar_dispatcher")
    """
    parts = op_name.split(".")
    base = parts[0]
    variant = parts[1] if len(parts) > 1 else None

    is_foreach = base.startswith("_foreach_")
    stripped = base.lstrip("_")
    n_lead = len(base) - len(stripped)  # leading underscores: _conv vs conv
    is_inplace = stripped.endswith("_")
    base_clean = stripped.rstrip("_") if is_inplace else stripped

    # Disambiguation token for leading underscores (private/internal ops).
    # foreach ops keep the conventional single leading '_' implicit (unique via
    # the "Foreach" prefix / "foreach_" core), so they are exempt to keep the
    # 71-op names stable. Every other leading underscore is preserved so
    # `_convolution` and `convolution` (both leaf CUDA ops) do not collide.
    priv_pascal = "" if is_foreach else "Priv" * n_lead
    priv_snake = "" if is_foreach else "priv_" * n_lead

    # --- PascalCase type name ---
    if is_foreach:
        core = base_clean[len("foreach_") :]
        type_base = "Foreach" + "".join(w.capitalize() for w in core.split("_") if w)
    else:
        type_base = "".join(w.capitalize() for w in base_clean.split("_") if w)
    if is_inplace:
        type_base += "Inplace"
    # A trailing underscore on the variant (e.g. "out_") marks a mutating variant
    # distinct from the non-mutating one ("out"); preserve it so range.out and
    # range.out_ do not collapse to the same name.
    variant_mut = variant.endswith("_") if variant else False
    if variant:
        type_base += "".join(w.capitalize() for w in variant.split("_") if w)
    if variant_mut:
        type_base += "Mut"
    fn_type = priv_pascal + type_base + "Fn"

    # --- snake_case dispatcher name ---
    disp = priv_snake + base_clean  # foreach already normalized (leading _ stripped)
    if is_inplace:
        disp += "_inplace"
    if variant:
        disp += "_" + variant.lower().rstrip("_")
    if variant_mut:
        disp += "_mut"
    dispatcher_name = disp + "_dispatcher"

    return fn_type, dispatcher_name


def kernel_name(fn_type: str) -> str:
    """AddTensorFn -> AddTensorKernelCuda"""
    return fn_type[:-2] + "KernelCuda"


def python_kernel_name(fn_type: str) -> str:
    """AddTensorFn -> AddTensorKernelPython"""
    return fn_type[:-2] + "KernelPython"


# ============================================================================
# FlagGems Python-path kernels (auto-discovered)
# ============================================================================
# Ops routed to the FlagGems Triton implementation via the embedded-Python
# caller (csrc/aten/backends/flagos/python_op_caller.{h,cc}). Rather than a
# hand-maintained map, discover_flaggems_ops() walks flag_gems._FULL_CONFIG and
# keeps only ops that can be wired SAFELY (see the arity/type gates below).
#
# Key facts (verified):
#   - flag_gems op names ARE aten schema names ("add.Tensor", "addmm.out",
#     "_softmax"), so no name mapping is needed.
#   - flag_gems functions take args by POSITION in aten faithful order (the
#     parameter *names* differ -- A/inp/self -- but positions align).
#   - Every generated kernel body packs the aten args (in faithful order) into a
#     std::vector<c10::IValue> and calls the generic caller; the func is located
#     by its "<module>.<name>" qualname.

# Types the generic IValue caller (python_op_caller.cc IValueToPython) can carry.
# ScalarType is deliberately EXCLUDED: an IValue stores it as a plain int, so it
# is indistinguishable from an ordinary int at runtime -> ops passing a dtype to
# the FlagGems function are dropped from the safe set (would silently mis-call).
_FLAGGEMS_GENERIC_OK = {
    "Tensor",
    "Tensor?",
    "Scalar",
    "Scalar?",
    "int",
    "int?",
    "float",
    "float?",
    "bool",
    "bool?",
    "SymInt",
    "SymInt?",
    "str",
    "str?",
}


def _flaggems_type_ok(t: str) -> bool:
    import re

    t = t.strip()
    if t in _FLAGGEMS_GENERIC_OK:
        return True
    if re.fullmatch(r"(int|SymInt|bool|float)\[\d*\]\??", t):
        return True
    if re.fullmatch(r"(int|SymInt|bool|float)\[\]\??", t):
        return True
    if t in ("Tensor[]", "Tensor?[]"):
        return True
    return False


def _flaggems_gems_npos(fn):
    """Number of positional params of a flag_gems function, or None if it has
    *args/**kwargs (uninspectable arity -> unsafe)."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def _flaggems_kwonly_names(fn):
    """Set of keyword-only parameter names of a flag_gems function (params after
    the `*` in its signature), or empty set if uninspectable. These are the args
    gems accepts only by name -- the codegen forwards the matching aten trailing
    args as kwargs (dtype/alpha/correction/...)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (ValueError, TypeError):
        return set()
    return {p.name for p in params if p.kind == p.KEYWORD_ONLY}


def _flaggems_required_kwonly(fn):
    """Set of keyword-only params of a gems function that have NO default, i.e.
    the ones a call MUST supply by name or Python raises TypeError.

    Returns None when the signature can't be inspected, so callers can tell
    "no required kwonly" (empty set) apart from "don't know" and fail closed.

    Every arity gate here counts POSITIONAL params only, so a signature like
    `sum_out(inp, *, dtype=None, out)` passes an npos check while the call the
    codegen emits omits `out` entirely -- gems then raises "missing 1 required
    keyword-only argument: 'out'" at run time. FlagGems master has 32 such
    `*_out` functions, so this is the rule, not an outlier: the out-variant path
    below matches them BY NAME against the aten out args and forwards them as
    kwargs, and every other path rejects the op rather than emitting a call that
    cannot succeed."""
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (ValueError, TypeError):
        return None
    return {
        p.name
        for p in params
        if p.kind == p.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    }


# aten types the keyword-arg forwarding path can express. ScalarType IS allowed
# here (unlike the positional _FLAGGEMS_GENERIC_OK set): the codegen tags a dtype
# kwarg so the caller converts it to a torch.dtype by name, sidestepping the
# "IValue stores ScalarType as int" ambiguity that blocks the positional path.
_FLAGGEMS_KWARG_OK = {
    "ScalarType",
    "ScalarType?",
    "Scalar",
    "Scalar?",
    "int",
    "int?",
    "float",
    "float?",
    "bool",
    "bool?",
    "str",
    "str?",
    "SymInt",
    "SymInt?",
    # Tensor: only ever produced by the out_variant_gemskwout path below, which
    # forwards an aten `out` arg into a gems required keyword-only tensor param.
    # BuildPyKwargs -> IValueToPython already handles isTensor(), so no C++
    # change is needed. Not reachable from _flaggems_split_kwargs: that path
    # only ever moves TRAILING aten args, and out args are never trailing.
    "Tensor",
}


def _flaggems_extra_trailing_ok(fn, ncall, out_names=()):
    """True if a gems function with more positional params than the `ncall` args
    we intend to pass can be safely called with exactly `ncall` positional args.

    Safe iff every gems param at index >= ncall has a default (so omitting it is
    fine) AND no such trailing param is named `out` sitting *before* a param we
    would still pass -- i.e. the extra params are strictly trailing, never
    interleaved. This rejects the reordering trap (e.g. gems gather inserts
    `out=None` at position 3, so aten's 4th arg `sparse_grad` would land in the
    `out` tensor slot). Returns False if uninspectable.

    `out_names` are the aten out arg names for an out-variant. A trailing param
    matching one of them is never droppable even though it has a default: gems
    writes `out=None` defaults it cannot honor (`logit_out(input, eps=None,
    out=None)` raises "logit_out requires an 'out' tensor" the moment it is
    omitted, and softshrink_out does the same). Refusing here lets the caller
    fall through to the gemsout branch, which passes the tensor positionally.
    """
    import inspect

    try:
        params = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        return False
    pos = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(pos) < ncall:
        return False
    # Every gems param beyond the ones we pass must have a default, and must not
    # be an out param we are on the hook for supplying.
    for p in pos[ncall:]:
        if p.default is inspect.Parameter.empty or p.name in out_names:
            return False
    # The first `ncall` gems params (which receive our aten args positionally)
    # must not include a param named `out`: that would mean gems expects `out`
    # among the leading positions and our aten arg would be misrouted into it.
    for p in pos[:ncall]:
        if p.name == "out":
            return False
    return True


# Categories the FlagGems Python path knows how to generate kernels for.
_FLAGGEMS_PY_CATEGORIES = {
    "functional_pure",
    "inplace",
    "tuple_return",
    "out_variant",
    "factory",
}

# gems ops whose wrapper is `(*args, **kwargs)` so inspect.signature can't
# recover the arity -> _flaggems_gems_npos() returns None and the main loop's
# `if npos is None: continue` gate rejects them (guarding the silent
# dropped-trailing-arg trap). For these hand-verified ops the aten schema IS the
# authoritative arity, so we supply the gems positional count explicitly. Each
# value is the number of leading aten args gems takes positionally; the op then
# falls through to its normal category branch (inplace/functional_pure/
# out_variant). ONLY simple elementwise ops confirmed to run correctly AND take
# every aten positional (no dropped kwarg) belong here -- e.g. logit_ takes both
# self and eps positionally (eps=None passes through), so npos=2.
_FLAGGEMS_ARITY_OVERRIDE = {
    "asinh_": 1,
    "sinh_": 1,
    "log1p_": 1,
    "digamma_": 1,
    "sgn_": 1,
    "hardswish_": 1,
    "logit_": 2,  # self + eps (both positional; eps=None ok)
    "zero": 1,
    "zero.out": 1,  # self only; out injected by out_variant branch
}

# TensorOptions field names carried by every factory schema after the shape/
# scalar positionals. The factory caller injects these itself (device=flagos,
# layout=strided, dtype forwarded, pin_memory=None), so they're stripped from
# the positional list the codegen passes. `memory_format` rides along on the
# *_like factories (like_factory caller injects it as None too).
_FLAGGEMS_TENSOROPT_FIELDS = {"dtype", "layout", "device", "pin_memory"}
_FLAGGEMS_LIKE_OPT_FIELDS = _FLAGGEMS_TENSOROPT_FIELDS | {"memory_format"}

# Random in-place ops we route to gems by injecting a CUDA generator (the
# RandomInplace caller). Whitelisted rather than type-detected because normal_
# is also inplace+Generator? but gems hardcodes generator=None internally (never
# forwards the one we pass -> IndexError on the empty PrivateUse1 default
# generator). Only ops verified to actually thread the generator through belong
# here. See the plan / [[flaggems-gap-analysis]].
_FLAGGEMS_RANDOM_INPLACE = {"uniform_", "exponential_", "bernoulli_.float"}

# RNG functional ops whose aten schema carries a trailing `Generator?` the
# positional caller can't express. We DROP that arg and let gems draw from the
# _patch_flaggems_philox() fallback generator (gems' gen=None path hits the
# patched philox_backend_seed_offset). Verified to run + sample correctly.
# randperm needs no entry here: its aten schema has no generator arg in this
# torch version, so the plain factory branch already covers it. normal_* stays
# excluded (hardcoded generator=None upstream, philox patch can't help). Distinct
# from _FLAGGEMS_RANDOM_INPLACE, which injects a generator explicitly by name.
_FLAGGEMS_RNG_DROPGEN = {"multinomial"}

# Ops manually held out of the FlagGems Python path (populated during the
# compile/import/numerical convergence loop with the reason as a comment).
FLAGGEMS_PYTHON_SKIP = {
    # mm.out used to live here: gems `mm_out(a, b, *, out)` forces `out` and the
    # positional caller could not supply it. The out_variant_gemskwout category
    # forwards required keyword-only out params by name, so it is routed now --
    # along with the other 30-odd gems `*_out` functions that share the shape.
    # Unconditional `assert X.device.type == device` where device == "cuda":
    # flagos tensors are genuinely PrivateUse1 ("privateuseone"), so the assert
    # can never pass without abandoning the boxing scheme. Op-specific to these.
    "maximum",
    "minimum",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "_upsample_bicubic2d_aa",
    # Same device assert ("Input tensor must be on CUDA device"): gems
    # _safe_softmax rejects the PrivateUse1 tensor before running.
    "_safe_softmax",
    # gems embedding_dense_backward raises ValueError("Inputs must be cuda
    # tensors...") when grad_output.device.type != flag_gems.device ("cuda").
    # flagos is PrivateUse1 ("flagos"), so it never matches -- surfaced by qwen3
    # TRAINING (the backward pass; forward `embedding` has no such assert and
    # stays routed). Falls back to the cuda embedding_dense_backward kernel.
    "embedding_dense_backward",
    # gems i0_/zero/zero.out (listed in _FLAGGEMS_ARITY_OVERRIDE for arity) hit an
    # `assert tensor.is_cuda, "...must be on CUDA device"` in their kernel launch
    # -- flagos is genuinely PrivateUse1 (is_cuda False), so it can never pass.
    # (zero's _launch_zero_kernel asserts on both the functional and out paths.)
    # The other 7 arity-override unary inplace ops (asinh_/sinh_/log1p_/digamma_/
    # sgn_/hardswish_/logit_) have no such assert and run fine.
    "i0_",
    "zero",
    "zero.out",
    # rand/randn/rand_like/randn_like/randperm/multinomial are now ROUTED (not
    # skipped). gems' internal philox_backend_seed_offset(increment) reads
    # torch_device_fn.default_generators[device] (torch.cuda for the nvidia/metax
    # branches) and unpacks its state as 2x int64 (seed, offset). The compat shim
    # (torch_fl/accelerator/cuda/_cuda_compat.py, .../metax/_metax_compat.py)
    # installs per-device CUDA generators as torch.cuda.default_generators and
    # routes torch.manual_seed there, so those ops find a real, seedable
    # generator with the philox layout gems expects -- no per-op generator
    # injection and no philox monkeypatch. (randperm's dominant randomness comes
    # from an internal torch.randint that routes to the native `= cuda` path, so
    # it inherits the native default-CUDA-generator's non-reproducibility.)
    # normal family stays skipped: gems' normal_ calls normal_distribution(...,
    # generator=None) with a hardcoded None, dropping any generator we pass, so it
    # hits the empty default_generators (IndexError). Upstream gems bug -- can't
    # fix from here.
    "normal_",
    "normal.Tensor_Tensor",
    "normal.Tensor_float",
    "normal.float_Tensor",
}


def discover_flaggems_ops(codegen_ops, funcs):
    """Discover ops that can be safely routed to the FlagGems Python path.

    Returns {op_name: (gems_func_qualname, category, kwargs)} where kwargs is a
    list of (aten_type, name) for trailing aten args that gems declares
    keyword-only and the kernel forwards by name (dtype/alpha/correction/...);
    empty for the common all-positional case.

    Safety gates (see plan; validated in scratch analysis):
      - op must have a generated dispatcher (op in codegen_ops) and a schema.
      - flag_gems function arity must be inspectable (no *args/**kwargs).
      - category in {functional_pure, inplace, tuple_return, out_variant}.
      - ARITY gate (guards the silent "dropped trailing scalar" trap):
          functional_pure/inplace/tuple_return: gems npos == #aten args.
          out_variant: gems npos == #aten NON-out args (gems doesn't take `out`).
      - REQUIRED-KEYWORD gate: the arity gates above count positional params
        only, so a gems function with a required keyword-only param would pass
        them and then raise TypeError on every call. An out-variant whose
        required kwonly names match the aten out args is routed via
        out_variant_gemskwout (they are forwarded by name); anything else is
        rejected.
      - every arg passed to the gems function has a generic-caller-covered type.
    """

    def _normalize_gems_qualname(fn):
        """Normalize vendor-specific module names to canonical flag_gems.ops.*

        FlagGems exposes ops via vendor aliases (_hygon.ops.*, _nvidia.ops.*, etc)
        that resolve to the same underlying flag_gems.ops.* implementation. Using
        fn.__module__ directly captures whichever alias was active at codegen time,
        breaking imports on other vendors. This strips the vendor prefix so the
        generated C++ always uses the canonical flag_gems.ops.* path.
        """
        module = fn.__module__
        # Vendor aliases: _hygon, _nvidia, _metax, etc.
        if module.startswith("_") and ".ops." in module:
            # Strip vendor prefix: "_hygon.ops.mm" -> "flag_gems.ops.mm"
            return f"flag_gems.ops.{module.split('.ops.', 1)[1]}.{fn.__name__}"
        return f"{module}.{fn.__name__}"

    try:
        # torch_fl activates the torch.cuda shim (CPU-torch reports no CUDA
        # otherwise), which flag_gems needs at import (get_device_name()). Must
        # run codegen through scripts/with_cuda_libtorch.sh so torch_fl imports.
        import torch_fl  # noqa: F401
        import flag_gems
    except Exception as e:
        print(
            f"  [flaggems] import failed ({e}); no python ops discovered",
            file=sys.stderr,
        )
        return {}

    result = {}
    for item in flag_gems._FULL_CONFIG:
        if len(item) < 2:
            continue
        op, fn = item[0], item[1]
        if op in FLAGGEMS_PYTHON_SKIP:
            continue
        if op not in codegen_ops or op not in funcs:
            continue
        # Respect version-condition (item[2]) if present.
        if len(item) > 2 and callable(item[2]) and not item[2]():
            continue
        npos = _flaggems_gems_npos(fn)
        if npos is None:
            # gems wrapper is (*args, **kwargs): arity uninspectable. Accept only
            # hand-verified ops via the explicit override (aten schema is the
            # authoritative arity); everything else stays rejected.
            if op in _FLAGGEMS_ARITY_OVERRIDE:
                npos = _FLAGGEMS_ARITY_OVERRIDE[op]
            else:
                continue
        func = funcs[op]
        cat = detect_category(func)
        if cat not in _FLAGGEMS_PY_CATEGORIES:
            continue
        s = func.func
        aten_args = list(s.arguments.flat_all)
        out_args = list(s.arguments.out) if hasattr(s.arguments, "out") else []
        resolved_cat = cat
        # kwargs: aten trailing args gems declares keyword-only (dtype/alpha/...).
        # Filled by the arity_short branch below; each entry is (name, aten_type).
        kwargs = []
        # *_like factory: functional op whose schema is a source tensor (+ maybe
        # a value positional) followed by TensorOptions + memory_format, all
        # keyword-only in gems. The like_factory caller injects device=flagos/
        # layout/memory_format/pin_memory and forwards dtype; only the non-option
        # positionals are passed. gems reads shape/device from the source tensor.
        if cat == "functional_pure" and op.endswith("_like"):
            all_pairs = [(str(a.type), a.name) for a in aten_args]
            positional = [
                (t, n) for t, n in all_pairs if n not in _FLAGGEMS_LIKE_OPT_FIELDS
            ]
            has_opts = any(n in _FLAGGEMS_LIKE_OPT_FIELDS for _t, n in all_pairs)
            byname = _flaggems_gems_byname_params(fn)
            # gems must accept dtype/layout/device by name and take exactly the
            # non-option positionals (the source tensor, plus full_like's
            # fill_value). npos may exceed via defaulted trailing kwonlys.
            if (
                not has_opts
                or not {"dtype", "layout", "device"} <= byname
                or npos < len(positional)
                or (
                    npos > len(positional)
                    and not _flaggems_extra_trailing_ok(fn, len(positional))
                )
            ):
                continue
            if not all(_flaggems_type_ok(t) for t, _ in positional):
                continue
            qualname = _normalize_gems_qualname(fn)
            result[op] = (qualname, "like_factory", positional)
            continue
        # Random in-place: whitelisted inplace op with a trailing Generator? arg.
        # The RandomInplace caller drops the generator positional and injects a
        # CUDA generator by name; only self + scalar positionals are passed.
        if cat == "inplace" and op in _FLAGGEMS_RANDOM_INPLACE:
            all_pairs = [(str(a.type), a.name) for a in aten_args]
            positional = [(t, n) for t, n in all_pairs if t != "Generator?"]
            has_gen = any(t == "Generator?" for t, _n in all_pairs)
            # gems must take exactly self + the scalar params (generator is
            # keyword-only in gems, supplied by name).
            if (
                not has_gen
                or npos < len(positional)
                or (
                    npos > len(positional)
                    and not _flaggems_extra_trailing_ok(fn, len(positional))
                )
            ):
                continue
            if not all(_flaggems_type_ok(t) for t, _ in positional):
                continue
            qualname = _normalize_gems_qualname(fn)
            result[op] = (qualname, "random_inplace", positional)
            continue
        # RNG functional with a trailing Generator? the caller can't express:
        # drop it and rely on the philox monkeypatch for RNG state. Routed as
        # functional_pure over the remaining positionals (e.g. multinomial's
        # self, num_samples, replacement); gems takes exactly those.
        if op in _FLAGGEMS_RNG_DROPGEN:
            all_pairs = [(str(a.type), a.name) for a in aten_args]
            positional = [(t, n) for t, n in all_pairs if t != "Generator?"]
            has_gen = any(t == "Generator?" for t, _n in all_pairs)
            if (
                not has_gen
                or npos != len(positional)
                or not all(_flaggems_type_ok(t) for t, _ in positional)
            ):
                continue
            qualname = _normalize_gems_qualname(fn)
            result[op] = (qualname, "rng_dropgen", positional)
            continue
        if cat == "factory":
            # Factory: split off the TensorOptions fields (dtype/layout/device/
            # pin_memory); the factory caller injects device=flagos,
            # layout=strided, pin_memory=None and forwards dtype. Only the
            # shape/scalar positionals are passed. Require gems to declare
            # dtype/layout/device by name (kwonly on every gems factory) and its
            # positional count to match the non-option args.
            all_pairs = [(str(a.type), a.name) for a in aten_args]
            positional = [
                (t, n) for t, n in all_pairs if n not in _FLAGGEMS_TENSOROPT_FIELDS
            ]
            has_opts = any(n in _FLAGGEMS_TENSOROPT_FIELDS for _t, n in all_pairs)
            byname = _flaggems_gems_byname_params(fn)
            # gems must accept dtype/layout/device by name, and take exactly the
            # shape/scalar positionals (npos may exceed via a defaulted step, e.g.
            # arange.start's gems has an extra `step` default -> allow >=).
            if (
                not has_opts
                or not {"dtype", "layout", "device"} <= byname
                or npos < len(positional)
                or (
                    npos > len(positional)
                    and not _flaggems_extra_trailing_ok(fn, len(positional))
                )
            ):
                continue
            if not all(_flaggems_type_ok(t) for t, _ in positional):
                continue
            qualname = _normalize_gems_qualname(fn)
            result[op] = (qualname, "factory", positional)
            continue
        # Required keyword-only gems params. Every arity gate below counts
        # POSITIONAL params only, so a signature like
        # `sum_out(inp, *, dtype=None, out)` sails through them while the call
        # the codegen emits omits `out` -> TypeError on every invocation.
        #
        # A required kwonly is fine as long as SOMETHING ends up forwarding it,
        # so this is not decided here: it is enforced once at the bottom, after
        # every branch has filled in `kwargs`. That keeps ops the existing kwarg
        # path already covers (sort.stable's required `stable`) working, and
        # only drops the ones nothing supplies. `None` = uninspectable, which no
        # branch can reason about -> reject outright.
        req_kwonly = _flaggems_required_kwonly(fn)
        if req_kwonly is None:
            continue
        out_names = {a.name for a in out_args}
        # Out-variants whose gems out param(s) are required keyword-only and
        # named exactly like the aten out args: forwarded by the kwout branch.
        kwout = cat == "out_variant" and bool(req_kwonly) and req_kwonly <= out_names
        if cat == "out_variant":
            non_out = [(str(a.type), a.name) for a in aten_args if a not in out_args]
            with_out = non_out + [(str(a.type), a.name) for a in out_args]
            if kwout:
                # gems declares the out tensor(s) as REQUIRED keyword-only params
                # (`sum_out(inp, *, dtype=None, out)`, `softmax_backward_out(...,
                # *, grad_input)`) whose names match the aten out args -- checked
                # above. Forward them by name so gems writes into the caller's
                # tensor directly.
                #
                # Only the out args move here; the remaining aten args still have
                # to line up positionally, or be split off by
                # _flaggems_split_kwargs when gems declares them as DEFAULTED
                # keyword-only params (sum_out's `dtype`).
                out_pairs = [(str(a.type), a.name) for a in out_args]
                if npos == len(non_out) or (
                    npos > len(non_out)
                    and _flaggems_extra_trailing_ok(fn, len(non_out), out_names)
                ):
                    passed = non_out
                elif npos < len(non_out):
                    passed, kwargs = _flaggems_split_kwargs(fn, npos, non_out)
                    if passed is None:
                        continue
                else:
                    continue
                kwargs = kwargs + out_pairs
                resolved_cat = "out_variant_gemskwout"
            elif npos == len(non_out) or (
                npos > len(non_out)
                and _flaggems_extra_trailing_ok(fn, len(non_out), out_names)
            ):
                # gems takes only the non-out args (plus optional trailing
                # defaults), returns a fresh tensor; kernel copy_'s it into out.
                passed = non_out
            elif npos == len(with_out) or (
                npos > len(with_out) and _flaggems_extra_trailing_ok(fn, len(with_out))
            ):
                # gems takes `out` positionally and writes into it; pass the out
                # tensor(s) too (plus optional trailing defaults like
                # memory_format=None). Distinguished from the kwout branch above,
                # whose gems out is a required keyword-only param.
                passed = with_out
                resolved_cat = "out_variant_gemsout"
            elif npos < len(non_out):
                passed, kwargs = _flaggems_split_kwargs(fn, npos, non_out)
                if passed is None:
                    continue
            else:
                continue
        else:
            all_args = [(str(a.type), a.name) for a in aten_args]
            if npos == len(all_args) or (
                npos > len(all_args) and _flaggems_extra_trailing_ok(fn, len(all_args))
            ):
                passed = all_args
            elif npos < len(all_args):
                # gems takes fewer positional args: the trailing aten args must be
                # gems keyword-only params, forwarded by name (see kwarg path).
                passed, kwargs = _flaggems_split_kwargs(fn, npos, all_args)
                if passed is None:
                    continue
            else:
                continue
        # Promote ScalarType args out of the positional list into kwargs: the
        # positional caller can't carry a ScalarType (IValue stores it as a plain
        # int), but the by-name kwarg path tags it is_dtype and converts to a
        # torch.dtype. Safe only if gems accepts the arg BY NAME (a param with the
        # same name that isn't positional-only). This recovers ops like
        # _softmax_backward_data / linalg_vector_norm where gems takes dtype as a
        # positional-or-keyword param. See [[flaggems-gap-analysis]] type_gate_dtype.
        promotable = _flaggems_gems_byname_params(fn)
        promote_idx = [
            i
            for i, (t, name) in enumerate(passed)
            if t.startswith("ScalarType") and name in promotable
        ]
        # Only promote if the ScalarType args form a strict SUFFIX of the
        # positional list. Promoting a middle arg would shift the following
        # positionals into gems's dtype slot (wrong). All current ops have dtype
        # last, so this holds; the guard rejects any future middle-dtype op.
        if promote_idx and promote_idx == list(range(promote_idx[0], len(passed))):
            promoted = passed[promote_idx[0] :]
            passed = passed[: promote_idx[0]]
            kwargs = promoted + kwargs
        elif promote_idx:
            continue  # non-suffix ScalarType -> can't safely reorder
        if not all(_flaggems_type_ok(t) for t, _ in passed):
            continue
        if kwargs and not all(t in _FLAGGEMS_KWARG_OK for t, _ in kwargs):
            continue
        # Final required-keyword gate. Now that every branch has settled
        # `kwargs`, demand that each required keyword-only gems param is
        # actually forwarded -- by the kwout branch, by _flaggems_split_kwargs,
        # or by the ScalarType promotion above. Anything left over is a param
        # nothing supplies, so the generated call would raise TypeError the
        # first time the op ran. Drops e.g. gems nonzero_static_out (requires
        # `size` on top of `out`) while keeping sort.stable (required `stable`,
        # already forwarded by name).
        if not req_kwonly <= {name for _t, name in kwargs}:
            continue
        qualname = _normalize_gems_qualname(fn)
        result[op] = (qualname, resolved_cat, kwargs)
    return result


def _flaggems_gems_byname_params(fn):
    """Names of gems params that can be passed BY NAME (keyword-only OR
    positional-or-keyword, i.e. anything except positional-only). Used to promote
    a ScalarType positional aten arg into a by-name kwarg the caller can convert
    to a torch.dtype."""
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (ValueError, TypeError):
        return set()
    return {
        p.name for p in params if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def _flaggems_split_kwargs(fn, npos, all_passed):
    """For an arity_short op (gems takes `npos` positional args, fewer than the
    `all_passed` aten args), split into (positional, kwargs). The trailing aten
    args beyond `npos` must ALL be gems keyword-only params matched BY NAME, else
    return (None, None) to reject the op (a name mismatch means we can't safely
    forward, and dropping the arg would silently change the result).

    Returns (positional_list, kwarg_list) where each list holds (aten_type, name)
    for positional and (aten_type, name) for kwargs.
    """
    positional = all_passed[:npos]
    trailing = all_passed[npos:]
    kwonly = _flaggems_kwonly_names(fn)
    for _t, name in trailing:
        if name not in kwonly:
            return None, None
    return positional, list(trailing)


def _flaggems_kwarg_cpp(kwargs):
    """Render the C++ `std::vector<PyKwarg>` initializer for the keyword args to
    forward to gems. Each PyKwarg is {name, value, is_dtype, is_none}; a dtype
    (ScalarType) arg is tagged is_dtype so the caller converts it to torch.dtype.
    Optional args (T?) are passed straight through -- the aten C++ arg is a
    std::optional, so a nullopt naturally maps to Python None inside the caller
    only if we mark it; but since we pass the IValue, an absent optional would
    become an int/None ambiguity for dtype. We therefore build the PyKwarg with a
    runtime `.has_value()` check for ScalarType? args."""
    parts = []
    for t, name in kwargs:
        is_dtype = t.startswith("ScalarType")
        if is_dtype:
            if t.endswith("?"):
                # std::optional<ScalarType>: None when absent, else tagged dtype.
                parts.append(
                    f'PyKwarg{{"{name}", {name}.has_value() ? '
                    f"c10::IValue(static_cast<int64_t>(*{name})) : c10::IValue(), "
                    f"/*is_dtype=*/true, /*is_none=*/!{name}.has_value()}}"
                )
            else:
                parts.append(
                    f'PyKwarg{{"{name}", '
                    f"c10::IValue(static_cast<int64_t>({name})), "
                    f"/*is_dtype=*/true}}"
                )
        else:
            # Scalar/int/bool/str/SymInt (and their optionals): IValue can carry
            # these directly. c10::IValue has implicit ctors for each.
            parts.append(f'PyKwarg{{"{name}", {name}}}')
    return "{" + ", ".join(parts) + "}"


def gen_flaggems_python_kernel(
    op, fn_type, ret_type, args, gems_func, category, func, kwargs=None
):
    """Generate a <Name>KernelPython forwarding to the FlagGems Python op.

    Signature matches the generated `*Fn` typedef exactly. The body packs the
    aten positional args (faithful order) into a std::vector<c10::IValue> and any
    keyword-only gems args (dtype/alpha/correction/...) into a
    std::vector<PyKwarg>, then calls the generic caller by qualname and adapts
    the result to the category's return convention:
      functional_pure -> UnboxToFlagos(result); return result;
      inplace         -> self.copy_(result); return self;  (or void)
      tuple_return    -> GenericTuple(...); unbox each; return make_tuple(...)
      out_variant     -> Generic/GenericTuple over NON-out args; out_i.copy_(res_i)
    """
    kn = python_kernel_name(fn_type)
    s = func.func
    aten_args = list(s.arguments.flat_all)
    out_args = list(s.arguments.out) if hasattr(s.arguments, "out") else []

    if category == "factory":
        # Factory: pass only the shape/scalar positionals; the factory caller
        # injects device=flagos/layout=strided/pin_memory=None and forwards
        # dtype + device. `kwargs` here carries the positional (aten_type, name) list.
        positional = kwargs or []
        pos_names = [n for _t, n in positional]
        dtype_name = next((a.name for a in aten_args if a.name == "dtype"), None)
        dtype_expr = dtype_name if dtype_name else "::std::nullopt"
        # Forward the aten `device` arg so the output lands on the requested
        # index. Without it the caller fell back to a hardcoded index 0, which
        # allocated on device 0 while the Triton kernel launched on the current
        # device -- a cross-device write that faults the GPU on multi-card runs.
        device_name = next((a.name for a in aten_args if a.name == "device"), None)
        device_expr = device_name if device_name else "::std::nullopt"
        pos_init = "{" + ", ".join(pos_names) + "}"
        infer = ""
        # arange: gems defaults dtype=None to int64 unconditionally, but aten
        # infers float when ANY of start/end/step is floating. Left alone, gems
        # returns silently-wrong integer values for arange(0.,3.,.5). Replicate
        # aten's rule here so gems always gets an explicit dtype. Only arange has
        # this value-dependent inference; the other factories default to float32,
        # which gems already produces correctly.
        if op.startswith("arange"):
            scalar_names = [n for t, n in positional if "Scalar" in t]
            any_float = " || ".join(f"{n}.isFloatingPoint()" for n in scalar_names)
            infer = (
                f"  ::std::optional<at::ScalarType> _dt = {dtype_expr};\n"
                f"  if (!_dt.has_value()) _dt = ({any_float})\n"
                f"      ? at::typeMetaToScalarType(at::get_default_dtype()) "
                f": at::kLong;\n"
            )
            dtype_expr = "_dt"
        body = (
            f"{infer}"
            f'  auto result = CallPythonOp_Factory("{gems_func}", '
            f"{pos_init}, {dtype_expr}, {device_expr});\n"
            f"  UnboxToFlagos(result);\n"
            f"  return result;"
        )
        return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"

    if category == "like_factory":
        # *_like: pass the source tensor (+ full_like's fill_value); the caller
        # injects device=flagos/layout/memory_format/pin_memory and forwards
        # dtype (nullopt -> None means "same as self") + device. `kwargs` here
        # carries the non-option positional (aten_type, name) list.
        positional = kwargs or []
        pos_names = [n for _t, n in positional]
        dtype_name = next((a.name for a in aten_args if a.name == "dtype"), None)
        dtype_expr = dtype_name if dtype_name else "::std::nullopt"
        # See the factory branch: forwarded so the result honors the requested
        # device index. nullopt here means "same device as self".
        device_name = next((a.name for a in aten_args if a.name == "device"), None)
        device_expr = device_name if device_name else "::std::nullopt"
        pos_init = "{" + ", ".join(pos_names) + "}"
        body = (
            f'  auto result = CallPythonOp_LikeFactory("{gems_func}", '
            f"{pos_init}, {dtype_expr}, {device_expr});\n"
            f"  UnboxToFlagos(result);\n"
            f"  return result;"
        )
        return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"

    if category == "random_inplace":
        # Random in-place: pass self + scalar params (generator dropped); the
        # caller injects a CUDA generator by name. gems writes into self and
        # returns it; copy_ the result back to self for the aten alias contract.
        positional = kwargs or []
        pos_names = [n for _t, n in positional]
        self_name = pos_names[0]
        pos_init = "{" + ", ".join(pos_names) + "}"
        ret_line = "" if ret_type == "void" else f"\n  return {self_name};"
        body = (
            f'  auto result = CallPythonOp_RandomInplace("{gems_func}", '
            f"{pos_init});\n"
            f"  {self_name}.copy_(result);{ret_line}"
        )
        return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"

    if category == "rng_dropgen":
        # RNG functional with the aten Generator? dropped: pass the remaining
        # positionals to gems; RNG state comes from the philox monkeypatch
        # (torch_fl._patch_flaggems_philox). Returns a fresh (flagos) tensor.
        positional = kwargs or []
        pos_names = [n for _t, n in positional]
        pos_init = "{" + ", ".join(pos_names) + "}"
        body = (
            f'  auto result = CallPythonOp_Generic("{gems_func}", {pos_init});\n'
            f"  UnboxToFlagos(result);\n"
            f"  return result;"
        )
        return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"

    kwargs = kwargs or []
    kw_names = {name for _t, name in kwargs}

    # arg names as they appear in the generated C++ signature. Kwarg-forwarded
    # args are removed from the positional list (they go into the PyKwarg vector).
    if category in ("out_variant", "out_variant_gemskwout"):
        # gemskwout carries its out args in `kwargs`, so the kw_names filter
        # already drops them; the out_args test is what covers plain out_variant.
        passed_names = [
            a.name for a in aten_args if a not in out_args and a.name not in kw_names
        ]
    elif category == "out_variant_gemsout":
        # gems takes the out tensor(s) positionally after the non-out args.
        passed_names = [
            a.name for a in aten_args if a not in out_args and a.name not in kw_names
        ] + [a.name for a in out_args]
    else:
        passed_names = [a.name for a in aten_args if a.name not in kw_names]
    ivalues = "{" + ", ".join(passed_names) + "}"

    # kwarg call selects the *Kw caller variant; positional-only uses the plain one.
    if kwargs:
        kw_init = _flaggems_kwarg_cpp(kwargs)
        gen_call = f'CallPythonOp_GenericKw("{gems_func}", {ivalues}, {kw_init})'

        def tuple_call(n):
            return (
                f'CallPythonOp_GenericKwTuple("{gems_func}", {ivalues}, {kw_init}, {n})'
            )
    else:
        gen_call = f'CallPythonOp_Generic("{gems_func}", {ivalues})'

        def tuple_call(n):
            return f'CallPythonOp_GenericTuple("{gems_func}", {ivalues}, {n})'

    if category == "functional_pure":
        body = (
            f"  auto result = {gen_call};\n  UnboxToFlagos(result);\n  return result;"
        )
    elif category == "inplace":
        self_name = passed_names[0]
        ret_line = "" if ret_type == "void" else f"\n  return {self_name};"
        body = f"  auto result = {gen_call};\n  {self_name}.copy_(result);{ret_line}"
    elif category == "tuple_return":
        n = ret_type.count(",") + 1  # ::std::tuple<a,b> -> 2
        unbox = "\n".join(f"  UnboxToFlagos(result[{i}]);" for i in range(n))
        make = ", ".join(f"result[{i}]" for i in range(n))
        body = f"  auto result = {tuple_call(n)};\n{unbox}\n  return {{{make}}};"
    elif category == "out_variant":
        out_names = [a.name for a in out_args]
        if ret_type.startswith("::std::tuple"):
            n = len(out_names)
            copies = "\n".join(
                f"  {out_names[i]}.copy_(result[{i}]);" for i in range(n)
            )
            make = ", ".join(out_names)
            body = f"  auto result = {tuple_call(n)};\n{copies}\n  return {{{make}}};"
        else:
            out_name = out_names[0]
            body = (
                f"  auto result = {gen_call};\n"
                f"  {out_name}.copy_(result);\n"
                f"  return {out_name};"
            )
    elif category == "out_variant_gemsout":
        # gems receives the out tensor(s) positionally and writes into them in
        # place, then returns them; just discard the returned handle and return
        # the aten out arg(s). (Only single-out ops reach here today.)
        out_name = [a.name for a in out_args][0]
        body = f"  {gen_call};\n  return {out_name};"
    elif category == "out_variant_gemskwout":
        # gems declares the out tensor(s) as REQUIRED keyword-only params and
        # writes into them in place (see the kwout branch in
        # discover_flaggems_ops). They are already in `kwargs`, so gen_call
        # forwards them by name; discard the returned handle -- it aliases the
        # very tensor(s) we were handed -- and return the aten out arg(s). No
        # copy_ back: that would be a self-copy.
        out_names = [a.name for a in out_args]
        if len(out_names) == 1:
            call, make = f"  {gen_call};", out_names[0]
        else:
            # median.dim_values / nanmedian.dim_values: gems requires `values`
            # and `indices` by name and returns a namedtuple of both. Use the
            # tuple caller so the sequence return is unpacked rather than run
            # through PythonToTensor, which would reject a non-Tensor.
            call = f"  {tuple_call(len(out_names))};"
            make = "{" + ", ".join(out_names) + "}"
        body = f"{call}\n  return {make};"
    else:
        raise ValueError(f"unsupported flaggems-python category {category} for {op}")

    return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"


# ============================================================================
# Category detection (torchgen metadata)
# ============================================================================


def detect_category(func) -> str:
    s = func.func
    arg_types = [str(a.type) for a in s.arguments.flat_all]

    has_tensorlist = any(t == "Tensor[]" for t in arg_types)
    has_optlist = any("Tensor?[]" in t for t in arg_types)
    has_tensor_in = any(t.startswith("Tensor") and "[]" not in t for t in arg_types)

    # Special case: new_* ops are factory-like (use at::empty + fill)
    op_name = str(s.name.name)
    if op_name.startswith("new_"):
        return "factory"

    if has_optlist:
        return "special_optlist"
    if not has_tensor_in and not has_tensorlist:
        return "factory"
    if has_tensorlist:
        # _foreach_*.out: TensorList out arg, call at::<base>_outf, void return.
        if s.is_out_fn():
            return "foreach_out"
        return "foreach_tensorlist"
    if s.is_out_fn():
        return "out_variant"
    if str(s.kind()).split(".")[-1] == "inplace":
        return "inplace"
    if len(s.returns) > 1:
        return "tuple_return"
    # split.Tensor / unbind.int: single Tensor[] return, no TensorList input.
    if len(s.returns) == 1 and str(s.returns[0].type) == "Tensor[]":
        return "vector_return"
    return "functional_pure"


# ============================================================================
# Authoritative signatures via torchgen (faithful = exploded TensorOptions)
# ============================================================================


def unified_sig(func):
    """
    CppSignature (faithful) for ALL uses: typedef, kernel, and wrapper.
    This ensures the typedef, REGISTER_IMPL, and m.impl all agree on types.

    Faithful = TensorOptions exploded into dtype/layout/device/pin_memory,
    which is exactly what PrivateUse1 boxing dispatch requires.

    Returns (ptr_type_str, ret_type_str, [(cpp_type_str, name), ...]).
    Must be called inside a `with local.parametrize(...)` context.
    """
    from torchgen.api.types import CppSignatureGroup

    group = CppSignatureGroup.from_native_function(
        func, method=False, fallback_binding=False
    )
    sig = (
        group.faithful_signature
        if group.faithful_signature is not None
        else group.signature
    )
    ptr = sig.ptr_type()
    ret_type = ptr.split("(*)", 1)[0].strip()
    args = [(a.type, a.name) for a in sig.arguments()]
    return ptr, ret_type, args


def fn_ptr_signature(ret_type: str, args: List[Tuple[str, str]]) -> str:
    return f"{ret_type} (*)({', '.join(t for t, _ in args)})"


def args_decl(args: List[Tuple[str, str]]) -> str:
    return ", ".join(f"{t} {n}" for t, n in args)


def call_args(args: List[Tuple[str, str]], rename: Dict[str, str] = None) -> str:
    if rename:
        return ", ".join(rename.get(n, n) for _, n in args)
    return ", ".join(n for _, n in args)


def at_api_base(op_name: str) -> str:
    """schema op -> at:: function base name. 'mm.out'->'mm', '_foreach_add_.Scalar'->'_foreach_add_'."""
    return op_name.split(".")[0]


# ============================================================================
# Per-category kernel templates
# ============================================================================


def tensor_arg_names(args: List[Tuple[str, str]]) -> List[str]:
    """Plain (non-optional, non-list) at::Tensor args - safe for DeviceBoxingGuard."""
    out = []
    for t, n in args:
        if "Tensor" not in t:
            continue
        if (
            "optional" in t
            or "List" in t
            or "TensorList" in t
            or "ArrayRef<at::Tensor" in t
        ):
            continue
        out.append(n)
    return out


def optional_tensor_names(args: List[Tuple[str, str]]) -> List[str]:
    return [n for t, n in args if "optional<at::Tensor>" in t]


def _generator_inject_line(args, device_expr):
    """Emit the CUDA-boxing prelude for a schema carrying `Generator?`.

    Translate a caller-supplied flagos (PrivateUse1) generator into a CUDA one
    before the inner ATen redispatch. Left untranslated, the generator's
    PrivateUse1 key routes the already-boxed call back into this same kernel,
    recursing until stack overflow -- the old prelude guarded this by
    TORCH_CHECK-rejecting the generator, but that broke HF-style code that
    seeds torch.Generator(device=<device string>) generically (diffusers
    randn_tensor). The flagos generator carries CPU mt19937 state, so
    translation draws this op's seed via c10::flagos::ReserveSeed
    (mutex-guarded; the caller-visible state advances exactly once per call,
    the same per-op seed-reservation semantics the flaggems/native routes use)
    and reseeds a clone of the shared CUDA generator. If no generator was
    supplied, inject the shared CUDA generator so torch.manual_seed unifies
    native and FlagGems RNG.

    Returns '' for non-RNG ops. `device_expr` is a C++ expression yielding the
    device index in the kernel body scope.
    """
    has_gen = any("Generator" in t for t, _ in args)
    if not has_gen:
        return ""
    # The generator parameter is always named `generator` in the faithful sig.
    return (
        "  generator = TranslateGeneratorForCudaBoxing(generator, "
        f"{device_expr});\n"
        f"  if (!generator.has_value()) generator = "
        f"at::native::flagos::GetFlagosDefaultCudaGenerator({device_expr});\n"
    )


# RNG bases whose *plain* overloads carry NO `Generator?` in the schema, so
# _generator_inject_line above never fires for them and they would silently fall
# back to ATen's own default CUDA generator -- unreachable from
# torch.manual_seed on the CPU-torch wheel. Each base does expose a sibling ATen
# overload with `optional<Generator>` inserted at a fixed position, so the
# corresponding template threads the flagos shared generator in explicitly.
# One set per template, because the insertion point differs:
#   factory      -> right after the size/n/high value args (before dtype)
#   *_like       -> right before dtype
#   out-variant  -> before the first of names / memory_format / out
_GEN_FACTORY_BASES = {"randint", "randperm", "rand", "randn"}
_GEN_LIKE_BASES = {"randint_like", "rand_like", "randn_like"}
_GEN_OUT_BASES = _GEN_FACTORY_BASES | _GEN_LIKE_BASES


# Pure view/alias ops whose generated kernel MUST call at::native:: directly
# instead of at::<op>. These ops are also registered on the PrivateUse1 key, so
# calling the re-dispatchable at::<op>(self) from inside the kernel routes right
# back into THIS kernel. Under eager, DeviceBoxingGuard hides the recursion by
# rewriting self's device metadata to CUDA before the call. But that trick fails
# for FakeTensors: their Python dispatch key sits ABOVE the backend keys, so the
# metadata rewrite can't redirect dispatch -- at::<op>(self) re-dispatches to
# PrivateUse1 -> infinite recursion -> stack overflow (segfault during
# torch.compile / dynamo tracing of any model that hits detach, e.g. nn.Linear).
# at::native::<op> is the metadata-only implementation and never re-dispatches,
# matching how strided_ops.cc registers view/as_strided/transpose for Ascend.
NATIVE_DIRECT_VIEW_OPS = {
    "detach",
}


def gen_functional_pure(op, fn_type, ret_type, args, func=None):
    kn = kernel_name(fn_type)
    api = f"at::{at_api_base(op)}"

    # Pure alias ops (detach): call at::native:: directly, no boxing, no
    # re-dispatch. See NATIVE_DIRECT_VIEW_OPS for why re-dispatch is fatal.
    if at_api_base(op) in NATIVE_DIRECT_VIEW_OPS:
        return f"""{ret_type} {kn}({args_decl(args)}) {{
  return at::native::{at_api_base(op)}({call_args(args)});
}}"""

    # Box plain Tensor inputs AND optional<Tensor> inputs (e.g. conv/linear bias).
    # An optional<Tensor> that lives on flagos must be boxed to CUDA too, or the
    # backend op receives a mix of boxed (input/weight) and unboxed (bias)
    # tensors -> "tensor does not have a device" / segfault. DeviceBoxingGuard
    # only accepts at::Tensor, so materialize each optional into a holder first
    # (same pattern as gen_out_variant / gen_tuple_return).
    plain = tensor_arg_names(args)
    opt_names = optional_tensor_names(args)
    contig_lines, rename = _contiguous_prelude(op, args)
    holder_lines = ""
    guard_names = [rename.get(n, n) for n in plain]
    for on in opt_names:
        holder_lines += (
            f"  at::Tensor {on}_t = {on}.has_value() ? *{on} : at::Tensor();\n"
        )
        guard_names.append(f"{on}_t")
    guard = ", ".join(guard_names)

    inject = (
        _generator_inject_line(args, f"{guard_names[0]}.get_device()")
        if guard_names
        else ""
    )

    # Unified RNG for generator-LESS *_like overloads.
    #
    # randint_like/rand_like/randn_like carry a `self` tensor, so they land on
    # this template rather than gen_factory -- and their plain overloads have no
    # `Generator?` arg, so _generator_inject_line above emits nothing and
    # at::randint_like(...) falls back to ATen's default CUDA generator, which
    # torch.manual_seed cannot reach. Each of these bases exposes a sibling
    # overload with `Generator?` inserted after the trailing value args and
    # before dtype (verified in ATen/ops/randint_like.h), so inject the flagos
    # shared generator at that position.
    base = at_api_base(op)
    if base in _GEN_LIKE_BASES and not any("Generator" in t for t, _ in args):
        dtype_name = next((n for t, n in args if "optional<at::ScalarType>" in t), None)
        if dtype_name is not None and guard_names:
            gen_call_names = []
            for _t, n in args:
                if n == dtype_name:
                    gen_call_names.append("_flagos_gen")
                gen_call_names.append(n)
            inject = (
                "  ::std::optional<at::Generator> _flagos_gen = "
                "at::native::flagos::GetFlagosDefaultCudaGenerator("
                f"{guard_names[0]}.get_device());\n"
            )
            return f"""{ret_type} {kn}({args_decl(args)}) {{
{holder_lines}  DeviceBoxingGuard guard({guard});
{inject}  auto result = {api}({", ".join(gen_call_names)});
  UnboxToFlagos(result);
  return result;
}}"""

    return f"""{ret_type} {kn}({args_decl(args)}) {{
{contig_lines}{holder_lines}  DeviceBoxingGuard guard({guard});
{inject}  auto result = {api}({call_args(args, rename)});
  UnboxToFlagos(result);
  return result;
}}"""


def gen_vector_return(op, fn_type, ret_type, args, func=None):
    """split.Tensor / unbind.int: single Tensor input, std::vector<Tensor> return
    (each element aliases the input storage). Box inputs, call the API, unbox
    every returned view."""
    kn = kernel_name(fn_type)
    guard = ", ".join(tensor_arg_names(args))
    api = f"at::{at_api_base(op)}"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard guard({guard});
  auto result = {api}({call_args(args)});
  UnboxTensorVecToFlagos(result);
  return result;
}}"""


def gen_inplace(op, fn_type, ret_type, args, func=None):
    """add_.Tensor / fill_.Scalar / masked_fill_.Scalar: mutate first tensor, return it (or void).

    Two calling conventions, selected by the op's torchgen `variants`:
      - method  (add_/masked_fill_): `self.add_(other, alpha)` -- these ops have
        NO free function, only a Tensor method.
      - function-only (silu_/gelu_/celu_/leaky_relu_/...activations): call the free
        function `at::silu_(self, ...)` -- these have NO Tensor method, so the
        method syntax fails to compile.
    Ops with both (fill_/zero_/relu_) work either way; we default to method.

    optional<Tensor> inputs must be materialized into a holder to be boxed by
    DeviceBoxingGuard (same pattern as gen_functional_pure / gen_out_variant).
    Missing this makes e.g. clamp_.Tensor pass unboxed flagos min/max into a
    CUDA self -> "tensor does not have a device" / segfault."""
    kn = kernel_name(fn_type)
    # Box plain Tensor inputs AND optional<Tensor> inputs (e.g. clamp_.Tensor's
    # min/max). An optional<Tensor> on flagos must be boxed to CUDA too, or the
    # backend op receives a mix of boxed and unboxed tensors -> "tensor does not
    # have a device" / segfault. DeviceBoxingGuard only accepts at::Tensor, so
    # materialize each optional into a holder first (same pattern as
    # gen_functional_pure / gen_out_variant / gen_tuple_return).
    guard_names = list(tensor_arg_names(args))
    holder_lines = ""
    for on in optional_tensor_names(args):
        holder_lines += (
            f"  at::Tensor {on}_t = {on}.has_value() ? *{on} : at::Tensor();\n"
        )
        guard_names.append(f"{on}_t")
    guard = ", ".join(guard_names)
    base = at_api_base(op)  # e.g. "add_" already has trailing underscore
    self_name = args[0][1]
    other_args = ", ".join(n for _, n in args[1:])

    variants = [str(v).split(".")[-1] for v in func.variants] if func else []
    has_method = "method" in variants

    if has_method:
        # Method syntax: self.add_(other, alpha)
        body_call = (
            f"{self_name}.{base}({other_args});"
            if other_args
            else f"{self_name}.{base}();"
        )
    else:
        # Function-only syntax: at::silu_(self, ...)
        body_call = f"at::{base}({call_args(args)});"

    if ret_type == "void":
        ret_line = ""
    else:
        ret_line = f"\n  return {self_name};"
    inject = _generator_inject_line(args, "self.get_device()")
    return f"""{ret_type} {kn}({args_decl(args)}) {{
{holder_lines}  DeviceBoxingGuard guard({guard});
{inject}  {body_call}{ret_line}
}}"""


def gen_out_variant(op, fn_type, ret_type, args, func=None):
    """Single- and multi-output out-variants (mm.out, sort.values, svd.U,
    native_batch_norm.out): call at::<base>_outf(<all args in faithful order>),
    which places the out args last -- exactly matching our generated arg order,
    so no reordering is needed. Then unbox each mutable Tensor& out.

    optional<Tensor> inputs must be materialized into a holder to be boxed by
    DeviceBoxingGuard (same pattern as gen_tuple_return)."""
    kn = kernel_name(fn_type)
    base = at_api_base(op)
    api = f"at::{base}_outf"

    # Mutable (non-const) Tensor& args are the outputs.
    out_names = [n for t, n in args if "at::Tensor &" in t and "const" not in t]

    # Box plain tensor inputs + optional<Tensor> inputs (via holders).
    plain = tensor_arg_names(args)
    opt_names = optional_tensor_names(args)
    contig_lines, rename = _contiguous_prelude(op, args)
    holder_lines = ""
    guard_names = [rename.get(n, n) for n in plain]
    for on in opt_names:
        holder_lines += (
            f"  at::Tensor {on}_t = {on}.has_value() ? *{on} : at::Tensor();\n"
        )
        guard_names.append(f"{on}_t")
    guard = ", ".join(guard_names)

    unbox_lines = "\n".join(f"  UnboxToFlagos({n});" for n in out_names)

    device_source = out_names[0] if out_names else guard_names[0]
    inject = _generator_inject_line(args, f"{device_source}.get_device()")
    call = call_args(args, rename)

    # Unified RNG for generator-LESS out-variants (rand.out, randint.out,
    # randint_like.out, randperm.out, ...). Same gap as the gen_factory /
    # gen_functional_pure cases: the plain overload carries no `Generator?`, so
    # _generator_inject_line emits nothing and at::<base>_outf() falls back to
    # ATen's default CUDA generator, which torch.manual_seed cannot reach.
    #
    # Every one of these bases exposes a sibling `_outf` overload with
    # `optional<Generator>` inserted after the trailing value args and before the
    # first trailing modifier -- `names` (rand/randn), `memory_format` (*_like),
    # or the `out` tensor itself when neither is present. Verified across
    # ATen/ops/{rand,randn,rand_like,randn_like,randint,randint_like,randperm}.h.
    if base in _GEN_OUT_BASES and not any("Generator" in t for t, _ in args):
        gen_call_names = []
        placed = False
        for t, n in args:
            if not placed and (
                "optional<at::DimnameList>" in t
                or "optional<at::MemoryFormat>" in t
                or n in out_names
            ):
                gen_call_names.append("_flagos_gen")
                placed = True
            gen_call_names.append(n)
        if placed:
            inject = (
                "  ::std::optional<at::Generator> _flagos_gen = "
                "at::native::flagos::GetFlagosDefaultCudaGenerator("
                f"{device_source}.get_device());\n"
            )
            call = ", ".join(gen_call_names)

    if ret_type.startswith("::std::tuple"):
        # tuple<Tensor&,...>: _ret references the out tensors; unbox out names.
        # (local is _ret, not result, because some ops have an out arg literally
        #  named `result`, e.g. _linalg_det.result.)
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{contig_lines}{holder_lines}  DeviceBoxingGuard guard({guard});
{inject}  auto _ret = {api}({call});
{unbox_lines}
  return _ret;
}}"""
    # single mutable Tensor& out.
    out_name = out_names[0]
    return f"""{ret_type} {kn}({args_decl(args)}) {{
{contig_lines}{holder_lines}  DeviceBoxingGuard guard({guard});
{inject}  {api}({call});
{unbox_lines}
  return {out_name};
}}"""


def gen_tuple_return(op, fn_type, ret_type, args, func=None):
    """nll_loss_forward / sort / topk: unbox each tuple element."""
    kn = kernel_name(fn_type)
    # optional<Tensor> weight needs a holder to be boxed by DeviceBoxingGuard
    opt_names = optional_tensor_names(args)
    plain = tensor_arg_names(args)
    api = f"at::{at_api_base(op)}"
    ntuple = ret_type.count(",") + 1  # ::std::tuple<a,b> -> 2

    # Layout normalization (e.g. native_group_norm.input) must run BEFORE the
    # guard so the contiguous copy happens while the tensor still lives on flagos.
    contig_lines, rename = _contiguous_prelude(op, args)

    holder_lines = ""
    guard_names = [rename.get(n, n) for n in plain]
    for on in opt_names:
        holder_lines += (
            f"  at::Tensor {on}_t = {on}.has_value() ? *{on} : at::Tensor();\n"
        )
        guard_names.append(f"{on}_t")
    guard = ", ".join(guard_names)

    unbox_lines = "\n".join(
        f"  UnboxToFlagos(std::get<{i}>(result));" for i in range(ntuple)
    )
    inject = _generator_inject_line(args, f"{guard_names[0]}.get_device()")
    return f"""{ret_type} {kn}({args_decl(args)}) {{
{contig_lines}{holder_lines}  DeviceBoxingGuard guard({guard});
{inject}  auto result = {api}({call_args(args, rename)});
{unbox_lines}
  return result;
}}"""


def _scalar_tensor_box_lines(args) -> str:
    """`guard.box({...})` lines for the plain const Tensor& args of a foreach op.

    TensorListBoxingGuard kernels used to box only the tensor *lists* (and, for
    out-variants, the mutable outs). Any remaining `const at::Tensor &` argument
    stayed on flagos, which is not merely a missed optimization: the at:: call
    dispatches on *all* its tensor arguments, so one unboxed flagos tensor sends
    it back to PrivateUse1 -- into this same kernel -- and it recurses until the
    stack is gone. Mutable outs are skipped here; the caller already boxed them.
    """
    lines = ""
    for t, n in args:
        if "TensorList" in t or "ITensorListRef" in t:
            continue
        if "at::Tensor" not in t:
            continue
        if "optional" in t:
            # e.g. the fused optimizers' grad_scale/found_inf.
            lines += f"  if ({n}.has_value()) guard.box({{*{n}}});\n"
            continue
        if "const" not in t:
            continue  # mutable out: boxed by the caller
        lines += f"  guard.box({{{n}}});\n"
    return lines


def gen_foreach(op, fn_type, ret_type, args, func=None):
    """cat + _foreach_*: materialize ITensorListRef, box, call API, unbox result."""
    kn = kernel_name(fn_type)
    api = f"at::{at_api_base(op)}"

    # Detect TensorList args (ITensorListRef in torch 2.13)
    tensorlist_args = [(t, n) for t, n in args if "TensorList" in t]

    # cat must drop legacy-empty (1-D size-0) inputs before dispatch: maca's
    # forked libtorch_cuda cat kernel mishandles them on its vectorized fast
    # path (see DropLegacyEmptyForCat in device_boxing.h).
    is_cat = at_api_base(op) == "cat"

    # Materialize ITensorListRef → std::vector<Tensor>
    # The vector implicitly converts to TensorList (ArrayRef) for PyTorch API
    materialize_lines = ""
    call_arg_names = []
    for t, n in args:
        if "ITensorListRef" in t or "TensorList" in t:
            mat_name = f"{n}_vec"
            if is_cat:
                materialize_lines += (
                    f"  auto {mat_name} = "
                    f"DropLegacyEmptyForCat(MaterializeToTensorVec({n}));\n"
                )
            else:
                materialize_lines += (
                    f"  auto {mat_name} = MaterializeToTensorVec({n});\n"
                )
            call_arg_names.append(mat_name)
        else:
            call_arg_names.append(n)

    call_args_str = ", ".join(call_arg_names)

    # Box the materialized vectors
    box_lines = ""
    for t, n in tensorlist_args:
        mat_name = f"{n}_vec"
        box_lines += f"  guard.box({mat_name});\n"

    # Plain (non-list) Tensor inputs must be boxed too. Boxing only the lists
    # leaves e.g. _foreach_mul.Tensor's `other` on flagos, and at::_foreach_mul
    # then re-dispatches to PrivateUse1 -- straight back into this kernel, i.e.
    # unbounded self-recursion ending in SIGSEGV.
    box_lines += _scalar_tensor_box_lines(args)

    # _amp_foreach_non_finite_check_and_unscale_ mutates found_inf in place,
    # even though it is not a TensorList output. It must be boxed alongside
    # self_vec or the vendor CUDA kernel receives a mixed PrivateUse1/CUDA
    # argument set and can crash while dispatching.
    for t, n in args:
        if "at::Tensor &" in t and "const" not in t:
            box_lines += f"  guard.box({{{n}}});\n"

    if ret_type == "void":
        body = f"  {api}({call_args_str});"
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}{body}
}}"""
    elif "vector" in ret_type:
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}  auto result = {api}({call_args_str});
  UnboxTensorVecToFlagos(result);
  return result;
}}"""
    else:
        # single Tensor return (cat)
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}  auto result = {api}({call_args_str});
  UnboxToFlagos(result);
  return result;
}}"""


def gen_foreach_out(op, fn_type, ret_type, args, func=None):
    """out-variants whose inputs include a TensorList. Two shapes:

      1. _foreach_*.out / split_copy.Tensor_out: the out is itself a
         `Tensor(a!)[]` list, return is void. Materialize + box every
         TensorList (inputs AND out list), call at::<base>_outf(...).
      2. cat.out / stack.out / block_diag.out / _chunk_cat.out: TensorList
         input(s) + a single mutable `Tensor(a!) out`, returning `Tensor&`.
         Box the input lists and the single out, call _outf, return the out.

    Args are already in faithful order (outs last), matching at::<base>_outf."""
    kn = kernel_name(fn_type)
    api = f"at::{at_api_base(op)}_outf"

    # cat.out must drop legacy-empty (1-D size-0) inputs, same as cat (see
    # DropLegacyEmptyForCat in device_boxing.h / gen_foreach).
    is_cat = at_api_base(op) == "cat"

    # Every mutable single Tensor& arg (non-list, non-const) must be boxed. For
    # void ops this includes an in-out like found_inf; for single-return ops it
    # is the lone `out`. Tuple-returning multi-Tensor& out RNN ops are skip-listed.
    mutable_tensors = [
        n
        for t, n in args
        if "at::Tensor &" in t
        and "const" not in t
        and "TensorList" not in t
        and "ITensorListRef" not in t
    ]

    materialize_lines = ""
    box_lines = ""
    call_arg_names = []
    for t, n in args:
        if "ITensorListRef" in t or "TensorList" in t:
            mat_name = f"{n}_vec"
            if is_cat:
                materialize_lines += (
                    f"  auto {mat_name} = "
                    f"DropLegacyEmptyForCat(MaterializeToTensorVec({n}));\n"
                )
            else:
                materialize_lines += (
                    f"  auto {mat_name} = MaterializeToTensorVec({n});\n"
                )
            box_lines += f"  guard.box({mat_name});\n"
            call_arg_names.append(mat_name)
        else:
            if n in mutable_tensors:
                box_lines += f"  guard.box({{{n}}});\n"
            call_arg_names.append(n)
    # const Tensor& inputs need boxing as much as the lists and the outs do:
    # split_with_sizes_copy.out left `self` on flagos, so at::split_with_sizes_
    # copy_outf re-dispatched to PrivateUse1 and recursed into this kernel until
    # the stack blew (SIGSEGV). This is FSDP2's all-gather copy-out path.
    box_lines += _scalar_tensor_box_lines(args)
    call_args_str = ", ".join(call_arg_names)

    # Return shape follows ret_type: void (no return) or single `Tensor&`
    # (return the lone out). tuple<Tensor&,...> is not produced here (skipped).
    ret_line = ""
    if ret_type != "void":
        ret_line = f"\n  return {mutable_tensors[-1]};"

    return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}  {api}({call_args_str});{ret_line}
}}"""


def gen_factory(op, fn_type, ret_type, args, func=None):
    """
    zeros / scalar_tensor / arange / arange.start_step / new_ones: build device tensor directly.
    Faithful args end with dtype/layout/device/pin_memory. Create via at::empty on
    the requested (or PrivateUse1) device, then fill.
    """
    kn = kernel_name(fn_type)
    names = [n for _, n in args]
    has_self = args and "at::Tensor" in args[0][0]

    # option field names present in the faithful signature
    def opt(name, default):
        return f"{name}.value_or({default})" if name in names else default

    dtype_default = f"{names[0]}.scalar_type()" if has_self else "at::kFloat"
    layout_default = f"{names[0]}.layout()" if has_self else "at::kStrided"
    device_default = (
        f"{names[0]}.device()" if has_self else "at::Device(at::kPrivateUse1, 0)"
    )

    options = (
        "  auto options = at::TensorOptions()\n"
        f"    .dtype({opt('dtype', dtype_default)})\n"
        f"    .layout({opt('layout', layout_default)})\n"
        f"    .device({opt('device', device_default)})\n"
        f"    .pinned_memory({opt('pin_memory', 'false')});"
    )

    base = at_api_base(op)
    size_arg = names[1] if has_self else (names[0] if names else "{}")

    # --- fill-allocators: our own at::empty allocator + a fill. No CUDA compute
    #     kernel, no recursion (at::empty is hand-registered as the real allocator). ---
    if base in ("zeros", "new_zeros"):
        make = f"  auto result = at::empty({size_arg}, options);\n  result.zero_();"
    elif base in ("ones", "new_ones"):
        make = f"  auto result = at::empty({size_arg}, options);\n  result.fill_(1);"
    elif base in ("full", "new_full"):
        # fill value is the lone by-value Scalar arg (not Scalar[] / optional<Scalar>)
        fill_val = next(
            (
                n
                for t, n in args
                if "Scalar" in t and "ArrayRef" not in t and "optional" not in t
            ),
            "0",
        )
        make = f"  auto result = at::empty({size_arg}, options);\n  result.fill_({fill_val});"
    elif base == "scalar_tensor":
        make = f"  auto result = at::empty({{}}, options);\n  result.fill_({names[0]});"
    else:
        # --- compute factories (arange/rand/randn/randint/randperm/normal/eye/
        #     linspace/logspace/*_window/fft_*freq/tril_indices/...): must run the
        #     real kernel. Calling it with a PrivateUse1 device re-dispatches into
        #     THIS kernel -> infinite recursion -> stack overflow. Redirect the
        #     device arg to CUDA (hits the external libtorch_cuda.so kernel), then
        #     unbox the result back to flagos. Generalizes the old arange special-case. ---
        device_arg = next((n for t, n in args if "optional<at::Device>" in t), None)
        if device_arg is not None:
            call_names = [
                "::std::optional<at::Device>(_cuda_dev)" if n == device_arg else n
                for _, n in args
            ]
            has_gen_arg = any("Generator" in t for t, _ in args)
            # Unified RNG for generator-LESS factory overloads.
            #
            # torch.randint(...) / torch.randperm(...) call the overloads WITHOUT a
            # `Generator?` arg, so _generator_inject_line (which only fires when the
            # schema already carries one) does nothing and at::randint(...) falls back
            # to ATen's default CUDA generator -- unreachable from torch.manual_seed.
            # These bases all expose a sibling overload with `Generator?` inserted
            # right after the size arg (verified in ATen/ops/{randint,randperm,
            # rand,randn}.h), so inject the flagos shared generator there to route
            # them through the same seedable generator FlagGems uses. randperm falls
            # out for free (its dominant randomness is an internal torch.randint).
            if not has_gen_arg and base in _GEN_FACTORY_BASES:
                # size is the first IntArrayRef/SymIntArrayRef positional (no self here)
                size_name = next(
                    (n for t, n in args if "ArrayRef" in t and "Dimname" not in t),
                    size_arg,
                )
                gen_call_names = []
                for _t, n in args:
                    cn = (
                        "::std::optional<at::Device>(_cuda_dev)"
                        if n == device_arg
                        else n
                    )
                    gen_call_names.append(cn)
                    if n == size_name:
                        gen_call_names.append("_flagos_gen")
                inject = (
                    "  ::std::optional<at::Generator> _flagos_gen = "
                    "at::native::flagos::GetFlagosDefaultCudaGenerator(_cuda_dev.index());\n"
                )
                make = (
                    f"  at::Device _req_dev = {device_arg}.has_value() ? *{device_arg} "
                    f": at::Device(at::kPrivateUse1, 0);\n"
                    "  at::Device _cuda_dev = _req_dev.type() == at::kPrivateUse1\n"
                    "      ? at::Device(at::kCUDA, _req_dev.index()) : _req_dev;\n"
                    f"{inject}"
                    f"  auto result = at::{base}({', '.join(gen_call_names)});\n"
                    "  if (result.device().type() == at::kCUDA) UnboxToFlagos(result);"
                )
                return f"""{ret_type} {kn}({args_decl(args)}) {{
{options}
{make}
  return result;
}}"""
            inject = _generator_inject_line(args, "_cuda_dev.index()")
            make = (
                f"  at::Device _req_dev = {device_arg}.has_value() ? *{device_arg} "
                f": at::Device(at::kPrivateUse1, 0);\n"
                "  at::Device _cuda_dev = _req_dev.type() == at::kPrivateUse1\n"
                "      ? at::Device(at::kCUDA, _req_dev.index()) : _req_dev;\n"
                f"{inject}"
                f"  auto result = at::{base}({', '.join(call_names)});\n"
                "  if (result.device().type() == at::kCUDA) UnboxToFlagos(result);"
            )
        else:
            # no device knob -> best-effort empty allocation
            make = f"  auto result = at::empty({size_arg}, options);"

    return f"""{ret_type} {kn}({args_decl(args)}) {{
{options}
{make}
  return result;
}}"""


def gen_optlist(op, fn_type, ret_type, args, func=None):
    """index.Tensor: box self + each defined optional<Tensor> in the list."""
    kn = kernel_name(fn_type)
    self_name = args[0][1]
    list_name = args[1][1]
    api = f"at::{at_api_base(op)}"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard self_guard({self_name});
  TensorListBoxingGuard indices_guard;
  for (int64_t i = 0; i < static_cast<int64_t>({list_name}.size()); ++i) {{
    auto opt = {list_name}.get(i);
    if (opt.has_value()) {{
      indices_guard.box({{*opt}});
    }}
  }}
  auto result = {api}({self_name}, {list_name});
  UnboxToFlagos(result);
  return result;
}}"""


CATEGORY_GENERATORS = {
    "functional_pure": gen_functional_pure,
    "vector_return": gen_vector_return,
    "inplace": gen_inplace,
    "out_variant": gen_out_variant,
    "tuple_return": gen_tuple_return,
    "foreach_tensorlist": gen_foreach,
    "foreach_out": gen_foreach_out,
    "factory": gen_factory,
    "special_optlist": gen_optlist,
}


# ============================================================================
# Wrapper functions (register.cc side)
# ============================================================================


def gen_wrapper(op, fn_type, dispatcher, ret_type, args):
    """Wrapper<Name> that forwards into the dispatcher (used by TORCH_LIBRARY_IMPL)."""
    wname = "Wrapper" + fn_type[:-2]
    ns = "at::native::flagos::"
    call = f"{ns}{dispatcher}({call_args(args)})"
    if ret_type == "void":
        body = f"  {call};"
    else:
        body = f"  return {call};"
    return f"{ret_type} {wname}({args_decl(args)}) {{\n{body}\n}}", wname


# ============================================================================
# at:: header includes needed by cuda_kernels.cc
# ============================================================================


def api_headers(op_info: Dict) -> List[str]:
    """Header file names come from torchgen's authoritative func.root_name.
    (e.g. schema '__ilshift__.Scalar' -> root_name 'lshift' -> ATen/ops/lshift.h;
    'add.out' and 'add.Tensor' both -> 'add'.) Never derive the header from the
    schema base by stripping underscores -- that mangles dunder operators."""
    bases = set()
    for op in op_info.values():
        bases.add(op["func"].root_name)
    return [f"#include <ATen/ops/{b}.h>" for b in sorted(bases)]


# ============================================================================
# Main
# ============================================================================


def main():
    repo_root = Path(__file__).parent.parent
    conf_path = repo_root / "torch_fl/configs/backends_cuda.conf"
    out_dir = repo_root / "csrc/aten/generated"
    out_dir.mkdir(exist_ok=True)

    print("Loading configuration and schemas...")

    root = Path(torchgen.__file__).parent
    nf = parse_native_yaml(
        str(root / "packaged/ATen/native/native_functions.yaml"),
        str(root / "packaged/ATen/native/tags.yaml"),
    )
    funcs = {str(f.func.name): f for f in nf.native_functions}

    # Ops the templates cannot compile yet; skipped -> fall through to cpu_fallback.
    # Grown empirically during the compile/import fix loop.
    skip_ops_path = repo_root / "torch_fl/codegen_skip_ops.txt"
    manual_skip = set()
    if skip_ops_path.exists():
        for line in skip_ops_path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                manual_skip.add(line)

    all_cuda = os.environ.get("FLAGOS_CODEGEN_ALL", "").strip() not in (
        "",
        "0",
        "false",
    )

    if all_cuda:
        from torchgen.model import DispatchKey

        cuda_index = nf.backend_indices[DispatchKey.CUDA]
        ops, skipped = enumerate_all_cuda_ops(nf, funcs, cuda_index)
        ops = [o for o in ops if o not in manual_skip]
        n_manual_skip = len(manual_skip)
        print(f"[FULL CUDA MODE] enumerated {len(ops)} ops to generate")
        for reason in sorted(skipped):
            print(f"   skipped[{reason}]: {len(skipped[reason])}")
        print(f"   skipped[template_skip_list]: {n_manual_skip}")
    else:
        ops = []
        for line in conf_path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line or "=" not in line:
                continue
            op, backend = line.split("=", 1)
            if backend.strip() == "cuda":
                ops.append(op.strip())
        print(f"Found {len(ops)} ops in backends_cuda.conf")

    op_info = {}
    categories = defaultdict(list)

    # Generate signatures per-operator with correct IListRef/ArrayRef setting
    for op in ops:
        if op not in funcs:
            print(f"  WARNING: {op} not in native_functions.yaml", file=sys.stderr)
            continue
        func = funcs[op]
        try:
            cat = detect_category(func)
            fn_type, dispatcher = schema_to_cpp_name(op)

            # Determine if this op needs ArrayRef (vs IListRef default)
            use_arrayref = should_use_arrayref(func)

            # Generate signature with operator-specific IListRef/ArrayRef and
            # const-ref-for-mutable-tensors settings, both taken from torchgen's
            # authoritative per-op flags (matches PyTorch's own registration).
            with local.parametrize(
                use_const_ref_for_mutable_tensors=func.use_const_ref_for_mutable_tensors,
                use_ilistref_for_tensor_lists=not use_arrayref,  # False=ArrayRef, True=IListRef
            ):
                # Use unified CppSignature (faithful) for typedef, kernel, AND wrapper
                ptr_type, ret_type, args = unified_sig(func)
        except Exception as e:
            if all_cuda:
                print(
                    f"  SKIP {op}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr
                )
                continue
            raise

        categories[cat].append(op)
        op_info[op] = dict(
            fn_type=fn_type,
            dispatcher=dispatcher,
            category=cat,
            ptr_type=ptr_type,
            ret_type=ret_type,
            args=args,
            func=func,
        )

    print("\nCategory breakdown:")
    for cat in sorted(categories):
        print(f"  {cat:20s} {len(categories[cat]):3d} ops")

    # ---- discover FlagGems Python-path ops ----
    # Only meaningful in full-CUDA mode (all generated ops have dispatchers).
    # discover_flaggems_ops returns {op: (gems_func_qualname, category)}; keep
    # only ops that are actually in op_info (have a generated dispatcher slot).
    flaggems_py = discover_flaggems_ops(set(op_info), funcs)
    flaggems_py = {op: v for op, v in flaggems_py.items() if op in op_info}
    py_cat_counts = defaultdict(int)
    for _op, (_q, _c, _kw) in flaggems_py.items():
        py_cat_counts[_c] += 1
    print(f"\nFlagGems Python-path ops discovered: {len(flaggems_py)}")
    for cat in sorted(py_cat_counts):
        print(f"  {cat:20s} {py_cat_counts[cat]:3d} ops")

    # ---- ops.h ----
    print("\nGenerating ops.h...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        "#pragma once",
        "",
        "#include <ATen/core/Tensor.h>",
        '#include "../dispatcher.h"',
        "",
        "namespace at::native::flagos {",
        "",
    ]
    for op in sorted(op_info):
        i = op_info[op]
        lines.append(f"using {i['fn_type']} = {i['ptr_type']};")
        lines.append(f"DECLARE_DISPATCHER({i['fn_type']}, {i['dispatcher']})")
        lines.append("")
    lines.append("} // namespace at::native::flagos")
    (out_dir / "ops.h").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} declarations")

    # ---- ops.cc ----
    print("Generating ops.cc...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        '#include "ops.h"',
        "",
        "namespace at::native::flagos {",
        "",
    ]
    for op in sorted(op_info):
        i = op_info[op]
        lines.append(
            f'ADD_IMPL_TO_DISPATCHER({i["fn_type"]}, {i["dispatcher"]}, "{op}")'
        )
    lines.append("")
    lines.append("} // namespace at::native::flagos")
    (out_dir / "ops.cc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} definitions")

    # ---- cuda_kernels.cc ----
    print("Generating cuda_kernels.cc...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        '#include "ops.h"',
        '#include "../device_boxing.h"',
        '#include "../backends/flagos/python_op_caller.h"',
        '#include "../../runtime/generator.h"',
        "",
        "#include <vector>",
        "#include <tuple>",
        "#include <mutex>",
        "",
    ]
    lines += api_headers(op_info)
    lines += [
        "",
        "namespace at::native::flagos {",
        "namespace {",
        "",
        "// Translate a caller-supplied generator into one the vendor CUDA RNG",
        "// kernels accept; returns the input unchanged when no translation is",
        "// needed.",
        "//",
        '// torch.Generator(device="flagos") creates a PrivateUse1 generator',
        "// backed by CPU mt19937 state, which vendor philox kernels cannot",
        "// consume. The old prelude TORCH_CHECK-rejected it (safe against the",
        "// generator-key redispatch recursion, but it broke generic HF code",
        "// that seeds torch.Generator from the runtime device string, e.g.",
        "// diffusers randn_tensor). So draw this op's seed from the flagos",
        "// generator through ReserveSeed -- mutex-guarded, advancing the",
        "// caller-visible state exactly once per call, matching the per-op",
        "// seed-reservation semantics the flaggems/native routes use -- and",
        "// reseed a *clone* of the shared CUDA generator, so the vendor",
        "// kernel runs on a fresh philox stream anchored at that seed while",
        "// torch.manual_seed on the shared default generator stays untouched.",
        "// The clone is taken under the shared generator's lock, per Note",
        "// [Acquire lock when using random generators].",
        "//",
        "// The translation also preserves the anti-recursion property of the",
        "// old reject: after boxing, every tensor in the inner at::<op> call",
        "// is CUDA, so forwarding a CUDA-typed generator keeps dispatch on",
        "// the CUDA path and never re-enters this PrivateUse1 kernel. CUDA",
        "// generators pass through unchanged; any other device type keeps",
        "// upstream ATen's native generator type error.",
        "::std::optional<at::Generator> TranslateGeneratorForCudaBoxing(",
        "    ::std::optional<at::Generator> generator,",
        "    c10::DeviceIndex cuda_index) {",
        "  if (generator.has_value() && generator->defined() &&",
        "      generator->device().type() == c10::DeviceType::PrivateUse1) {",
        "    // ReserveSeed takes the caller generator's own lock internally,",
        "    // so draw the per-op seed first and never hold both generator",
        "    // locks at once.",
        "    const uint64_t seed = c10::flagos::ReserveSeed(generator, cuda_index);",
        "    at::Generator shared =",
        "        at::native::flagos::GetFlagosDefaultCudaGenerator(cuda_index);",
        "    at::Generator translated;",
        "    {",
        "      // clone() copies the shared generator's philox state without",
        "      // synchronization while a concurrent default-generator draw",
        "      // mutates it; hold the shared generator's lock while copying",
        "      // (Note [Acquire lock when using random generators]).",
        "      std::lock_guard<std::mutex> lock(shared.mutex());",
        "      translated = shared.clone();",
        "    }",
        "    // The clone is private to this call, so reseeding it needs no",
        "    // lock.",
        "    translated.set_current_seed(seed);",
        "    return translated;",
        "  }",
        "  return generator;",
        "}",
        "",
    ]
    for op in sorted(op_info):
        i = op_info[op]
        gen = CATEGORY_GENERATORS[i["category"]]
        lines.append(gen(op, i["fn_type"], i["ret_type"], i["args"], i["func"]))
        lines.append("")
    lines.append("} // namespace")
    lines.append("")
    for op in sorted(op_info):
        i = op_info[op]
        kn = kernel_name(i["fn_type"])
        lines.append(
            f"REGISTER_IMPL_TO_DISPATCHER({i['fn_type']}, {i['dispatcher']}, Backend::kCuda, {kn})"
        )
    lines.append("")
    lines.append("} // namespace at::native::flagos")
    (out_dir / "cuda_kernels.cc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} kernels")

    # ---- flaggems_python_kernels.cc ----
    # FlagGems Python-path kernels (Backend::kFlagOsPython slot) for the ops
    # auto-discovered by discover_flaggems_ops(). The whole file body is
    # guarded by FLAGOS_FLAGGEMS_PYTHON so it is a no-op unless the build sets
    # -DFLAGOS_FLAGGEMS_PYTHON (FLAGGEMS_PYTHON=ON), keeping non-flaggems builds
    # from pulling in python_op_caller / pybind.
    print("Generating flaggems_python_kernels.cc...")
    py_ops = [op for op in sorted(op_info) if op in flaggems_py]
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        "#ifdef FLAGOS_FLAGGEMS_PYTHON",
        "",
        '#include "ops.h"',
        '#include "../device_boxing.h"',
        '#include "../backends/flagos/python_op_caller.h"',
        "",
        "namespace at::native::flagos {",
        "namespace {",
        "",
    ]
    for op in py_ops:
        i = op_info[op]
        gems_func, category, kwargs = flaggems_py[op]
        lines.append(
            gen_flaggems_python_kernel(
                op,
                i["fn_type"],
                i["ret_type"],
                i["args"],
                gems_func,
                category,
                i["func"],
                kwargs,
            )
        )
        lines.append("")
    lines.append("} // namespace")
    lines.append("")
    for op in py_ops:
        i = op_info[op]
        pkn = python_kernel_name(i["fn_type"])
        lines.append(
            f"REGISTER_IMPL_TO_DISPATCHER({i['fn_type']}, {i['dispatcher']}, "
            f"Backend::kFlagOsPython, {pkn})"
        )
    lines.append("")
    lines.append("} // namespace at::native::flagos")
    lines.append("")
    lines.append("#endif  // FLAGOS_FLAGGEMS_PYTHON")
    (out_dir / "flaggems_python_kernels.cc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(py_ops)} flaggems-python kernels")

    # ---- register.inc ----
    print("Generating register.inc...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "// Included by register.cc: wrapper fns + m.impl() lines.",
        "",
        "// ---- wrapper functions ----",
        "#ifdef FLAGOS_GEN_WRAPPERS",
    ]
    impl_lines = []
    for op in sorted(op_info):
        i = op_info[op]
        wrapper, wname = gen_wrapper(
            op, i["fn_type"], i["dispatcher"], i["ret_type"], i["args"]
        )
        lines.append(wrapper)
        impl_lines.append(f'  m.impl("{op}", {wname});')
    lines.append("#endif  // FLAGOS_GEN_WRAPPERS")
    lines.append("")
    lines.append("// ---- m.impl() registrations ----")
    lines.append("#ifdef FLAGOS_GEN_IMPLS")
    lines += impl_lines
    lines.append("#endif  // FLAGOS_GEN_IMPLS")
    (out_dir / "register.inc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} wrappers + impls")

    # In full mode, regenerate backends_cuda.conf so GetBackendForOp routes every
    # generated op to cuda. Ops NOT in op_info (skipped) are absent -> they hit
    # the PrivateUse1 cpu_fallback, exactly as before this change.
    if all_cuda:
        # Every generated conf carries the repo's Apache header; without it each
        # regeneration would strip the header off the checked-in files.
        conf_license = [
            "# Copyright 2026 FlagOS Contributors",
            "#",
            '# Licensed under the Apache License, Version 2.0 (the "License");',
            "# you may not use this file except in compliance with the License.",
            "# You may obtain a copy of the License at",
            "#",
            "#     http://www.apache.org/licenses/LICENSE-2.0",
            "#",
            "# Unless required by applicable law or agreed to in writing, software",
            '# distributed under the License is distributed on an "AS IS" BASIS,',
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            "# See the License for the specific language governing permissions and",
            "# limitations under the License.",
            "",
        ]
        conf_lines = conf_license + [
            "# flagos op backend config -- AUTO-GENERATED (full CUDA mode)",
            "# Regenerated by scripts/codegen_ops.py with FLAGOS_CODEGEN_ALL=1.",
            "# Every generated boxing kernel is routed to the cuda backend; ops not",
            "# listed here are not registered and fall through to cpu_fallback.",
            "#",
            "# Format: op_name = backend   (backend: flaggems | flagos_python | cuda)",
            "",
        ]
        for op in sorted(op_info):
            conf_lines.append(f"{op} = cuda")
        conf_path.write_text("\n".join(conf_lines) + "\n")
        print(f"   regenerated {conf_path.name} with {len(op_info)} cuda routes")

        # Ops whose flag_gems implementation falls back to `torch.<op>` when the
        # input is not on a "cuda" device. Our flagos tensors are genuinely
        # PrivateUse1 (device.type == "flagos"), so that fallback re-enters the
        # PrivateUse1 dispatcher, lands on the same flagos_python kernel, and
        # recurses until RecursionError. Unlike the device-assert ops (which
        # raise, and are dropped from codegen via FLAGGEMS_PYTHON_SKIP) the
        # kernel itself is fine -- only the route is wrong -- so the kernel stays
        # generated (reachable via FLAGOS_OP_<op>=flagos_python for debugging)
        # and just the default route goes to the cuda boxing kernel.
        #
        # Verified against flag_gems v5.3.1 by scanning every routed op for a
        # `device.type != "cuda"` guard whose body re-enters torch.<same op>:
        # mul is the only one. as_strided_copy / conv_transpose2d / scaled_mm
        # have the same guard but return a clone / None / False, so they degrade
        # safely. mul_.Tensor is also safe: it calls the gems helper with out=A,
        # which exits via `torch.mul(..., out=)` -> mul.out -> cuda.
        flaggems_recursive_fallback = {
            "mul.Tensor",
        }

        # Ops the auto-discovery routes to flagos_python but whose gems call
        # fails or returns wrong numerics at runtime. Discovery only checks
        # arity/type statically, so these get caught by execution, not by the
        # gates. Kernel stays generated (reachable via
        # FLAGOS_OP_<op>=flagos_python) and only the default route goes to cuda.
        #
        # Established by running every newly routed op against its CPU reference
        # on Hygon DCU (flag_gems 5.4.0dev, triton 3.6.0) and re-checking each
        # failure on the cuda route: all 26 pass as cuda and fail as
        # flagos_python, so each entry is a genuine route regression.
        #
        # Grouped by cause:
        #  1. gems declares `out` as a REQUIRED keyword-only arg, but the
        #     out_variant caller passed only the non-out args and copy_'d the
        #     result -> TypeError: missing keyword-only argument 'out'.
        #     FIXED at the source: discover_flaggems_ops now routes these
        #     through out_variant_gemskwout (out forwarded by name) or, for the
        #     `out=None`-but-raises signatures like logit_out, through
        #     out_variant_gemsout (out passed positionally). The entries stay
        #     listed until the DCU re-test confirms them, so this group is a
        #     conservative hold, not a known breakage -- drop an op from here
        #     once it passes on hardware.
        #  2. gems asserts the input is a real CUDA tensor (device.type ==
        #     "cuda"); a flagos PrivateUse1 tensor trips the assert.
        #  3. DTK triton (hcu backend) fails to compile the gems kernel.
        #  4. wrong numerics on the gems path (verified correct via cuda).
        flaggems_runtime_broken = {
            # (1) gems `out` is required keyword-only -- codegen fixed, pending
            # a DCU re-test before these come off the list.
            "addcdiv.out",
            "addcmul.out",
            "as_strided_copy.out",
            "baddbmm.out",
            "logit.out",
            "_log_softmax.out",
            "_log_softmax_backward_data.out",
            "median.dim_values",
            "median.out",
            "nanmedian.dim_values",
            "nanmedian.out",
            "smooth_l1_loss.out",
            "_softmax.out",
            "_softmax_backward_data.out",
            # (2) gems asserts a real CUDA tensor
            "_cdist_backward",
            "im2col",
            "smooth_l1_loss",
            "smooth_l1_loss_backward",
            "special_i1.out",
            "_upsample_bilinear2d_aa",
            "_upsample_nearest_exact2d_backward",
            "upsample_trilinear3d",
            # (3) DTK triton cannot compile the gems kernel
            "gcd",
            "gcd.out",
            # (4) wrong numerics or memory safety on the gems path
            "adaptive_max_pool3d_backward",
            "leaky_relu_",
            # layer_norm_backward assumes a 2-D input and computes M from
            # input.shape[0] instead of normalized_shape. For 3-D transformer
            # inputs this makes the weight/bias gradient kernel write past its
            # outputs; keep the operation on the CUDA boxing path until the
            # upstream FlagGems fix lands.
            "native_layer_norm_backward",
            # The gems index_select triton launch is not stream-ordered against
            # the flagos (PrivateUse1) stream that produced the index tensor.
            # In HF cached beam search (DynamicCache.reorder_cache ->
            # index_select(0, beam_idx)) the kernel can read a stale index
            # entry, fail its indices < N validity mask, and leave the output
            # column unwritten (recycled torch.empty bytes), poisoning the KV
            # cache with garbage/NaN. The kernel math itself is correct
            # standalone; the missing happens-before is in the flagos <->
            # flag_gems launch integration. Keep on the CUDA boxing path until
            # launches are ordered with the flagos current stream.
            "index_select",
        }

        # backends_flaggems.conf: same cuda routes, but the auto-discovered
        # FlagGems Python-path ops are flipped to flagos_python. Used for testing
        # / running the FlagGems path (FLAGOS_BACKEND_CONFIG=...backends_flaggems.conf).
        fg_conf_path = repo_root / "torch_fl/configs/backends_flaggems.conf"
        fg_lines = conf_license + [
            "# flagos op backend config -- AUTO-GENERATED (flaggems python mode)",
            "# Regenerated by scripts/codegen_ops.py with FLAGOS_CODEGEN_ALL=1.",
            "# Same as backends_cuda.conf, but the auto-discovered FlagGems ops are",
            "# routed to the flagos_python (kFlagOsPython) slot; all others to cuda.",
            "# Exceptions kept on cuda: flaggems_recursive_fallback (gems re-enters",
            "# torch.<op> for non-cuda device types, which would recurse forever on a",
            "# PrivateUse1 tensor) and flaggems_runtime_broken (gems call raises or",
            "# returns wrong numerics on the flagos device; verified per-op on DCU).",
            "#",
            "# Format: op_name = backend   (backend: flaggems | flagos_python | cuda)",
            "",
        ]
        # Both sets force a route back to cuda, for different reasons; the conf
        # builders only need the union.
        flaggems_forced_cuda = flaggems_recursive_fallback | flaggems_runtime_broken
        n_fg_fallback = 0
        for op in sorted(op_info):
            if op in flaggems_forced_cuda:
                backend = "cuda"
                if op in flaggems_py:
                    n_fg_fallback += 1
            else:
                backend = "flagos_python" if op in flaggems_py else "cuda"
            fg_lines.append(f"{op} = {backend}")
        fg_conf_path.write_text("\n".join(fg_lines) + "\n")
        print(
            f"   regenerated {fg_conf_path.name} with "
            f"{len(flaggems_py) - n_fg_fallback} flagos_python routes "
            f"({n_fg_fallback} recursive-fallback ops forced back to cuda)"
        )

        # backends_metax_flaggems.conf: identical to backends_flaggems.conf, but
        # the ops triton-metax / flag_gems cannot run on the flagos device are
        # forced back to the cuda boxing kernel (maca libtorch_cuda via mcblas)
        # instead of flagos_python. In the MetaX boxing wheel the fallback MUST be
        # cuda, not metax: the hand-written mxcc backend is not registered
        # (METAX_KERNEL=OFF).
        #   - mm/bmm(.out): FlagGems uses a SPLIT_K kwarg triton-metax rejects.
        #   - mean.dim: FlagGems' non-inner-dim path uses a CUDA context that
        #     fails on triton-metax.
        #   - flag_gems device-guarded ops: several flag_gems kernels check
        #     tensor.device.type against flag_gems.device (== "cuda") and either
        #     (a) fall back to torch.<op> -> re-enters flagos_python dispatch ->
        #         infinite recursion (flaggems_recursive_fallback above, folded in
        #         below -- that set is device-independent, so it applies here too),
        #         or
        #     (b) raise ValueError("Inputs must be cuda tensors ...")
        #         (embedding_dense_backward, i0, reflection_pad2d, soft_margin_loss,
        #          special_i0e, special_i1, ...).
        #     Our flagos tensors have device.type "flagos", so both paths fail.
        #     Route these to the cuda boxing kernel (runs directly on maca
        #     libtorch_cuda). Derived from flag_gems ops that guard on device.type.
        # Grow this set as testing reveals more triton-metax / flag_gems gaps.
        metax_triton_fallback = flaggems_forced_cuda | {
            "mm",
            "mm.out",
            "bmm",
            "bmm.out",
            "mean.dim",
            # flag_gems ops that guard on device.type == "cuda" (recurse or raise
            # on the flagos device); route to cuda boxing instead of flagos_python.
            # (mul.Tensor and friends come in via flaggems_recursive_fallback.)
            "conv_transpose2d",
            "scaled_mm",
            "as_strided_copy",
            "embedding_dense_backward",
            "i0",
            "i0.out",
            "i0_",
            "reflection_pad2d",
            "reflection_pad2d.out",
            "reflection_pad3d",
            "reflection_pad3d.out",
            "soft_margin_loss",
            "special_i0e",
            "special_i0e.out",
            "special_i1",
            "special_i1.out",
            "prelu",
            "_prelu_kernel_backward",
            "arcsinh",
            "im2col",
            "lift_fresh_copy",
            "resolve_conj",
            "t_copy",
            "zero",
            "special_gammainc",
            "special_scaled_modified_bessel_k1",
            "_upsample_nearest_exact1d",
            # slice_backward: FlagGems' slice_backward_kernel does an out-of-bounds
            # access on large tensors (e.g. grad_output [1, 6144, 35] scattered into
            # [1, 6144, 32], as in Qwen3.5 linear-attn conv1d bwd), raising an
            # Xnack Error / ATU Fault (0x8) in the shader. On MetaX that fault
            # DISABLES the whole process's mcruntime (mcGetDevice ->
            # mcErrorIllegalAddress), poisoning every subsequent op -- unrecoverable
            # in-process. Route to the cuda boxing kernel, which is bounds-safe.
            "slice_backward",
        }
        mfg_conf_path = repo_root / "torch_fl/configs/backends_metax_flaggems.conf"
        mfg_lines = conf_license + [
            "# flagos op backend config -- AUTO-GENERATED (metax boxing + flaggems)",
            "# Regenerated by scripts/codegen_ops.py with FLAGOS_CODEGEN_ALL=1.",
            "# Same as backends_flaggems.conf, but ops triton-metax cannot run",
            "# (mm/bmm/mean.dim) fall back to the cuda boxing kernel (maca",
            "# libtorch_cuda), NOT metax (mxcc backend is off in boxing mode).",
            "# Selected at runtime by FLAGOS_METAX_BOXING=1 + FLAGOS_USE_FLAGGEMS=1.",
            "#",
            "# Format: op_name = backend   (backend: flaggems | flagos_python | cuda)",
            "",
        ]
        n_metax_fallback = 0
        for op in sorted(op_info):
            if op in metax_triton_fallback:
                backend = "cuda"
                if op in flaggems_py:
                    n_metax_fallback += 1
            else:
                backend = "flagos_python" if op in flaggems_py else "cuda"
            mfg_lines.append(f"{op} = {backend}")
        mfg_conf_path.write_text("\n".join(mfg_lines) + "\n")
        print(
            f"   regenerated {mfg_conf_path.name} "
            f"({n_metax_fallback} flaggems ops forced back to cuda)"
        )

        # backends_dcu_flaggems.conf: identical to backends_flaggems.conf, but the
        # ops DTK's triton (the `hcu` backend) cannot run are forced back to the
        # cuda boxing kernel (which on DCU is libtorch_hip via the CUDA dispatch
        # key). Selected at runtime by ACCELERATOR=dcu + FLAGOS_USE_FLAGGEMS=1.
        #   - slice_backward: the flag_gems kernel triggers a hardware VMFault
        #     ("Invalid address access") on hcu. It only manifests once the
        #     grad it produces is consumed by MIOpen's convolution_backward
        #     (tests/integration/ops/test_conv1d_dispatch.py, C=6144 depthwise);
        #     the kernel's own output metadata and values check out, so this is a
        #     hcu codegen bug, not a shape/stride mismatch on our side.
        #   - silu_backward: the flag_gems kernel calls tl.math.div_rn, whose
        #     lowering (builder.create_precise_divf) is missing in hcu triton --
        #     it returns None, so compilation dies with
        #     AttributeError("'NoneType' object has no attribute 'type'").
        # flaggems_recursive_fallback is folded in: that set is device-independent
        # (the guard those kernels trip is device.type != "cuda", true for any
        # PrivateUse1 tensor), so it applies on DCU exactly as on metax.
        # Note this fallback set overlaps metax_triton_fallback's slice_backward
        # entry by coincidence, not cause: metax hits an out-of-bounds Xnack fault
        # inside the gems kernel, DCU faults only downstream in MIOpen.
        # Grow this set as testing reveals more triton-hcu gaps.
        dcu_triton_fallback = flaggems_forced_cuda | {
            "slice_backward",
            "silu_backward",
        }
        dfg_conf_path = repo_root / "torch_fl/configs/backends_dcu_flaggems.conf"
        dfg_lines = conf_license + [
            "# flagos op backend config -- AUTO-GENERATED (dcu boxing + flaggems)",
            "# Regenerated by scripts/codegen_ops.py with FLAGOS_CODEGEN_ALL=1.",
            "# Same as backends_flaggems.conf, but ops DTK's triton (hcu backend)",
            "# cannot run fall back to the cuda boxing kernel -- which on DCU is",
            "# libtorch_hip, reached via the CUDA dispatch key.",
            "# Selected at runtime by ACCELERATOR=dcu + FLAGOS_USE_FLAGGEMS=1.",
            "#",
            "# Format: op_name = backend   (backend: flaggems | flagos_python | cuda)",
            "",
        ]
        n_dcu_fallback = 0
        for op in sorted(op_info):
            if op in dcu_triton_fallback:
                backend = "cuda"
                if op in flaggems_py:
                    n_dcu_fallback += 1
            else:
                backend = "flagos_python" if op in flaggems_py else "cuda"
            dfg_lines.append(f"{op} = {backend}")
        dfg_conf_path.write_text("\n".join(dfg_lines) + "\n")
        print(
            f"   regenerated {dfg_conf_path.name} "
            f"({n_dcu_fallback} flaggems ops forced back to cuda)"
        )

    print("\nDone. Files in:", out_dir)


if __name__ == "__main__":
    main()
