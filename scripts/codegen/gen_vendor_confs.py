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

"""Generate one full-coverage backend conf per vendor.

Every vendor conf lists the *same* op set -- the authoritative list of ops
torch_fl generates wrappers for -- so operator support is countable by reading
one file per platform instead of diffing sparse configs against each other.

Each op maps to exactly one of five keys:

    flaggems_cpp   FlagGems C++ path (liboperators.so), Backend::kFlagGemsCpp
    flaggems       FlagGems Python/Triton path, Backend::kFlagGems
    tileops        TileOps Python shim path, Backend::kTileOps
    <vendor>       vendor-native kernel (musa, ascend, gcu, metax, ...)
    none           no accelerated impl on this platform -> cpu_fallback

Routing priority is FlagGems-first, tileops as an intermediate Python path,
vendor-native as the fallback for what neither covers, and `none` for ops
with no accelerated implementation:

    flaggems_cpp > flaggems > tileops > <vendor> > none

`flaggems_cpp` is only emitted for the confs a FLAGGEMS_KERNEL=ON build selects
(see FLAGGEMS_CPP_PLATFORMS); everywhere else those ops take the Python path to
the same kernels, because the C++ dispatcher slot is not compiled in.

`tileops` routes to Backend::kTileOps, which calls into
torch_fl.tileops.generated.shims via CallPythonOp_Generic. The 60 ops that
have a TileOps shim implementation are inlined in TILEOPS_OPS below.

`none` is recorded explicitly rather than by omission. That is the whole point
of the full-coverage shape: "absent from the file" and "known to be
unsupported" used to look identical, which made support impossible to count
and let sparse confs rot silently against a growing codegen. Omission is not
even a safe way to say "unsupported" -- GetBackendForOp() returns kFlagGemsCpp on a
table miss, so an unlisted op claims a FlagGems kernel by default.

What the file may claim is bounded by what the platform registers on
PrivateUse1. FlagGems coverage is measured on CUDA and is only a ceiling: an op
this platform does not register cannot reach any kernel, FlagGems included, so
it is written `none`. Registration sets are read from the generated
`*_register.inc` files that csrc/aten/register.cc includes -- the same list the
compiler sees -- which is why only platforms whose registration is a subset of
the op list get a generated conf. See VENDORS.

Usage:
    python3 scripts/codegen/gen_vendor_confs.py            # rewrite vendor confs
    python3 scripts/codegen/gen_vendor_confs.py --check     # exit 1 if any is stale
    python3 scripts/codegen/gen_vendor_confs.py --stats     # print a coverage table
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONF_DIR = REPO_ROOT / "torch_fl/configs"

# tests/unit/test_gen_vendor_confs.py loads this file by path (scripts/ is not an
# importable package), so a bare `import backend_coverage` would not resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The op list every vendor conf must cover: what the build can route at all, read
# from the generated registration list.
#
# This used to name backends_cuda.conf, which codegen_ops.py rewrites with one
# line per generated wrapper, so it agreed with registration by construction --
# but it made every platform's op universe a function of the *CUDA platform's*
# routing table. PPU reads a different conf precisely because the two disagree
# about routing, so deriving PPU's universe from CUDA's conf meant a CUDA-side
# edit that added or dropped an op line resized backends_ppu.conf without
# touching PPU, and PPU overriding a value in a file it also enumerates from is
# the round trip this script avoids everywhere else.
#
# generated/register.inc is the artifact the CUDA line and the PPU line both
# compile (the `#else` branch of csrc/aten/register.cc), which is the same reason
# the vendor kernel sets below are read from the generated C++ rather than from a
# conf.
GENERATED_REGISTER_INC = REPO_ROOT / "csrc/aten/generated/register.inc"

# Where the backend coverage sets come from. These used to be three shared conf
# files (backends_flaggems.conf, backends_flaggems_cpp.conf, backends_tileops.conf)
# that torch_fl/configs/ carried purely as generator input -- they routed nothing
# on their own and no platform selected them. They are now data in
# scripts/codegen/backend_coverage.py, so torch_fl/configs/ holds exactly one conf per
# platform and nothing else.
from backend_coverage import (  # noqa: E402
    FLAGGEMS_CPP_OPS,
    FLAGGEMS_PYTHON_OPS,
    TILEOPS_OPS,
)

# Vendors that get a full-coverage conf.
#
# Only a platform whose PrivateUse1 registration is a *subset* of the op list
# belongs here, because that is what makes `none` honest. Claiming an op on
# PrivateUse1 with no kernel behind it does not fall back -- the call reaches the
# dispatcher and raises on the empty slot -- whereas leaving it unregistered lets
# it box to cpu_fallback. So `none` means cpu_fallback only where registration
# skips the op, and these three platforms are the ones that skip:
#
#   platform  registration branch in csrc/aten/register.cc   m.impl count
#   musa      musa_register.inc + musa_flaggems_register.inc  158
#   gcu       gcu_register.inc + gcu_flaggems_register.inc     546
#   ascend    ascend_register.inc                              372
#
# metax, tsingmicro and dcu fall through to the `#else` branch and register the
# full generated list (~2034), so a `none` entry there would be a guaranteed
# raise rather than a fallback. metax and dcu are handled as CUDA-boxing
# platforms below (see BOXING_PLATFORMS); backends_tsingmicro.conf stays
# hand-written.
#
#   vendor -> (conf naming its native kernels, other .inc files it registers)
#
# The sets are read from the generated C++ rather than from the conf this script
# rewrites, so there is no round trip to survive: registration and routing are
# derived from the same artifact the compiler sees.
VENDORS = {
    "musa": (
        "backends/musa/generated/musa_register.inc",
        ("backends/musa/generated/musa_flaggems_register.inc",),
    ),
    "gcu": (
        "backends/gcu/generated/gcu_register.inc",
        ("backends/gcu/generated/gcu_flaggems_register.inc",),
    ),
    "ascend": ("backends/ascend/generated/ascend_register.inc", ()),
}

# Ops a vendor registers straight from register.cc instead of its .inc, so they
# are invisible to the .inc scan. Ascend claims aten::matmul (and the
# matmul_backward the derivative then binds to) on PrivateUse1 under
# `#if defined(USE_ASCEND)` so the call hits the fused aclnnMatmul rather than
# decomposing into mm/bmm/view; WrapperMatmul reads this conf to decide, so the
# entry has to be here or the fused kernel is unreachable.
EXTRA_NATIVE = {"ascend": {"matmul", "matmul_backward"}}

# CUDA-boxing platforms use the same four-key shape. Their fallback kernel is the
# CUDA boxing kernel, so the `<vendor>` slot is spelled "cuda" and it covers
# every op -- which is why a boxing conf has no `none` entries while a
# native-kernel vendor has many.
#
#   platform -> (fallback backend key, conf naming that platform's triton gaps)
#
# The gap set is per-platform, not a FlagGems defect: the kernel works elsewhere,
# but this platform's triton backend cannot run it, so the op is forced back to
# boxing. MetaX's and DCU's live in codegen_ops.py (flaggems_forced_cuda plus the
# per-platform additions) and are recovered here by diffing that platform's
# existing conf against backends_flaggems.conf, rather than restated -- one
# source of truth, and the per-op diagnosis stays in the codegen comments. PPU's
# is a literal (BOXING_TRITON_GAPS) because it is vendor-first no longer, and the
# diff cannot tell a gap from a policy exception on a flaggems-first platform.
BOXING_PLATFORMS = {
    "metax": "backends_metax.conf",
    "dcu": "backends_dcu.conf",
    # PPU routes FlagGems first; its triton gap set is the mm/bmm family. PPU's
    # triton build rejects the `num_ldmatrixes` kwarg that FlagGems' _hygon
    # mm/bmm kernel passes to triton's mm_kernel: mm raises KeyError at
    # triton/runtime/jit.py:_pack_args, and bmm reaches the same kernel and
    # stalls inside triton compilation. Measured on the PPU runner, routing mm to
    # flaggems turned a 2.87s test_factory_ops.py into 1801s (28 minutes inside
    # test_mm alone) and left the process exiting on SIGSEGV. PPU previously had
    # no conf of its own and read backends_cuda.conf, so it silently inherited
    # every CUDA FlagGems route; a separate conf is what lets the two platforms
    # disagree about mm/bmm. Filed upstream as FlagGems issue #6225.
    "ppu": "backends_ppu.conf",
}
BOXING_FALLBACK = "cuda"

# Generated conf notes for platform-specific FlagGems fallbacks. Keep these in
# the generator so regeneration preserves the measured diagnosis and upstream
# issue reference instead of erasing a hand-edited comment.
BOXING_GAP_NOTES = {
    "dcu": (
        "mm/bmm are pinned to cuda: FlagGems issue #6227 (first profiled",
        "mm hangs until the 50-minute CI timeout on the DCU HCU backend).",
        "mse_loss is pinned to cuda: FlagGems issue #6221 (mse_loss_backward",
        "is not implemented on DCU).",
        "transpose.int is pinned to cuda: FlagGems issue #6219 (transpose is",
        "not implemented in the FlagGems module on DCU).",
        "_conj is pinned to cuda, for a contract reason rather than a compile",
        "one: ATen's conj is a lazy view -- it sets the Conjugate bit and leaves",
        "the storage alone -- which tests/integration/test_math_bits_contract.py",
        "pins, while flag_gems' _conj is a real kernel that materializes the",
        "conjugated values, so registering it on PrivateUse1 replaces the view",
        "with an eager copy and is_conj() comes back False. Boxed to the CUDA",
        "kernel the view survives: measured on DCU, t._conj().is_conj() is True",
        "and the result shares the input's storage (data_ptr equality).",
        "relu/relu_ are pinned to cuda, for the same kind of contract reason:",
        "with them on FlagGems the profiler-parity workload launches gems'",
        "relu_forward_kernel_rank_1 instead of ATen's",
        "at::native::vectorized_elementwise_kernel, and the demangling guard in",
        "tests/integration/test_profiler_parity.py requires the trace to carry",
        "an at::native template kernel -- with relu on FlagGems no kernel name",
        "in that trace contains '::' and the guard fails as vacuous. Measured",
        "on DCU by holding the build fixed and switching only the route: the",
        "flaggems arm's kernel set has no '::' name, the cuda arm's has",
        "at::native::vectorized_elementwise_kernel<4, ...launch_clamp_scalar>.",
    ),
    "ppu": (
        "47 of the 482 FlagGems-covered ops are pinned to cuda; the other 435",
        "route to flaggems. mm/bmm are the first four (FlagGems issue #6225:",
        "PPU rejects the _hygon kernel's num_ldmatrixes kwarg) and 33 are the",
        "2026-09-15 overload survey's measured route-dependent failures. Four",
        "more are reflection-padding routes that survey cannot reach: it derives",
        "`padding` from the rank, so every profile fails ATen's arity check",
        "before the device guard behind it can fire, and that guard is the",
        "device-name mismatch the CUDA alignment fixes and leaves alone on every",
        "other vendor. The last six came from the CI steps that survey cannot",
        "see: the addmm family, whose autotune picks a BLOCK_SIZE_K < 16 config",
        "on small-K shapes that the ppu triton rejects in tl.dot, and _conj,",
        "which FlagGems materializes instead of setting the Conjugate bit. See",
        "BOXING_TRITON_GAPS, which carries each op's failure, and",
        "docs/reference/operator-support.md for the full survey.",
    ),
}

# Boxing platforms whose triton gap set is stated here instead of recovered by
# diffing the generated conf (see boxing_triton_gaps). Those two are the same
# thing only while the platform is vendor-first: the diff infers "this op is a
# gap" from "this op is spelled `cuda` even though FlagGems covers it", which
# stops being readable the moment a platform routes FlagGems first by policy --
# then the file's `cuda` entries are the exceptions and the diff would return
# just them, which is exactly right the first time and self-erasing the second.
#
# PPU: FlagGems-first is the intent (most of its 478 covered ops run the Python
# path), so its gaps are a measured exception list rather than a residue. The
# first four are the mm/bmm family from FlagGems issue #6225: PPU's triton
# rejects the `num_ldmatrixes` kwarg FlagGems' _hygon mm/bmm kernel passes to
# triton's mm_kernel -- mm raises KeyError at triton/runtime/jit.py:_pack_args
# and bmm reaches the same kernel and stalls inside triton compilation (measured:
# routing mm to flaggems turned a 2.87s test_factory_ops.py into 1801s, 28
# minutes inside test_mm alone, then SIGSEGV). `mm.out`/`bmm.out` are the out=
# spellings of the same kernel.
#
# Add an op here when a run reports it failing on the flaggems route, with the
# failure recorded next to it. The rest of the PPU set is the 2026-09-15 overload
# survey's verdict on the widened route table: tests/manual/flaggems_overload_survey.py
# reports 42 of the 478 routes with no correct case, and re-running those 42 with
# FLAGOS_OP_<op>=cuda splits them into 33 that pass on the boxing kernel and 9
# that fail on both. Only the 33 are route-dependent and appear here; the 9 stay
# on flaggems and their failures are recorded as pre-existing in
# docs/reference/operator-support.md. Each entry below carries the failure the
# survey measured, in the same profile the route is reached with.
BOXING_TRITON_GAPS = {
    "ppu": {
        # -- mm/bmm family, FlagGems issue #6225 (see above) --
        "bmm",
        "bmm.out",
        "mm",
        "mm.out",
        # -- FlagGems' addmm autotune selects a BLOCK_SIZE_K < 16 config on
        # small-K shapes, and FlagTree's ppu triton rejects that in tl.dot:
        # min_dot_size[2] is 16, so triton/language/semantic.py:1555 raises
        # "Input shapes should have M >= 1, N >= 1 and K >= 16" as a
        # CompilationError inside the autotuner's do_bench_cudagraph replay.
        # Measured on flagos:0 with flag_gems 5.4.0rc2.post1+gd45285ba6:
        # M/N/K = 4/8/8, 128/128/8, 2/2/1 and 4/4/4 fail while 4/8/12, 8/8/15,
        # 4/8/16 and 8/8/31 pass, so the trigger is which config the alignment
        # strategy picks rather than K < 16 outright. nn.Linear(8, 8) over a
        # (4, 8) input is the shape that reaches it from
        # tests/integration/test_factory_ops.py::TestCopyTransfer::
        # test_module_cpu_after_forward. baddbmm has the same dot structure and
        # does not fail: measured OK at K = 8 and K = 16. --
        "addmm",
        "addmm.dtype",
        "addmm.dtype_out",
        "addmm.out",
        "addmm_",
        # -- `_conj` is a metadata operator in PyTorch: torch.conj() sets the
        # Conjugate bit and leaves storage untouched. FlagGems' _conj
        # (flag_gems/ops/_conj.py) computes the conjugate into a fresh tensor
        # instead, so torch.conj(x).is_conj reads False on the flaggems route --
        # right values, wrong contract. tests/integration/test_math_bits_
        # contract.py asserts that contract for every backend and takes it as a
        # precondition ("torch.conj must stay lazy for this contract to apply"),
        # which fails 7 of its cases on the flaggems route and passes all 12 on
        # the boxing kernel. The value-level overload survey cannot see this
        # one: eager materialization is numerically indistinguishable. --
        "_conj",
        # -- FlagGems' reflection-padding wrappers reject a flagos operand before
        # they reach a kernel, and the overload survey cannot see it: the harness
        # derives `padding` from the tensor rank, so ATen's arity check fails
        # every profile first ("padding size is expected to be 4, but got: 1"
        # for 2d, "... 6, but got: 1" for 3d) and the device guard underneath it
        # is never reached. All seven cases on each of the four routes are
        # INVALID_CASE, which is why they appear in no rollback group even
        # though they cannot run. Both modules compare the operand against
        # flag_gems' own device name --
        #   ops/reflection_pad2d.py:106,136 and ops/reflection_pad3d.py:125,158
        #   if input.device.type != flag_gems.device:
        #       raise ValueError(f"input must be a {flag_gems.device} tensor")
        # -- and on PPU that name is "cuda" while the operands these two ops are
        # handed are flagos tensors -- torch.accelerator.current_accelerator()
        # is `flagos`, and such a tensor reports device(type='flagos') with
        # is_cuda False -- so the guard rejects them. Measured with the route
        # forced:
        #   FLAGOS_OP_reflection_pad2d=flaggems
        #   torch.ops.aten.reflection_pad2d.default(<flagos tensor>, [1,1,1,1])
        #   -> ValueError: input must be a cuda tensor
        # while the same call on a tensor made with device="cuda" returns the
        # right answer: the boxing backend makes that spelling CUDA-typed, so
        # the failure is the guard and not the kernel. See
        # docs/reference/operator-support.md: the name comes from the `_thead`
        # vendor descriptor (device_name="cuda"), and
        # torch_fl.accelerator.cuda._cuda_compat.patch_flaggems_device_name,
        # which realigns it on the NVIDIA path, returns immediately for any
        # vendor but nvidia. reflection_pad1d and reflection_pad3d_backward
        # carry no such guard and stay on flaggems. --
        "reflection_pad2d",
        "reflection_pad2d.out",
        "reflection_pad3d",
        "reflection_pad3d.out",
        # -- FlagGems refuses the flagos device before running a kernel. The
        # tensor is on the PrivateUse1 device, so `is_cuda` is False and each of
        # these raises its own guard rather than a triton failure:
        #   i0 / i0.out        "input tensor must be on cuda device"
        #   special_i0e/i1     "Tensors must be cuda tensors"
        #   bessel_k0(.out)    "Tensors must be CUDA tensors"
        #   bessel_k1(.out)    "input tensor must be on CUDA device"
        #   soft_margin_loss   "input and target must be cuda tensors"
        #   upsample_bicubic2d "This Triton kernel requires CUDA tensors" --
        "i0",
        "i0.out",
        "soft_margin_loss",
        "special_i0e",
        "special_i1",
        "special_modified_bessel_k0",
        "special_modified_bessel_k0.out",
        "special_scaled_modified_bessel_k1",
        "special_scaled_modified_bessel_k1.out",
        "upsample_bicubic2d",
        # -- triton CompilationError on the FlagTree ppu backend, no kernel
        # produced: randint at 20:15, randint_like at 25:13,
        # norm.ScalarOpt_dim at 14:16 --
        "norm.ScalarOpt_dim",
        "randint",
        "randint_like",
        # -- `out=` is not written through, or the alias does not accept the
        # schema's arguments: cosh.out "cosh_out() missing 1 required positional
        # argument: 'out'"; sum.out returns the input's shape (32, 32) instead of
        # the reduction (); mul_.Tensor dispatches to aten::mul.out, which has no
        # CPU fallback --
        "cosh.out",
        "mul_.Tensor",
        "sum.out",
        # -- numerically wrong, every profile that ran: measured max_diff on
        # float32 unless noted --
        "_pdist_backward",  # nan
        "elu",  # 0.85
        "elu_",  # 0.76
        "elu_backward",  # 0.93
        "histc",  # 1024
        "range",  # returns float64 for a float32 range
        "special_chebyshev_polynomial_v",  # nan
        "special_chebyshev_polynomial_w",  # 1.0
        "special_shifted_chebyshev_polynomial_u",  # 798.6
        "special_shifted_chebyshev_polynomial_w",  # 2.1e11
        # -- raises on every profile that ran --
        "_unique2",  # "return arity 3 != 1"
        "dequantize.self",  # int_repr has no CPU kernel
        "norm.Scalar",  # "look up dimensions by name, got: name = None"
        "randperm",  # bare AssertionError
        "special_chebyshev_polynomial_u",  # n must be in [0, 5]
        "special_hermite_polynomial_h",  # n must be in [0, 9]
        "unique_consecutive",  # RecursionError
    },
}

# Where the registration .inc files live, relative to the repo root.
CSRC_DIR = REPO_ROOT / "csrc/aten"

# Which generated confs may route to `flaggems_cpp` at all.
#
# That slot is Backend::kFlagGemsCpp, registered in csrc/aten/flaggems_cpp_kernels.cc
# behind `#ifdef FLAGOS_FLAGGEMS_CPP`, which csrc/CMakeLists.txt defines only for
# a FLAGGEMS_KERNEL=ON build -- and CMakeLists.txt force-sets FLAGGEMS_KERNEL OFF
# for ascend, dcu, musa, bpu, tsingmicro and non-boxing metax, because the path
# needs FlagGems' liboperators.so built for that vendor.
#
# One platform now means one conf, so a conf can no longer be reserved for the
# opt-in build that compiles the slot: backends_metax.conf is what every MetaX
# build reads, with or without FLAGGEMS_KERNEL=ON. The key stays legal there
# because Dispatcher::GetFn (csrc/aten/dispatcher.h) degrades kFlagGemsCpp to the
# boxing kernel when the slot is empty, instead of raising "backend not
# registered". So the 17 measured C++ routes are used when a MACA-built FlagGems
# is present and silently box when it is not -- one file, both builds.
#
# The native-kernel vendors (musa/gcu/ascend) still withhold the key. Not for
# safety, but because it would be noise: FlagGems' C++ runtime is not built for
# any of them, so every entry would degrade, and `flaggems` records the same
# routing decision honestly. Every op in the C++ set is also in the Python set
# (test_flaggems_cpp_set_is_a_subset_of_the_python_set pins that), so they reach
# the same kernel through python_op_caller; only the GIL-free entry point
# differs. (backends_flaggems_cpp.conf, the generic CUDA one, is codegen output
# and not written here.)
FLAGGEMS_CPP_PLATFORMS = {"metax"}

# Native Ascend wheels deliberately disable FLAGGEMS_PYTHON: the CI image does
# not ship Triton/FlagGems, and set_env_ascend.sh sets FLAGGEMS_PYTHON=0. Keep
# the generated conf honest by not routing Ascend through a dispatcher slot that
# is absent from the wheel. Boxing platforms compile the Python caller and may
# use the measured FlagGems coverage -- including PPU, whose FlagTree install
# (triton with the `ppu` backend) is what set_env_ppu.sh provisions.
FLAGGEMS_PYTHON_PLATFORMS = {"ascend", "metax", "dcu", "gcu", "musa", "ppu"}

# Ops a native-kernel vendor must keep on its own kernel because that platform's
# triton backend cannot compile the FlagGems kernel. The boxing platforms express
# this by pinning the op to `cuda` and recovering it with boxing_triton_gaps();
# a native vendor's conf spells the fallback as the vendor name, which is also
# what an op with no FlagGems coverage looks like, so the two cases cannot be
# told apart by reading the file back. Stated here instead.
#
# ascend: pow/rsqrt used to be excluded here. They crashed bishengir-compile with
# "LLVM ERROR: unsupported datatype for arith::ExtFOp to hfusion" on triton-ascend
# 3.2.2 / CANN 9.0.0 / Ascend910_9382 (CI run 34792677968, filed upstream as
# FlagGems issue #6226). That stack is no longer what Ascend runs -- FlagGems now
# goes through FlagTree 0.6.2a1+ascend3.5 (Triton 3.5) -- so the exclusion was
# re-measured rather than inherited. On Ascend910 with CANN 9.0.0, FlagTree
# 0.6.2a1+ascend3.5 and FlagGems d45285ba, all five ops -- pow.Scalar,
# pow.Tensor_Scalar, pow.Tensor_Tensor, rsqrt and rsqrt_ -- compile, run on the
# FlagGems route and match the CPU reference for shapes (1,), (7,), (128, 256)
# and (3, 5, 17) in fp32/fp16/bf16, three seeds each, plus the backward through
# the FlagGems kernels. They are no longer excluded.
#
# ascend: mul_.Tensor is here for a defect inside FlagGems, not in the Triton
# backend. flag_gems/ops/mul.py is the only operator module in FlagGems that gates
# its Triton path on the runtime device *name* -- `_DEVICE_NAME =
# runtime_device.name` (line 34) -- and when the operand's device type differs it
# falls back with `torch.ops.aten.mul.out.redispatch(_FALLBACK_KEYSET, a, b,
# out=out)` (line 589). On this backend the runtime name is "npu" while the
# tensor's device type is "flagos", so the fallback always fires, and it hands the
# boxed `mul.out` schema the caller's raw operand: `t.mul_(b)` dies with
# "Unable to cast ... to Tensor" against aten::mul.out(Tensor self, Tensor other,
# ...), and the `.Scalar` spelling with "Expected a value of type 'Tensor' for
# argument 'other' but instead found type 'float'". ATen boxes mul_.Scalar onto
# mul_.Tensor, so one entry covers both spellings. No other elementwise op is
# affected because no other module carries that gate -- measured: add_.Tensor,
# div_.Tensor, sub_.Tensor and add_.Scalar all pass on the FlagGems route while
# mul_.Tensor and mul_.Scalar both fail. Ascend has a native mul kernel, so the
# route back is free. mul.Tensor is outside the FlagGems coverage set already, so
# only the in-place form needed an entry. Measured on Ascend910 with FlagGems
# d45285ba + FlagTree 0.6.2a1+ascend3.5. Not yet filed upstream.
#
# ascend: sort/sort.stable fail inside the kernel's own compilation:
# flag_gems/ops/sort.py:111 is rejected by BiShengHIR with "error: ub overflow,
# requires 4620288 bits while 1572864 bits available! (possible reason: tiling
# basic block is too large ...)", and the op never launches (measured both through
# torch.sort and torch.sort(stable=True) on Ascend910 with FlagTree
# 0.6.2a1+ascend3.5). This is the same kernel the MUSA entry below routes around
# for a different reason -- there the failure is mudnn's missing uint32 cast, here
# it is the Ascend compiler's unified-buffer budget. Ascend's own sort emits both
# the values and the indices, and argsort/msort are composites over sort, so one
# entry covers all four.
#
# ascend: the last three generator-consuming RNG ops, which fail *after* the
# philox bridge (torch_fl/accelerator/ascend/_ascend_compat.py) fixes the state
# contract, plus randperm, which was never a generator problem at all. Each dies
# inside BiShengHIR's compilation of the FlagGems kernel, so the op never
# launches. Measured on Ascend910 with FlagGems 6d31db9aa + FlagTree
# 0.6.2a1+ascend3.5:
#
#   rand, rand_like  flag_gems/ops/rand.py:35 is rejected with "ub overflow,
#                    requires 2294016 bits while 1572864 bits available!",
#                    the same unified-buffer budget that rejects sort below.
#                    The .out overloads are already native and unaffected.
#   exponential_     flag_gems/ops/exponential_.py:152 rejects the kernel's own
#                    compare: "'arith.cmpi' op attribute 'predicate' is not a
#                    valid ...". FlagGems' Ascend vendor list already carries this
#                    op in CUSTOMIZED_UNUSED_OPS, so upstream agrees it is unused.
#   randperm         wrong before it is slow. randperm(50) returns all zeros --
#                    silently, and identically for two different seeds, which is
#                    what the RNG dispatch test reads as "ignores the seed" --
#                    while randperm(2000) fails to compile with "ub overflow,
#                    requires 10092544 bits" through topk.py:266 -> topk.py:193.
#                    The failure is shape-dependent, so a small-N check proves
#                    nothing about the route.
#   native_dropout,  flag_gems/ops/dropout.py:33 is rejected the same way, and
#   native_dropout_  again only for large tensors: the 100_000-element dropout in
#   backward         tests/integration/ops/test_rng_dispatch.py needs 5112576
#                    bits of unified buffer, while the 256-element one the same
#                    file uses elsewhere fits. Both overloads have an aclnn
#                    kernel, so the route back is free; the ATen composite the
#                    test describes as unreachable-by-seed is not what `ascend`
#                    resolves to.
#
# Ascend's own kernels implement all six, so the route back is free.
#
# ascend: every comparison overload FlagGems routes here, because the kernels
# evaluate the comparison in float32 -- flag_gems/ops/{ge,gt,le,lt}.py apply
# `x.to(tl.float32)` to the operand and eq/ne cast both sides. That is exact for
# the float dtypes these kernels were written for and silently wrong for integer
# operands wider than float32's 24-bit mantissa. Two measured defects, both on
# Ascend910 with FlagGems 6d31db9aa + FlagTree 0.6.2a1+ascend3.5:
#
#   1. the cast itself. `ge` over [2**53+1, 2**53] against the same operands
#      shifted by one returns [True, True] where ATen returns [False, False],
#      and `eq` over four distinct int64 values reports all four equal.
#   2. the scalar operand. A Python int outside int32 range collapses to 0
#      before the comparison, which is what `torch.all(values >= -(1 << 63))` in
#      tests/integration/ops/test_rng_dispatch.py trips over. A one-line Triton
#      kernel with no FlagGems involved reproduces it:
#      `tl.load(x).to(tl.float32) >= y` over [-(1<<63), -1, 0, 1] returns
#      [F, F, T, T] for y = -(1<<63) and [F, T, T, T] for y = 1<<40, where ATen
#      returns all-False for both. The same kernel with a float y, or with a
#      0-dim int64 tensor operand, is correct -- so it is the int-to-float32
#      scalar path, not the width of the value.
#
# (1) is a FlagGems defect and (2) is a backend one; neither has a routing-free
# workaround, and both are silent. Ascend implements all twelve overloads
# through aclnn, so the route back is free. Not yet filed upstream.
#
# ascend: mm and mm.out, for the same defect MetaX routes around below -- the
# Ascend tune config tunes a kernel that cannot accept one of its own keys.
# flag_gems/runtime/backend/_ascend/tune_configs.yaml declares
# BLOCK_M/BLOCK_N/BLOCK_K/SPLIT_K for `mm:`, while the kernel that key is fetched
# for -- flag_gems/ops/mm.py:mm_kernel_general, wrapped in
# `@libtuner(configs=runtime.get_tuned_config("mm"))` and reached from `mm` /
# `mm_out` through the general path -- takes only
# BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/IS_FP64. The first call therefore dies inside
# the autotuner's own benchmark run, before any kernel is compiled, with
# "KeyError: 'Keyword argument SPLIT_K was specified but unrecognised'" raised
# from triton/spec/ascend/runtime/jit.py:_pack_args. Measured on Ascend910 /
# CANN 9.0.0 / FlagTree 0.6.2a1+ascend3.5 against the revision CI pins (FlagGems
# d45285ba) and against master (6d31db9aa); the yaml entry and the kernel
# signature are identical in both. Ascend has aclnn kernels for both overloads,
# so the route back is free. The rest of the family is unaffected: bmm, bmm.out
# and addmm run on FlagGems and match a float64 CPU reference to 1.6e-7
# relative, three orders tighter than the aclnn path's 1.7e-4, so the gap is
# exactly these two overloads. matmul is not in the FlagGems coverage this conf
# is generated from and already routes to `ascend`. Not yet filed upstream.
#
# musa: index_add and randn_like/randn were the first entries in this set (#275,
# 2026-09-15), recorded as "index_add returns all zeros instead of accumulating"
# and "randn crashes unpacking generator state". Neither signature reproduces.
# Re-measured on MTT S5000 with FlagGems 4d9c34775 + flagtree 0.6.2a3+mthreads3.6,
# and re-checked on the CI pin e7b4a865f (an ancestor of it, see below):
#
#   - index_add / index_add_: bit-exact against the CPU for duplicate indices,
#     dim 0 and dim 1, float64, and alpha != 1, in both the in-place and
#     out-of-place spellings. A new flag_gems code-cache entry appears per run, so
#     the mthreads kernel compiles and runs on device rather than the op passing
#     on the CPU. Neither op has a mudnn kernel, so before this they could not be
#     registered at all -- they routed to `none`, and the `FLAGOS_OP_*` override
#     could not reach them either, which is why the old "returns all zeros"
#     signature was measured by hand. It has the shape of a cross-stream read: the
#     in-place wrapper is a FlagGems kernel writing a clone followed by a copy
#     into `self`, and #275 fixed a split default stream in the same change, so
#     the copy could read the clone before the kernel filling it had retired.
#
#   - randn / randn_like: finite, seed-reproducible, and seed-sensitive over 65536
#     samples, in float32/float16/bfloat16, and randn_like inherits shape and dtype
#     from its input. FlagGems runs off the Philox bridge that torch_fl installs
#     for MUSA, which is exactly the generator state the old crash was about.
#
# Both now route to FlagGems; the promotion is recorded in
# docs/reference/operator-support.md.
#
# musa: add/sub/div.Tensor hit a bf16 promotion mismatch that only the .Tensor
# overloads reach. A Python-float operand arrives as a float64 0-dim tensor --
# ATen's wrapped-number boxing, which is how `bf16_tensor + 0.5` still yields
# bf16 out of core TensorIterator. FlagGems' pointwise promotion does not honour
# is_wrapped_number, so it promotes the sum to fp64 and the store then casts
# double -> bfloat16. mthreads' LLVM lowering declares
# `llvm.musa.float2bfloat16(float)` with no double overload, so triton dies with
# "intrinsic call operand #0 has type double but ... expects float" and the op
# never runs. bf16 is the only dtype affected (fp16/fp32 have a double down-cast)
# and .Tensor the only overload (the .Scalar forms pass the number as a scalar
# argument and are unaffected), which is why mul.Tensor -- already native -- and
# add/sub/div.Scalar keep working. Re-measured on MTT S5000 with FlagGems
# 4d9c34775 + flagtree 0.6.2a3+mthreads3.6.
#
# musa: the four in-place arithmetic ops. add_/sub_/div_ are the same bf16
# wrapped-number mismatch as their .Tensor counterparts above, reached through a
# different entry point. ATen boxes `add_.Scalar` (and `_foreach_add_`, and so
# AdamW's foreach path) onto `add_.Tensor`, so routing only the out-of-place op
# leaves every in-place caller on the failing FlagGems kernel -- measured:
# `torch._foreach_add_(x, 1.0)` on bf16 logs `[flagos dispatch] add_.Tensor ->
# flagos_python` and then dies in triton with "intrinsic call operand #0 has type
# double but llvm.musa.float2bfloat16 expects float". mudnn has real in-place
# Binary kernels, so the route back is free.
#
# `mul_.Tensor` is in the same group but for a different, and larger, defect:
# flag_gems' generic `mul` is written for its own device name, so
# `mul_broadcast_func` (flag_gems/ops/mul.py, the `device.type != _DEVICE_NAME`
# guard at line 587) takes that branch for every FlagGems tensor on this backend
# (`'flagos' != 'musa'`) and delegates -- to `aten.mul.out.redispatch` when it was
# given an out=, which has no kernel at that dispatch key, so the op dies with
# "no fallback function is registered for schema aten::mul.out" for *every* dtype
# and both operand shapes. Unlike the wrapped-number case this is not bf16-only:
# measured failing for f32 tensor-tensor as well, while `torch._foreach_mul_(x,
# 1.0)` on bf16 passes because it never reaches this function. `aten.mul.out`
# itself is usable on MUSA when called directly, so the missing piece is only the
# redispatch target. Re-measured on MTT S5000 with FlagGems 4d9c34775 + flagtree
# 0.6.2a3+mthreads3.6.
#
# musa: _conj is here for a contract reason, not a compile one. `conj` in ATen is
# a lazy view -- it sets the Conjugate bit and leaves the storage alone, which
# tests/integration/test_math_bits_contract.py pins. flag_gems' `_conj` is a real
# kernel that materializes the conjugated values, so registering it on
# PrivateUse1 replaces the lazy view with an eager copy and `is_conj()` comes
# back False. mudnn has no Conjugate-bit path either, so the correct route is
# `none`: leave the op unregistered and let ATen's composite implement it. Being
# in this set does both -- it is dropped from the FlagGems registration list and,
# having no native kernel, route() falls through to `none`.
#
# musa: sort is here for what flag_gems' kernel does *inside* itself, not for what
# it computes. Its radix path casts the cumulative histogram to uint32
# (flag_gems/ops/sort.py, `ex_cumsum_bins.to(torch.uint32)`), and mudnn's
# Unary::CAST has no UInt16/32/64 case -- MudnnSupportsDtype stops at kBool -- so
# the cast raises "MudnnCopy: unsupported dtype Long -> UInt32" before the sort
# runs at all. mudnn's own sort is a real kernel, measured here to return the
# right values *and* indices for the shape the profiler contract uses
# (torch.randn(16)), so routing the op back loses no coverage. sort.stable is the
# same kernel through the stable entry point, and argsort/msort are composites
# over sort, so this one entry fixes all four. Measured on MTT S5000 with FlagGems
# 4d9c34775 + flagtree 0.6.2a3+mthreads3.6.
#
# Re-confirmed on 2026-09-15: the failure is unchanged and still names the cast,
# not the sort -- `RuntimeError: MudnnCopy: unsupported dtype Long -> UInt32`,
# raised both from `torch.sort` when the spellings it dispatches to are routed to
# FlagGems and from `flag_gems.ops.sort_stable` called directly, in every dtype
# probed (f32, int64). The mudnn route answers correctly for all of them, which is
# why both `sort` and `sort.stable` stay in this set.
#
# musa: integer division. Two separate defects, both in the *integral* path and
# both silent -- the kernel returns a plausible answer with the wrong dtype or
# one stale element, so nothing upstream can be blamed for it (issue #266).
#
# `floor_divide` and `floor_divide_.Tensor` lose the last element's store for
# integer operands on the mthreads Triton backend: with a = [10,20,30] and
# b = [2,4,5], `a // b` returns [5,5,<garbage>] wherever numel is not a power of
# two (measured wrong at n = 3,5,6,7,9,15,17,31,33,100; right at n = 1,2,4,8,16,
# 32,64,1024). Float operands are correct at every size. mudnn's FLOORDIV is the
# same mode the op already claims, so routing it back is free.
#
# `div.Tensor_mode` / `div_.Tensor_mode` are wrong differently: FlagGems computes
# the integer `rounding_mode='floor'`/`'trunc'` forms as integer division but
# drops the same trailing element, and the absent-mode form is correct only
# because the promotion to float happens to hide it. The mudnn kernels added for
# them take the mode from aten's `rounding_mode` at run time, so one entry covers
# all three spellings. `floor_divide.Scalar` and the `div.*_mode` scalar forms are
# composites over the Tensor entries above, so they follow those routes.
#
# Measured on MTT S5000 with FlagGems 4d9c34775 + flagtree 0.6.2a3+mthreads3.6.
# 4d9c34775 is a strict superset of the CI pin e7b4a865f (it is a descendant, and
# the diff between them touches only flash_attention_backward, the ascend
# masked_scatter_backward removal and the mthreads linear, none of which is in
# this set), so every entry here also holds for the pinned revision. The two
# promotions described above were re-checked on the pin itself as well.
#
# gcu: the four matmul overloads. FlagGems' matmul kernels are autotuned with
# extra Triton compile options that the enflame backend of flagtree 0.6.1 does
# not accept -- the launch dies before any GCU code runs, on a keyword the
# compiler refuses:
#
#   mm / mm.out      KeyError: 'Keyword argument SPLIT_K was specified but
#                    unrecognised'  (flag_gems/ops/mm.py passes SPLIT_K to
#                    triton.autotune's config list)
#   bmm / bmm.out    TypeError: dynamic_func() missing 7 required positional
#                    arguments: TILE_M, TILE_N, TILE_K, GROUP_M, DIVISIBLE_M,
#                    DIVISIBLE_N, DIVISIBLE_K  (flag_gems/ops/bmm.py builds its
#                    configs as primal heuristics, which flagtree's
#                    triton.autotune does not unpack at all)
#
# These are host-side argument errors, not miscomputes: nothing is launched, so
# the failure is loud rather than silent. topsaten has real GEMM kernels for all
# four overloads (gcu_register.inc claims them), so the route back is free --
# and `addmm`/`addmm.out`, which use a third kernel in the same FlagGems module,
# were measured correct and stay on FlagGems.
#
# gcu: everything else below, measured on the S60. GCU300 rejects every 64-bit
# data type in the kernel IR -- `error: 64-bit data type not supported on
# GCU300!`, surfaced to Python as `RuntimeError: Pipeline run failed: PassManager
# execution failed` -- so any FlagGems kernel that widens an operand to i64/f64
# fails to compile. That single compiler limit accounts for most of this set, but
# not all of it: the RNG entries also miscompute silently.
#
# The pointwise overloads (add/sub/div/mul/rsqrt/clamp.Tensor) hit it through
# ATen's wrapped-number boxing, the same defect MUSA records for bf16: a Python
# scalar arrives as a float64 0-dim tensor, and pointwise_dynamic promotes with
# torch._prims_common.elementwise_dtypes, which does not honour
# is_wrapped_number. Measured: `f32 + 0.5`, `f32 + 1`, `i32 + 1`, `f32 - 1` and
# `f32 / 2` all raise, while `f32 + f32` and every tensor-tensor form compute
# correctly. It is the overload that is broken, but routing is per op, so the
# whole entry moves. The scalar forms that are already native -- add.Scalar,
# sub.Scalar, mul.Scalar -- are unaffected and stay on FlagGems, as do `clamp`
# with Python bounds, `pow.*`, `mul.Tensor` and every comparison. `mul_.Tensor`
# is the one non-64-bit entry here: it raises `NotImplementedError: There were no
# tensor arguments to this function`, the same recursive re-entry codegen_ops.py
# records for `mul.Tensor`.
#
# The factory entries widen a little differently. `arange` (all three overloads,
# float and int alike), `linspace`, `full`/`full_like` with an integer fill,
# `sum` on an integer operand and every integer-dtype `zeros`/`zeros_like`/
# `ones`/`ones_like`/`zero_` fail to compile; `scalar_tensor` compiles but returns
# float32 for an integer argument. `zeros` is also unreliable at float32: on a
# first-touch 262144-element allocation it left 21207 (shape (64,64,64)) and
# 22536 (shape (32,32,256)) elements unwritten, and was clean on every repeat.
# The float32 `ones` path measured clean in isolation but the suite still reports
# a partially written (64,64,64) result, so the family moves back together
# rather than per-dtype.
#
# The gather/loss/sort entries -- `embedding`, `floor_divide`, `index_add`,
# `nll_loss_forward`, `nll_loss_backward`, `sort`, `sort.stable` -- all fail on
# index/offset arithmetic that legalizes to a 64-bit extension (`arith.extsi` /
# `arith.extui` marked illegal); `sort` additionally segfaults the process
# outright.
#
# The RNG family is the one group that is wrong rather than merely unbuildable,
# which is why all of it moves back to the vendor's handwritten kernels
# (codegen_gcu.py HANDWRITTEN_OPS) -- `rand`, `randn`, `randperm`, `multinomial`,
# `uniform_`, `exponential_`, `native_dropout`, `binomial`, `_standard_gamma`
# and `_sample_dirichlet` raise the 64-bit error, and the rest return the wrong
# distribution: test_rng_dispatch.py measures `torch.rand(2000)` with mean 0.143
# against a 0.5 contract, `randn` with mean 0.378, `exponential_(1.0)` with mean
# 0.501, `uniform_(-1, 1)` with mean 2.497, a dropout keep-rate off by 5x, and
# `bernoulli` returning values outside {0, 1}.
#
# Several entries have no topsaten kernel claimed by codegen_gcu.py (`arange`,
# `linspace`, `full`, `full_like`, `scalar_tensor`, `zero_`, `constant_pad_nd`,
# `embedding`, `nll_loss_*`, `floor_divide*`, `index_add*`, `sort*`). There the
# gap drops them from both the conf's FlagGems route and the generated
# registration, so route() falls through to `none` and the call reaches ATen's
# cpu_fallback. That is the route these ops already took before GCU had a
# FlagGems path, so nothing regresses; it is not a vendor kernel because no
# vendor kernel is claimed for them yet.
#
# Measured on S60 with FlagGems master 3c6f7537d2d5d3aa680c55bbee5c70f2100c5b85
# (5.4.0.dev0) + flagtree 0.6.1+enflame3.6.
NATIVE_TRITON_GAPS = {
    "ascend": {
        "eq.Scalar",
        "eq.Tensor",
        "exponential_",
        "ge.Scalar",
        "ge.Tensor",
        "gt.Scalar",
        "gt.Tensor",
        "le.Scalar",
        "le.Tensor",
        "lt.Scalar",
        "lt.Tensor",
        "mm",
        "mm.out",
        "mul_.Tensor",
        "native_dropout",
        "native_dropout_backward",
        "ne.Scalar",
        "ne.Tensor",
        "rand",
        "rand_like",
        "randperm",
        "sort",
        "sort.stable",
    },
    "gcu": {
        # Pointwise overloads broken by ATen's float64 wrapped-number boxing.
        "add.Tensor",
        "add_.Tensor",
        "clamp.Tensor",
        "div.Scalar",
        "div.Scalar_mode",
        "div.Tensor",
        "div.Tensor_mode",
        "div_.Scalar",
        "div_.Scalar_mode",
        "div_.Tensor",
        "div_.Tensor_mode",
        "mul_.Tensor",
        "rsqrt",
        "rsqrt_",
        "sub.Tensor",
        "sub_.Tensor",
        # Factory / creation ops that widen to a 64-bit element type.
        "arange",
        "arange.start",
        "arange.start_step",
        "constant_pad_nd",
        "full",
        "full_like",
        "linspace",
        "ones",
        "ones_like",
        "scalar_tensor",
        "sum",
        "zero_",
        "zeros",
        "zeros_like",
        # Index/offset arithmetic that legalizes to a 64-bit extension.
        "embedding",
        "floor_divide",
        "floor_divide_.Tensor",
        "index_add",
        "index_add_",
        "nll_loss_backward",
        "nll_loss_forward",
        "sort",
        "sort.stable",
        # Matmul: flagtree refuses the autotune kwargs before any GCU code runs.
        "bmm",
        "bmm.out",
        "mm",
        "mm.out",
        # RNG: the vendor's handwritten kernels, for the wrong distributions.
        "bernoulli_.float",
        "exponential_",
        "multinomial",
        "native_dropout",
        "native_dropout_backward",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "uniform_",
        # The int64 element-type family. These kernels are correct for f32/i32/bool
        # and fail only for an integral operand, because GCU300 cannot represent
        # any 64-bit type in the kernel IR at all -- the pointer parameter is the
        # rejected type, so it is the *operand* that has to be avoided, not the
        # op. Routing is per op, so an op that any real caller hands an int64
        # tensor has to move as a whole. Measured on the S60 (see the note below
        # the set): every entry below computes correctly on i32/f32/bool and
        # raises `Pipeline run failed` on i64.
        #
        # Comparisons have topsaten kernels for every dtype, so gapping them
        # routes back to `gcu` with its TopsatenSupportsDtype CPU round-trip for
        # the int64 case; the bitwise/fill/mask group has no topsaten kernel, so
        # it lands on `none` and the call reaches cpu_fallback -- the same route
        # these ops already took before GCU had a FlagGems path.
        "eq.Scalar",
        "eq.Tensor",
        "ne.Scalar",
        "ne.Tensor",
        "lt.Scalar",
        "lt.Tensor",
        "le.Scalar",
        "le.Tensor",
        "gt.Scalar",
        "gt.Tensor",
        "ge.Scalar",
        "ge.Tensor",
        "bitwise_and.Scalar",
        "bitwise_and.Scalar_Tensor",
        "bitwise_and.Tensor",
        "bitwise_and_.Scalar",
        "bitwise_and_.Tensor",
        "bitwise_not",
        "bitwise_not_",
        "bitwise_or.Scalar",
        "bitwise_or.Scalar_Tensor",
        "bitwise_or.Tensor",
        "bitwise_or_.Scalar",
        "bitwise_or_.Tensor",
        "bitwise_xor.Scalar",
        "bitwise_xor.Scalar_Tensor",
        "bitwise_xor.Tensor",
        "bitwise_xor_.Scalar",
        "bitwise_xor_.Tensor",
        "fill.Scalar",
        "fill.Scalar_out",
        "fill.Tensor",
        "fill.Tensor_out",
        "fill_.Scalar",
        "fill_.Tensor",
        "masked_fill.Scalar",
        "masked_fill.Tensor",
        "masked_fill_.Scalar",
        "masked_fill_.Tensor",
        "masked_select",
        "where.self",
        "all",
        "any",
        # _conj is here for the same contract reason as musa's entry: flag_gems'
        # `_conj` is a real kernel that materializes the conjugation, where ATen
        # keeps it as a lazy view. GCU cannot materialize a Conjugate bit at all
        # (`view_as_real` has no PrivateUse1 kernel, so the FlagGems kernel warns
        # and aborts on a complex operand), so the op is unregistered and ATen's
        # composite runs instead -- which is what test_math_bits_contract.py
        # probes for before deciding whether its contract applies.
        "_conj",
        # The scan family. flag_gems lowers every one of these to a Triton kernel
        # whose accumulator is a 64-bit type, so the GCU300 front end rejects the
        # whole family. Measured on the S60 with 4-element operands:
        #
        #   cumsum      i64 FAIL  f32 PASS      cumsum.out      i64 FAIL  f32 PASS
        #   cumprod     i64 FAIL  f32 PASS      cumprod_        i64 FAIL  f32 PASS
        #   logcumsumexp i64 FAIL f32 PASS      logcumsumexp.out i64 FAIL f32 PASS
        #   cummax      i64 FAIL  f32 FAIL      cummin          i64 FAIL  f32 FAIL
        #
        # cummax/cummin fail for float32 as well -- their kernel carries the index
        # tensor as int64 unconditionally, so they are unusable on GCU at any
        # dtype. That is what test_diff_then_cumsum hits: `diff(...).cumsum(-1)` on
        # an int64 index tensor raises `flag_gems/ops/cumsum.py:343: 64-bit data
        # type not supported on GCU300!`.
        #
        # No topsaten kernel exists for any of these, so gapping lands on `none`
        # and cpu_fallback -- the route they took before GCU had a FlagGems path.
        # The .out forms of cummax/cummin/cumprod and cumsum_ route to `none`
        # already and need no entry.
        "cumsum",
        "cumsum.out",
        "cumprod",
        "cumprod_",
        "cummax",
        "cummin",
        "logcumsumexp",
        "logcumsumexp.out",
        # Everything below comes from the full sweep rather than from targeted
        # probes: tests/manual/flaggems_overload_survey.py --conf
        # torch_fl/configs/backends_gcu.conf run on the S60 at FlagGems master
        # 3c6f7537d2d5d3aa680c55bbee5c70f2100c5b85 (5.4.0.dev0) + flagtree
        # 0.6.1+enflame3.6, seven profiles per overload (2d/4d/1d float32, 2d
        # float16, 2d int64, 2d bool, 2d strided), all 374 FlagGems routes
        # measured. Per-op evidence, repro and the unchanged remainder are in
        # docs/vendors/gcu/flaggems-test-results.md.
        #
        # Group A -- FlagGems is wrong at float16/float32, the dtypes every GCU
        # caller uses, so the route is unusable and the op goes back to topsaten
        # (or to `none` and cpu_fallback where topsaten has no kernel either).
        # Three signatures account for most of it: `Pipeline run failed:
        # PassManager execution failed`, the GCU300 front end rejecting the
        # 64-bit type the kernel carries internally (reductions, scans, index
        # ops); `UNREACHABLE executed at
        # .../flagtree-enflame3.6-gcu400/...` after `unsupported extern
        # elementwise: __nv_asinf`, a hard compiler abort on the libm externs
        # flag_gems emits for the inverse-trig and special-function families;
        # and `PROCESS-DEATH` for the four ops whose launch kills the driver
        # thread outright. The rest are measured wrong answers (`elu` off by
        # 0.89, `_softmax_backward_data` returning int8, `histc`, `kthvalue`,
        # `sum.out` losing its dims) or a flag_gems wrapper that refuses an
        # input ATen accepts.
        "_adaptive_avg_pool2d",
        "_batch_norm_no_update",
        "_euclidean_dist",
        "_linalg_eigvals",
        "_log_softmax_backward_data",
        "_pdist_backward",
        "_softmax_backward_data",
        "_unique2",
        "_weight_norm_interface",
        "_weight_norm_interface_backward",
        "addmm.out",
        "addmm_",
        "addr",
        "aminmax",
        "any.dims",
        "argmax",
        "argmin",
        "asin",
        "asin_",
        "bucketize.Tensor",
        "cosh.out",
        "count_nonzero",
        "dequantize.self",
        "elu",
        "elu_",
        "elu_backward",
        "erfinv",
        "erfinv_",
        "histc",
        "index_copy",
        "index_copy_",
        "isin.Tensor_Scalar",
        "isin.Tensor_Tensor",
        "kthvalue",
        "lgamma",
        "lgamma_",
        "max.dim",
        "median",
        "median.dim",
        "min.dim",
        "mode",
        "mse_loss",
        "nanmedian",
        "nanmedian.dim",
        "native_batch_norm",
        "native_layer_norm",
        "nextafter",
        "nextafter_",
        "nonzero",
        "norm.Scalar",
        "norm.ScalarOpt_dim",
        "range",
        "renorm",
        "repeat_interleave.Tensor",
        "scatter.src",
        "scatter_.src",
        "scatter_add_",
        "silu_backward",
        "soft_margin_loss",
        "soft_margin_loss_backward",
        "special_chebyshev_polynomial_u",
        "special_chebyshev_polynomial_v",
        "special_chebyshev_polynomial_w",
        "special_hermite_polynomial_h",
        "special_modified_bessel_k0",
        "special_modified_bessel_k0.out",
        "special_shifted_chebyshev_polynomial_u",
        "special_shifted_chebyshev_polynomial_w",
        "sum.out",
        "topk",
        "tril.out",
        "tril_",
        "triu",
        "triu_",
        "unfold_backward",
        "unique_consecutive",
        "unique_dim",
        "upsample_bicubic2d",
        "var.correction",
        "var_mean.correction",
        "vdot",
        # Group B -- correct at float16/float32 and at bool, and broken only for
        # an int64 operand (the 64-bit kernel-type rejection again; a few also
        # return the input dtype where ATen promotes, which only an integral or
        # bool operand reaches). These are gapped because topsaten has a kernel
        # for every one of them: gapping restores the vendor kernel's
        # TopsatenSupportsDtype CPU round-trip for int64 and costs nothing at
        # float16/float32, which is a strictly better outcome than a route that
        # raises `Pipeline run failed` for an operand type it accepts.
        #
        # The rest of the group -- 76 overloads that are equally broken for
        # int64 and have no topsaten kernel to fall back to -- deliberately stay
        # on FlagGems and are listed in docs/vendors/gcu/flaggems-test-results.md
        # instead. Gapping those would demote their float16/float32 path to
        # cpu_fallback, which is a far larger regression than the int64 raise it
        # would avoid: `threshold_backward`, `relu_`, `clamp_min`, `clamp_max`,
        # `max`, `min`, `nan_to_num` and `masked_scatter` are all in that set,
        # and none of them is handed an int64 tensor by a float model.
        "abs",
        "acos",
        "addmm",
        "amax",
        "amin",
        "atan",
        "ceil",
        "cos",
        "cosh",
        "erf",
        "exp",
        "expm1",
        "flip",
        "floor",
        "fmod.Scalar",
        "log",
        "log10",
        "log1p",
        "log2",
        "logical_and",
        "logical_or",
        "mse_loss_backward",
        "neg",
        "pow.Tensor_Scalar",
        "pow.Tensor_Tensor",
        "reciprocal",
        "relu",
        "remainder.Scalar",
        "sigmoid",
        "sin",
        "sinh",
        "sqrt",
        "sum.dim_IntList",
        "tanh",
        "tril",
        "trunc",
    },
    "musa": {
        "_conj",
        "add.Tensor",
        "add_.Tensor",
        "div.Tensor",
        "div.Tensor_mode",
        "div_.Tensor",
        "div_.Tensor_mode",
        "floor_divide",
        "floor_divide_.Tensor",
        "mul_.Tensor",
        "sort",
        "sort.stable",
        "sub.Tensor",
        "sub_.Tensor",
    },
}

# Vendors whose native kernel outranks FlagGems for any op that has both, until a
# per-op hardware sweep says otherwise. This inverts the usual priority, so it is
# deliberately narrow.
#
# Empty for now: all vendors use FlagGems-first routing where both backends exist.
# Move a vendor here if its Triton stack is immature and unverified routes cause
# regressions. Failures for specific ops go in NATIVE_TRITON_GAPS above with their
# diagnosis; this set is for blanket "prefer native until measured" policies.
NATIVE_KERNEL_PREFERRED = set()

# Platforms whose build can compile the TileOPs slot (Backend::kTileOps). The
# shims are Triton kernels needing an SM90 device plus the `tileops` package, and
# setup.py force-sets TILEOPS_KERNEL=OFF for every ACCELERATOR != "cuda" -- so on
# any other platform the slot is empty, Dispatcher::GetFn degrades kTileOps to
# cuda_fn_, and on a native-kernel vendor that is empty too. Naming `tileops` in
# those confs would therefore route a real op at nothing, which is the "backend
# not registered" class of failure this full-coverage rework exists to remove.
# No shipped conf is generated for plain cuda (codegen_ops.py writes
# backends_cuda.conf), so this is currently empty and the key reaches confs only
# through the `# tileops` annotation that FLAGOS_USE_TILEOPS reads.
TILEOPS_PLATFORMS = set()

# The 17 FlagGems C++ ops verified on MetaX hardware (of the 18 in the shared C++
# set). `mm` is the omission and is deliberate: gems' `mm` passes a SPLIT_K kwarg
# triton-metax rejects, so it stays on the boxing kernel. Recorded here because
# the conf that used to hold this measurement no longer exists separately; see
# boxing_cpp_ops(). test_metax_conf_keeps_mm_boxed pins the exception.
METAX_CPP_MEASURED = {
    "_softmax",
    "_softmax_backward_data",
    "addmm",
    "addmm.out",
    "argmax",
    "bmm",
    "bmm.out",
    "embedding",
    "max",
    "max.dim",
    "nonzero",
    "sort",
    "sort.stable",
    "sum",
    "sum.dim_IntList",
    "topk",
    "zeros",
}

LICENSE = """\
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


def parse_conf(path: Path) -> dict:
    """Parse `op = backend` lines. Ignores comments, blanks and `include`."""
    table = {}
    if not path.exists():
        return table
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or "=" not in line:
            continue
        op, backend = line.split("=", 1)
        op, backend = op.strip(), backend.strip()
        if op and backend:
            table[op] = backend
    return table


def flaggems_python_ops(conf_dir: Path = None) -> set:
    """Ops FlagGems covers through its Python/Triton path (Backend::kFlagGems).

    `conf_dir` is accepted and ignored: this used to parse the `flagos_python`
    entries out of backends_flaggems.conf. That file existed only to carry this
    set, so it is now data (see backend_coverage.py) and the argument is kept so
    callers do not all have to change at once.
    """
    return set(FLAGGEMS_PYTHON_OPS)


def flaggems_cpp_ops(conf_dir: Path = None) -> set:
    """Ops FlagGems covers through its C++ runtime (liboperators.so, kFlagGemsCpp).

    Same story as flaggems_python_ops(): formerly the `flagos` entries of
    backends_flaggems_cpp.conf, now data.
    """
    return set(FLAGGEMS_CPP_OPS)


def tileops_ops(conf_dir: Path = None) -> set:
    """Ops with a TileOps shim behind them (Backend::kTileOps).

    Formerly the `tileops` entries of backends_tileops.conf, which was otherwise
    a copy of backends_cuda.conf. scripts/codegen/codegen_tileops.py generates both the
    shims and this set.
    """
    return set(TILEOPS_OPS)


def registered_impls(inc_path: Path) -> set:
    """Ops an `.inc` file claims on PrivateUse1, read from its `m.impl("op", ...)`.

    These files are codegen output included by csrc/aten/register.cc, so this is
    the same list the compiler sees -- the authoritative answer to "is this op
    registered on this platform", which is what decides whether `none` boxes to
    CPU or raises.
    """
    if not inc_path.exists():
        return set()
    return set(re.findall(r'm\.impl\(\s*"([^"]+)"', inc_path.read_text()))


def vendor_native_ops(vendor: str) -> set:
    """Ops with a vendor-native kernel, read from that vendor's own `.inc`.

    Earlier this was recovered from the conf being rewritten, which needed a
    trailing `# <vendor>` annotation to survive the round trip. Reading the
    generated C++ instead removes that dependency: the kernel set cannot drift
    from what is compiled, and a conf edit cannot invent or lose a kernel.

    The annotation is still *emitted* -- it is what tells a reader (and
    ALL_USE_VENDOR) that a kernel exists behind an op FlagGems currently wins.
    """
    native_inc, _ = VENDORS[vendor]
    return registered_impls(CSRC_DIR / native_inc) | EXTRA_NATIVE.get(vendor, set())


def vendor_registered_ops(vendor: str) -> set:
    """Every op this vendor claims on PrivateUse1, native kernel or FlagGems.

    MUSA registers a second file (musa_flaggems_register.inc) whose wrappers go
    through the conf-driven dispatcher to the FlagGems Python slot, so those ops
    are registered without being native kernels. Anything outside this set is
    unregistered and therefore genuinely reaches cpu_fallback.
    """
    native_inc, extra_incs = VENDORS[vendor]
    ops = registered_impls(CSRC_DIR / native_inc)
    for inc in extra_incs:
        ops |= registered_impls(CSRC_DIR / inc)
    return ops | EXTRA_NATIVE.get(vendor, set())


def _platform_from_conf(filename: str) -> str:
    """backends_ppu.conf -> ppu. Empty for any name that does not follow it."""
    if filename.startswith("backends_") and filename.endswith(".conf"):
        return filename[len("backends_") : -len(".conf")]
    return ""


def boxing_triton_gaps(conf_dir: Path, filename: str, fg_py: set, fg_cpp: set) -> set:
    """Ops a boxing platform must keep on CUDA even though FlagGems covers them.

    Two sources, in this order:

    * BOXING_TRITON_GAPS -- platforms whose gap set is a measurement that cannot
      be recovered from the conf. PPU is one: it is flaggems-first by policy, so
      the ops it keeps on `cuda` are exceptions to that policy rather than the
      residue of one, and reading them back out of the file would re-derive an
      empty set from the first flaggems-first conf. Intersected with the shared
      coverage so an op dropped from FlagGems cannot linger here.

    * Diffing the platform's existing conf against the shared FlagGems sources
      (MetaX, DCU): an op that FlagGems covers but this conf routes to `cuda` is
      a deliberate per-platform triton gap (MetaX Xnack/ATU faults, DCU hcu
      VMFault and the missing div_rn lowering -- diagnosed per op in
      codegen_ops.py). This branch reads the file it later rewrites, so the gap
      set has to survive the round trip; it does, because a gap stays spelled
      `cuda` in the generated conf.
    """
    measured = BOXING_TRITON_GAPS.get(_platform_from_conf(filename))
    if measured is not None:
        return measured & (fg_py | fg_cpp)
    table = parse_conf(conf_dir / filename)
    covered = fg_py | fg_cpp
    return {op for op, backend in table.items() if backend == "cuda" and op in covered}


def boxing_cpp_ops(conf_dir: Path, filename: str, fg_cpp: set) -> set:
    """Which FlagGems C++ ops a boxing platform actually routes to that path.

    Per-platform measured, not shared: MetaX verified 17 of the 18 ops on-device
    and keeps `mm` on the boxing kernel, because gems' `mm` passes a SPLIT_K
    kwarg triton-metax rejects. That exception is the reason this set is spelled
    out here rather than derived as `fg_cpp` -- re-deriving all 18 would silently
    promote `mm` to a route that raises on the first call.

    Stated as a literal because it can no longer be recovered by reading the file
    this rewrites. One platform means one conf now, so the measurement's former
    home (backends_metax_flaggems_cpp.conf) is gone, and a conf that carries both
    the C++ and non-C++ routings cannot be diffed back into its two inputs.
    Intersected with `fg_cpp` so an op dropped from the FlagGems C++ set cannot
    linger here.
    """
    return METAX_CPP_MEASURED & fg_cpp if filename == "backends_metax.conf" else set()


def route_boxing(op: str, fg_cpp: set, fg_py: set, gaps: set, tileops: set = ()) -> str:
    """Pick the backend key for one op on a CUDA-boxing platform.

    Same priority as a native vendor, with `cuda` in the `<vendor>` slot. Boxing
    reuses the CUDA kernels wholesale, so every op has an impl and `none` never
    appears -- the difference from a native-kernel vendor is coverage, not shape.

    A triton gap outranks every accelerated key, `tileops` included: the gap is a
    measured statement that this platform's triton backend cannot run the kernel,
    and the TileOps shims are Triton-generated too.
    """
    if op in gaps:
        return BOXING_FALLBACK
    if op in fg_cpp:
        return "flaggems_cpp"
    if op in fg_py:
        return "flaggems"
    if op in tileops:
        return "tileops"
    return BOXING_FALLBACK


def route(
    op: str,
    vendor: str,
    fg_cpp: set,
    fg_py: set,
    native: set,
    registered: set,
    tileops: set = (),
) -> str:
    """Pick the backend key for one op. FlagGems first, vendor as fallback.

    `fg_py` / `fg_cpp` / `tileops` are coverage sets measured on CUDA, so they
    are a *ceiling*, not this platform's routing set. An op only routes to an
    accelerated key if the platform also registers it on PrivateUse1: those
    wrappers are reached through the op's PrivateUse1 registration, so routing an
    unregistered op to `flaggems` names a kernel no call can arrive at, and the
    op silently boxes to CPU while the conf claims coverage. Gating on
    `registered` is what keeps the file's counts true.

    `tileops` sits below both FlagGems paths and above the vendor kernel. Below
    FlagGems because FlagGems is the broader, longer-measured library and its
    routes are the ones the per-platform gap sets are calibrated against; above
    the vendor kernel because the TileOps shims are the newer path being brought
    up, so where both exist the shim is what a run should exercise. The vendor
    kernel stays reachable through the `# <vendor>` annotation and ALL_USE_VENDOR.

    An op the vendor implements but FlagGems or TileOps wins gets a trailing
    `# <vendor>` annotation. That is what tells a reader a kernel exists behind
    the winning route, and what makes ALL_USE_VENDOR able to move the op.
    """
    if op not in registered:
        return "none"
    if op in fg_cpp:
        return f"flaggems_cpp  # {vendor}" if op in native else "flaggems_cpp"
    if op in fg_py:
        return f"flaggems  # {vendor}" if op in native else "flaggems"
    if op in tileops:
        return f"tileops  # {vendor}" if op in native else "tileops"
    if op in native:
        return vendor
    return "none"


def render(platform: str, vendor: str, routes: dict, boxing: bool = False) -> str:
    """Build the conf text for one platform.

    `platform` names the conf (backends_<platform>.conf); `vendor` is the key in
    the third slot -- a native kernel name, or "cuda" for a boxing platform.
    """
    counts = {}
    for value in routes.values():
        backend = value.split("#", 1)[0].strip()
        counts[backend] = counts.get(backend, 0) + 1
    total = len(routes)
    covered = total - counts.get("none", 0)

    if boxing:
        vendor_desc = "CUDA boxing kernel (this platform is CUDA-compatible)"
    else:
        vendor_desc = "vendor-native kernel"

    lines = [LICENSE.rstrip("\n"), ""]
    lines += [
        f"# flagos op backend config for {platform} -- AUTO-GENERATED.",
        "# Regenerated by scripts/codegen/gen_vendor_confs.py -- do not edit manually.",
        "#",
        "# Full-coverage config: every op torch_fl can route is listed exactly",
        "# once, so operator support for this platform is countable from this",
        "# file alone.",
        "#",
        "# Values:",
        "#   flaggems_cpp  FlagGems C++ runtime (liboperators.so)",
        "#   flaggems      FlagGems Python/Triton path",
        "#   tileops       TileOPs Triton shims (needs TILEOPS_KERNEL=ON + SM90)",
        f"#   {vendor:<13} {vendor_desc}",
    ]
    if boxing:
        lines += [
            "#",
            f"# Routing priority: flaggems_cpp > flaggems > tileops > {vendor}.",
            "# Boxing reuses the CUDA kernels for every op, so nothing routes to",
            "# 'none' here.",
            "#",
        ]
    else:
        lines += [
            "#   none          no accelerated impl here -> reaches cpu_fallback",
            "#",
            f"# Routing priority: flaggems_cpp > flaggems > tileops > {vendor} > none.",
            "#",
        ]
    lines += [
        f"# Coverage: {covered}/{total} ops accelerated "
        f"({100.0 * covered / total:.1f}%)",
    ]
    if platform in BOXING_GAP_NOTES:
        lines += ["#"] + [f"# Note: {line}" for line in BOXING_GAP_NOTES[platform]]
    for backend in ("flaggems_cpp", "flaggems", "tileops", vendor, "none"):
        if backend in counts:
            lines.append(f"#   {backend:<13} {counts[backend]:>5}")
    lines += [
        "#",
        "# Override one op at runtime with FLAGOS_OP_<name>=<backend> (dots in",
        "# the op name become double underscores: FLAGOS_OP_mm__out=flaggems).",
        "# Collapse the whole table onto one backend for A/B measurement with",
        "# ALL_USE_FLAGGEMS=1 or ALL_USE_VENDOR=1 (mutually exclusive). An op",
        "# only moves if that backend implements it -- known from the routed",
        "# value plus its `# <backend>` annotation. The rest are reported on",
        "# stderr and stay as configured, so ALL_USE_VENDOR is partial by",
        "# nature: a vendor implements far fewer ops than FlagGems.",
        "",
    ]
    for op in sorted(routes):
        lines.append(f"{op} = {routes[op]}")
    return "\n".join(lines) + "\n"


def build_all(conf_dir: Path) -> dict:
    """Compute {platform: (conf_text, routes, vendor_key)} for every platform.

    Covers both native-kernel vendors and CUDA-boxing platforms: same five keys,
    same priority, differing only in what sits in the vendor slot.
    """
    all_ops = registered_impls(GENERATED_REGISTER_INC)
    if not all_ops:
        raise SystemExit(
            f"error: no ops found in {GENERATED_REGISTER_INC}; "
            "run scripts/codegen/codegen_ops.py first"
        )

    fg_py = flaggems_python_ops(conf_dir)
    fg_cpp = flaggems_cpp_ops(conf_dir)
    tileops = tileops_ops(conf_dir)

    natives = {vendor: vendor_native_ops(vendor) for vendor in VENDORS}
    registered = {vendor: vendor_registered_ops(vendor) for vendor in VENDORS}

    # A vendor kernel can exist for an op with no entry in the generated op list
    # (Ascend's fused matmul, claimed straight from register.cc). Dropping
    # it would silently un-route a working kernel -- and for matmul it would make
    # the aclnn path unreachable, since WrapperMatmul consults this conf. Widen
    # the list instead; the union keeps every platform's table over the same op
    # set, which is what makes coverage comparable across files.
    extra = set().union(*natives.values()) - all_ops if natives else set()
    if extra:
        print(
            f"note: {len(extra)} vendor-only op(s) added to the op list: "
            f"{', '.join(sorted(extra))}",
            file=sys.stderr,
        )
        all_ops |= extra

    out = {}
    for vendor in VENDORS:
        # A vendor build is FLAGGEMS_KERNEL=OFF, so the flaggems_cpp slot has no
        # kernel -- withhold the key and let those ops take the Python path.
        cpp_here = fg_cpp if vendor in FLAGGEMS_CPP_PLATFORMS else set()
        py_here = fg_py if vendor in FLAGGEMS_PYTHON_PLATFORMS else set()
        py_here = py_here - NATIVE_TRITON_GAPS.get(vendor, set())
        # Native kernel wins wherever both exist; FlagGems keeps the ops the
        # vendor has no kernel for. See NATIVE_KERNEL_PREFERRED.
        if vendor in NATIVE_KERNEL_PREFERRED:
            py_here = py_here - natives[vendor]
        tileops_here = tileops if vendor in TILEOPS_PLATFORMS else set()
        routes = {
            op: route(
                op,
                vendor,
                cpp_here,
                py_here,
                natives[vendor],
                registered[vendor],
                tileops_here,
            )
            for op in sorted(all_ops)
        }
        out[vendor] = (render(vendor, vendor, routes), routes, vendor)

    for platform, sparse_conf in BOXING_PLATFORMS.items():
        gaps = boxing_triton_gaps(conf_dir, sparse_conf, fg_py, fg_cpp)
        # Only the C++-runtime conf routes to flaggems_cpp; elsewhere the slot is
        # unused and every C++ op is reached through the Python path instead.
        cpp_here = (
            boxing_cpp_ops(conf_dir, sparse_conf, fg_cpp)
            if platform in FLAGGEMS_CPP_PLATFORMS
            else set()
        )
        tileops_here = tileops if platform in TILEOPS_PLATFORMS else set()
        routes = {
            op: route_boxing(op, cpp_here, fg_py, gaps, tileops_here)
            for op in sorted(all_ops)
        }
        out[platform] = (
            render(platform, BOXING_FALLBACK, routes, boxing=True),
            routes,
            BOXING_FALLBACK,
        )
    return out


def print_stats(built: dict) -> None:
    platforms = sorted(built)
    total = len(next(iter(built.values()))[1])
    print(f"op list: {total} ops\n")
    print(
        "'native routed' counts ops that end up on the vendor kernel; ops the\n"
        "vendor also implements but that FlagGems covers are routed to FlagGems\n"
        "by priority and counted in 'kernels' only.\n"
    )
    header = (
        f"{'platform':<20} {'flaggems_cpp':>12} {'flaggems':>9} "
        f"{'vendor routed':>14} {'kernels':>8} {'none':>6} {'covered':>13}"
    )
    print(header)
    print("-" * len(header))
    for platform in platforms:
        _, routes, vendor = built[platform]
        counts = {}
        kernels = 0
        for value in routes.values():
            backend, _, annotation = value.partition("#")
            backend = backend.strip()
            counts[backend] = counts.get(backend, 0) + 1
            if backend == vendor or annotation.strip() == vendor:
                kernels += 1
        none = counts.get("none", 0)
        covered = total - none
        print(
            f"{platform:<20} {counts.get('flaggems_cpp', 0):>12} "
            f"{counts.get('flaggems', 0):>9} {counts.get(vendor, 0):>14} "
            f"{kernels:>8} {none:>6} "
            f"{covered:>6} ({100.0 * covered / total:>2.0f}%)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the confs on disk match what would be generated",
    )
    ap.add_argument(
        "--stats", action="store_true", help="print a coverage table and exit"
    )
    args = ap.parse_args()

    built = build_all(CONF_DIR)

    if args.stats:
        print_stats(built)
        return 0

    stale = []
    for platform, (text, _, _) in sorted(built.items()):
        path = CONF_DIR / f"backends_{platform}.conf"
        current = path.read_text() if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(path.name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    if args.check:
        if stale:
            print(
                "error: stale vendor conf(s): "
                + ", ".join(stale)
                + "\nrun: python3 scripts/codegen/gen_vendor_confs.py",
                file=sys.stderr,
            )
            return 1
        print("all vendor confs up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
