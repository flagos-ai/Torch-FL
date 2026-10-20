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

`flaggems_cpp` is only emitted for the confs a FLAGOS_BUILD_FLAGGEMS=ON build selects
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
#   musa      musa_register.inc + musa_flaggems_register.inc  517
#   gcu       gcu_register.inc + gcu_flaggems_register.inc     401
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
    # Kunlunxin has no FlagGems .inc to fold in yet: the vendor torch shim on
    # this SDK hardcodes the FlagGems runtime device name to "cuda" while the
    # tensors dispatch on PrivateUse1, so flag_gems' own gate always falls
    # through and every FlagGems Python route dies at its first op. Until that
    # is fixed (or the SDK ships a PrivateUse1-named runtime), kunlunxin has a
    # single native .inc and no extra registration file.
    "kunlunxin": ("backends/kunlunxin/generated/kunlunxin_register.inc", ()),
    # Apple Silicon (Metal). Like kunlunxin this is a native backend with a
    # single registration .inc and no FlagGems file to fold in: no Metal
    # FlagTree/Triton line exists, so backends_mps.conf routes the ops
    # backends/mps/generated/mps_register.inc claims to the `mps` kernel and
    # everything else to `none` (cpu_fallback). After stage 2 that is the
    # `unary`, `binary`, `binary_scalar` and `reduce` categories; see
    # scripts/codegen/codegen_mps.py.
    "mps": ("backends/mps/generated/mps_register.inc", ()),
    # Iluvatar is still FlagGems-primary, but it now has a native (non-boxing)
    # .inc for the first cohort of corex ixdnn/ixblas kernels, so the native slot
    # is no longer empty. backends_iluvatar.conf routes an op to `flaggems` where
    # FlagGems covers it, to `iluvatar` where only the vendor kernel does, and to
    # `none` otherwise (cpu_fallback). The two .inc files are included into the
    # same PrivateUse1 TORCH_LIBRARY_IMPL body, so they must be disjoint -- the
    # generator (scripts/codegen/codegen_iluvatar.py) intersects its claims
    # against the FlagGems .inc for exactly that reason.
    "iluvatar": (
        "backends/iluvatar/generated/iluvatar_register.inc",
        ("backends/iluvatar/generated/iluvatar_flaggems_register.inc",),
    ),
}

# Ops a vendor registers straight from register.cc instead of its .inc, so they
# are invisible to the .inc scan. Ascend claims aten::matmul (and the
# matmul_backward the derivative then binds to) on PrivateUse1 under
# `#if defined(USE_ASCEND)` so the call hits the fused aclnnMatmul rather than
# decomposing into mm/bmm/view; WrapperMatmul reads this conf to decide, so the
# entry has to be here or the fused kernel is unreachable.
EXTRA_NATIVE = {"ascend": {"matmul", "matmul_backward"}}

# Ops a hand-written wrapper claims for every platform, so they appear in no
# .inc at all. Same cause as EXTRA_NATIVE, different owner:
# csrc/aten/sdp_choice_stub.cc registers aten::scaled_dot_product_attention on
# PrivateUse1 for the CUDA-boxing builds, and that override reads this conf to
# choose between the FlagGems path and the boxing path -- so the entry has to be
# here or the choice is unreachable. It cannot be routed as its leaf ops like
# every other entry: scaled_dot_product_attention is a composite whose
# fused-backend selection runs *inside* the composite and then branches on
# query.device().type(), so on PrivateUse1 the leaves are never consulted.
EXTRA_ROUTED = {"scaled_dot_product_attention"}

# Native kernels that exist behind an op FlagGems currently *wins*, so they
# claim no m.impl of their own and the .inc scan cannot see them: the FlagGems
# .inc is included into the same PrivateUse1 TORCH_LIBRARY_IMPL body, and a
# second m.impl for one op is a duplicate registration. They are not routed --
# FlagGems stays the route. They are reachable through FLAGOS_FORCE_BACKEND=vendor
# and per-op FLAGOS_OP_*_=<vendor> overrides, which move an op only when it is
# annotated as a vendor candidate; the annotation is therefore load-bearing.
#
# They are NOT reachable through the dispatcher's runtime dtype escape, despite
# the shape of that mechanism: ResolveFn (dispatcher.h) substitutes the vendor
# slot only when FlagGemsRejectsArgs() says yes, and that predicate is compiled
# per vendor -- its `#else` arm (common.cc) returns false, and iluvatar has no
# arm. So for an iluvatar build the escape never fires and the force/override
# routes above are the only ones. (The escape's other prerequisite is present:
# WrapperMm & co. register into the vendor slot, so VendorSlot() resolves.)
# Adding a USE_ILUVATAR arm is a per-dtype decision that needs a BI-V150
# measurement of a FlagGems gap, which does not exist yet; until then this
# comment must not describe the kernels as escape-reachable.
#
# Listing them here is what keeps the annotation truthful. Without it an op like
# `mm` reads "= flaggems" with no marker -- the same line a platform with no GEMM
# kernel at all would emit -- and a kernel that only a force/override can reach
# would be indistinguishable from one that is unreachable. The generator's own
# header warns against exactly that documentation failure.
#
# Deliberately folded into NEITHER vendoring function. `vendor_native_ops()` is
# what the per-vendor generators subtract from their own registration set, so
# adding these there would make codegen_iluvatar_flaggems.py drop mm/bmm/addmm
# and the two _native_batch_norm_legit* variants from the FlagGems `.inc` -- and
# because codegen_iluvatar.py omits exactly those ops from the native `.inc`
# (that is what keeps the two files disjoint), the ops would end up registered
# nowhere and every call would raise "backend not registered".
# `vendor_registered_ops()` is the registration closure the conf's `none` gate
# reads; these ops are registered by FlagGems, so widening that set would change
# no route. They are used only by render(), to annotate the route line.
# XLA/musa/gcu/ascend/kunlunxin have no escape-only kernels.
ESCAPE_NATIVE = {
    "iluvatar": {
        "mm",
        "bmm",
        "addmm",
        "_native_batch_norm_legit",
        "_native_batch_norm_legit_no_training",
    }
}

# CUDA-boxing platforms use the same four-key shape. Their fallback kernel is the
# CUDA boxing kernel, so the `<vendor>` slot is spelled "cuda" and it covers
# every op -- which is why a boxing conf has no `none` entries while a
# native-kernel vendor has many.
#
#   platform -> (fallback backend key, conf naming that platform's triton gaps)
#
# The gap set is per-platform, not a FlagGems defect: the kernel works elsewhere,
# but this platform's triton backend cannot run it, so the op is forced back to
# boxing. MetaX's and DCU's are recovered here by diffing that platform's
# existing conf -- an op FlagGems covers whose conf line reads `cuda` is a gap
# that platform measured, so the set is read back from the artifact instead of
# restated, and the per-op diagnosis is in BOXING_GAP_NOTES below. That round
# trip only holds while this script is the confs' sole writer: codegen_ops.py
# wrote backends_metax.conf and backends_dcu.conf as well until #459, from
# fallback sets that had drifted away from the confs, so a documented full
# regeneration silently reverted the pins the diff had recovered. PPU's
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
        "add/sub/div reached with a Python float are pinned to cuda: their",
        "FlagGems pointwise kernels promote the element loop to f64, which the",
        "DCU HCU backend cannot lower. ATen boxes a Python float as an f64",
        "wrapped number, so `t + 2.0` on a bf16 tensor arrives at add.Tensor",
        "rather than add.Scalar, and the kernel takes an f64 operand beside the",
        "bf16 one. TritonHCUGPUToLLVM's TruncFOpConversion::createDestOps then",
        'asserts `inElemTy.isF32() && "unsupported conversion"`',
        "(third_party/hcu/lib/TritonHCUGPUToLLVM/ElementwiseOpHCUToLLVM.cpp:2358),",
        "surfacing as RuntimeError: PassManager::run failed through",
        "triton/backends/hcu/compiler_hcu.py make_llir. Same defect as FlagGems",
        "issue #6212 (reported for MetaX); NATIVE_TRITON_GAPS['gcu'] carries the",
        "same list. Measured on DCU, 2026-09-17, by calling every overload",
        "through torch.ops.aten with a bf16 tensor -- the only way to reach the",
        "sibling spellings, since `t + 2.0` lands on add.Tensor whatever the conf",
        "says about add.Scalar -- and by replaying all eleven through",
        "FLAGOS_OP_<op>=flaggems, dots doubled, so every verdict stays",
        "re-runnable without the conf that no longer routes them there.",
        "Python ints, f16/f32 tensors and tensor operands are unaffected:",
        "add.Scalar, div.Scalar, div_.Scalar and the six clamp spellings stay on",
        "flaggems, all measured passing, and mul was already on cuda. Pinned:",
        "add.Tensor, add_.Tensor, sub.Tensor, sub_.Tensor, div.Tensor,",
        "div.out, div_.Tensor, div.Tensor_mode, div_.Tensor_mode,",
        "div.Scalar_mode, div_.Scalar_mode. The four *_mode entries have a",
        "separate cause, and it needs a named rounding mode: 'trunc' routes",
        "flag_gems to trunc_div_func/trunc_div_func_tensor_scalar, whose whole",
        "body is `trunc(div_rn(x, y))`, and div_rn there is",
        "triton.language.extra.hip.libdevice.div_rn -- the shim carries the",
        "symbol, so use_tl_extra returns it and flag_gems' own x/y + tl.floor",
        "fallback never runs -- while that symbol lowers to None on the HCU",
        "backend. trunc(None) then raises 'cannot convert None of type",
        "<class NoneType> to tensor', and the floor path dies the same way in",
        "_float_floordiv with 'unsupported operand type(s) for -: NoneType and",
        "int' on `q - 1`. Exactly the mode flag_gems' own triton_lang_helper.py",
        "documents for asin on another fork: the symbol exists at the Python",
        "level but lowers to None. rounding_mode=None routes to true_divide",
        "instead and runs, which is why the overload survey measures all four",
        "as passing -- the rounding_mode it synthesizes is None -- so they are",
        "pinned on the named-mode measurement, not on the survey's.",
        "Reached in Qwen-Image-2512: QwenImageRMS_norm in diffusers'",
        "autoencoder_kl_qwenimage.py returns `normalized * self.scale *",
        "self.gamma + self.bias`, and self.bias is the Python float 0.0",
        "whenever the layer is built without a bias, which is how norm_out is",
        "built; with the pipeline loaded as bfloat16 (sweep.py's torch_dtype)",
        "that is add.Tensor(bf16_tensor, 0.0).",
        "scatter_add_, argmin, mean and mean.dim are pinned to cuda for a",
        "correctness reason, on FlagGems kernels that return a wrong answer or",
        "raise rather than on anything the DCU backend cannot compile. Each was",
        "measured on DCU with DAS boxing and FlagGems over the same tensor, both",
        "arms in a fresh process and the build held fixed:",
        "scatter_add_ with an index that is a valid PyTorch view(...).expand(...)",
        "of stride (1, 0) -- logically (340, 12), physically 340 elements --",
        "mismatches all 120 output elements, max |diff| 284.0, while boxing",
        "matches all 120; the FlagGems 2-D path walks the index as if it were",
        "contiguous and addresses through logical offset 4079, which is a silent",
        "wrong answer here and an HSA VMFault on other allocations. Reported as",
        "FlagGems issue #6448, with the same defect already filed as #6017.",
        "argmin over a 20-column reduction returns 20, outside the valid [0, 19],",
        "because the 32-lane kernel does not mask the tail; the survey cannot see",
        "it, since its profiles reduce over the whole of a power-of-two row.",
        "Reported as FlagGems issue #6447.",
        "mean and mean.dim raise ZeroDivisionError on an empty reduction --",
        "empty(0).mean() and empty(2, 0).mean(dim=1) -- where ATen returns NaN, so",
        "a shape-agnostic call site turns a defined result into an exception.",
        "Reported as FlagGems issue #6333.",
        "The conf is overload-granular and cannot express a shape condition, so",
        "the safe direction is the whole family on the boxing kernel --",
        "scatter_add, scatter_add.out, argmin.out, mean.out and mean.dtype_out",
        "were already there -- and the routes can be re-enabled per op with",
        "FLAGOS_OP_<name>=flaggems once the FlagGems side is repaired.",
        "native_batch_norm is pinned to cuda for an availability reason, one",
        "level below the kernels: the route does not fail, it cannot be",
        "entered. This op is routed through the FlagGems *Python* path, and",
        "the generated kernel for it calls flag_gems.native_batch_norm",
        "(csrc/aten/generated/flaggems_python_kernels.cc), a symbol the",
        "FlagGems cohort pinned in this environment (5.3.4.post1.dev1) does",
        "not export: it has batch_norm, batch_norm_backward and",
        "_batch_norm_no_update, but no native_batch_norm. The Python call",
        "resolves the attribute at run time, so the route dies with",
        "AttributeError: module 'flag_gems.ops' has no attribute",
        "'native_batch_norm' -- on every call, training or eval, and for any",
        "shape. Measured on DCU, 2026-09-28, as nn.BatchNorm2d over a",
        "(2, 4, 8, 8) fp32 input: the flaggems arm raises that AttributeError",
        "for train=True and train=False alike, while the cuda arm returns a",
        "result matching ATen (max |diff| 4.8e-07) and leaves the input",
        "unmodified. Reported as Torch-FL issue #295, which mirrors FlagGems",
        "issue #6332; on the FlagGems revision #295 measured (5.4.0rc2.post1)",
        "the symbol exists, so what #295 saw was wrong output and a mutated",
        "input rather than a missing entry point. Both revisions agree on the",
        "conclusion -- the route is unusable for this op -- and the pin is the",
        "same repair either way.",
        "The rest of the family stays where it is, and deliberately: pinning",
        "it wholesale would trade this failure for another. Measured on the",
        "same build, native_batch_norm_backward matches ATen on both arms, and",
        "_batch_norm_no_update is the inverse of this op -- it matches ATen on",
        "flaggems and raises 'Expected tensor to have CPU Backend' on cuda, so",
        "it must stay on flaggems. Only native_batch_norm moves.",
        "_native_batch_norm_legit and its variants were already on cuda.",
        "slice_backward and silu_backward are pinned to cuda for two hcu",
        "backend gaps rather than for wrong answers: both gems kernels are",
        "correct where they can run, and neither can run here. slice_backward",
        "faults the hardware -- 'Invalid address access' -- and only once the",
        "grad it produced is consumed by MIOpen's convolution_backward",
        "(tests/integration/ops/test_conv1d_dispatch.py, C=6144 depthwise), so",
        "it is an hcu codegen bug rather than a shape/stride mismatch on our",
        "side: the kernel's own output metadata and values check out when",
        "measured on its own. silu_backward calls tl.math.div_rn, whose",
        "lowering is the same div_rn shim the div.*_mode entries above are",
        "pinned for -- it returns None on the hcu backend, and here the None",
        "reaches the builder, so compilation dies with AttributeError:",
        "'NoneType' object has no attribute 'type'. MetaX pins slice_backward",
        "too, for a cause that is not this one (an out-of-bounds Xnack fault",
        "inside the gems kernel); the two entries overlap by coincidence.",
        "The .out / .grad_input spellings are spelled cuda as well, but that",
        "is not a gap: FlagGems does not cover those overloads.",
    ),
    "metax": (
        "MetaX is FlagGems-first like every generated conf, but the ops",
        "triton-metax or flag_gems cannot run on the flagos device are pinned",
        "to cuda and reach the boxing kernel (maca libtorch_cuda via mcblas).",
        "The fallback has to be cuda and not a metax-native backend: the",
        "hand-written mxcc backend is not registered in the MetaX boxing wheel",
        "(FLAGOS_BUILD_VENDOR=OFF).",
        "mm/bmm and their .out spellings are the FlagGems SPLIT_K kwarg",
        "triton-metax rejects; mean.dim is FlagGems' non-inner-dim path, which",
        "builds a CUDA context that fails on triton-metax.",
        "add.Tensor, div.Scalar and div.Tensor are the scalar-arithmetic group:",
        "MetaX Triton cannot lower it once FlagGems normalizes the scalar to a",
        "tensor overload and promotes it to f64, because the compiler then",
        "rejects f64 -> bf16. _conj is a contract pin rather than a compile",
        "one: ATen's conj is a lazy view that sets the Conjugate bit and leaves",
        "the storage alone, while FlagGems' kernel materializes the value.",
        "sort and sort.stable, plus the matmul -> relu -> sum.dim_IntList",
        "workload, are held for the profiler: the MetaX FlagGems C++ sort path",
        "corrupts that workload, and the contract above is what MetaX activity",
        "correlation is being stabilized against.",
        "The rest are flag_gems kernels that guard on tensor.device.type",
        'against flag_gems.device (== "cuda") and so fail on a flagos device',
        "either by falling back to torch.<op>, which re-enters flagos_python",
        "dispatch and recurses, or by raising ValueError('Inputs must be cuda",
        "tensors ...'): embedding_dense_backward, i0, reflection_pad2d,",
        "soft_margin_loss, special_i0e, special_i1, im2col, zero and the",
        "mul_.Tensor group among them.",
        "slice_backward is held for a fault rather than a wrong answer: gem's",
        "slice_backward_kernel reads out of bounds on large tensors (a",
        "grad_output [1, 6144, 35] scattered into [1, 6144, 32], from the",
        "conv1d backward of a linear-attention model), and on MetaX that",
        "Xnack/ATU fault (0x8) disables the whole process's mcruntime -- every",
        "later op fails with mcErrorIllegalAddress -- so the bounds-safe boxing",
        "kernel is the only way back.",
        "slice.Tensor is held on a vestigial assertion: flag_gems/ops/slice.py",
        "rejects complex64 and complex128, while its body is a pure as_strided",
        "view that never consults a dtype, and diffusers'",
        "QwenImageTransformer2DModel slices complex rotary frequencies.",
        "Reported upstream as FlagGems issue #6356; the entry can come out when",
        "that lands.",
        "Most of the rest were measured by the 2026-09-15 differential survey",
        "of the 166 ops MetaX gained at the FlagGems 5.4.0rc2.post1 ceiling,",
        "which reached each op on both routes with one host-built operand pair.",
        "The per-op verdicts, and the five flagged ops that fail on both routes",
        "and are therefore not held, are pinned by the route each op takes in",
        "this file; the differential survey that produced them is recorded in",
        "docs/reference/operator-support.md, and the routing itself is now",
        "asserted for every platform by tests/integration/flaggems/.",
        "This set is not restated here: gen_vendor_confs.py reads it back from",
        "backends_metax.conf on every run (see BOXING_PLATFORMS), so a pin is",
        "recorded by the route it takes in this file, and the gap does not",
        "drift away from the conf the way it did while codegen_ops.py wrote a",
        "second copy from a stale literal (issue #459).",
    ),
    "ppu": (
        "49 of the 482 FlagGems-covered ops are pinned to cuda; the other 433",
        "route to flaggems. mm/bmm are the first four (FlagGems issue #6225:",
        "PPU rejects the _hygon kernel's num_ldmatrixes kwarg) and 33 are the",
        "2026-09-15 overload survey's measured route-dependent failures. Four",
        "more are reflection-padding routes that survey cannot reach: it derives",
        "`padding` from the rank, so every profile fails ATen's arity check",
        "before the device guard behind it can fire, and that guard is the",
        "device-name mismatch the CUDA alignment fixes and leaves alone on every",
        "other vendor. The last eight came from checks that survey cannot",
        "see: the addmm family, whose autotune picks a BLOCK_SIZE_K < 16 config",
        "on small-K shapes that the ppu triton rejects in tl.dot; _conj, which",
        "FlagGems materializes instead of setting the Conjugate bit; _unsafe_view,",
        "whose FlagGems reshape can silently return a copy; and slice.Tensor,",
        "whose FlagGems route rejects complex dtypes. See",
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
        # -- ATen's `_unsafe_view` has the same aliasing/error contract as view:
        # a viewable layout aliases the input, while an incompatible stride
        # raises RuntimeError. FlagGems implements it as `reshape`, which may
        # silently allocate a copy for that incompatible stride. Measured on
        # PPU: a viewable input aliases on both routes, but a non-contiguous
        # transpose returns a non-aliasing tensor on FlagGems where CUDA boxing
        # raises "view size is not compatible". Keep the kernel generated for
        # explicit A/B diagnosis, but preserve the default PyTorch contract. --
        "_unsafe_view",
        # -- Qwen-Image-2.1 slices a complex64 rotary-embedding cache from the
        # second denoising step onward. FlagGems' slice wrapper asserts that
        # complex64/complex128 are unsupported before reaching its as_strided
        # implementation, while the CUDA boxing route preserves the complex
        # dtype and view semantics. Measured on PPU in the PR #342 full 40-step
        # case: the FlagGems route aborts at step 2 with that assertion; boxing
        # completes the same slice contract. --
        "slice.Tensor",
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
# a FLAGOS_BUILD_FLAGGEMS_CPP=ON build. That switch defaults ON for cuda and
# tsingmicro and OFF everywhere else, and CMakeLists.txt pins it OFF for dcu,
# musa and bpu, because the path needs FlagGems' liboperators.so built for that
# vendor's toolkit.
#
# One platform now means one conf, so a conf can no longer be reserved for the
# opt-in build that compiles the slot: backends_metax.conf is what every MetaX
# build reads, with or without FLAGOS_BUILD_FLAGGEMS_CPP=ON. The key stays legal there
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

# Which platforms' confs may route the FlagGems Python path. Ascend is in the
# set: backends_ascend.conf is FlagGems-first, so those routes must stay
# resolvable -- keep the generated conf honest by never routing a platform
# through a dispatcher slot its wheel does not compile. Boxing platforms
# compile the Python caller and may use the measured FlagGems coverage --
# including PPU, whose FlagTree install (triton with the `ppu` backend) is
# what set_env_ppu.sh provisions.
#
# Kunlunxin is deliberately absent, and a native vendor rather than a boxing
# one: libxpuapi.so (baidu::xpu::api) has the operator surface, and there is no
# FlagGems liboperators.so for the XPU SDK, so no FlagGems C++ wrapper can be
# built either. The Python path is blocked by the SDK itself -- its torch shim
# reports the FlagGems runtime device name as "cuda" while the tensors dispatch
# on PrivateUse1, so flag_gems' device gate never matches and every route falls
# through. backends_kunlunxin.conf therefore routes only the ops the native
# kunlunxin kernels back, and everything else stays at `none` (cpu_fallback).
# Add kunlunxin here only together with a measured FlagGems cohort, so the gate
# and the conf stay in step.
#
# Iluvatar is in the set from the start: it has no boxing path at all, so the
# FlagGems Python route is the only route its conf can emit, and everything it
# does not claim falls to `none` (cpu_fallback). The allowlist it registers is
# the FLAGGEMS_PYTHON_OPS ceiling; per-op pruning follows the survey measurement
# the way kunlunxin's entry above is waiting on, not a guess.
FLAGGEMS_PYTHON_PLATFORMS = {"ascend", "metax", "dcu", "gcu", "musa", "ppu", "iluvatar"}

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
# ascend: mm and mm.out used to be pinned here for a FlagGems tuner defect, and
# are no longer. The pin recorded that the Ascend tune config tunes a kernel
# that cannot accept one of its own keys:
# flag_gems/runtime/backend/_ascend/tune_configs.yaml declares
# BLOCK_M/BLOCK_N/BLOCK_K/SPLIT_K for `mm:`, while the kernel that key is fetched
# for -- flag_gems/ops/mm.py:mm_kernel_general, reached from `mm` / `mm_out`
# through the general path -- takes only BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/
# IS_FP64. The first call would die inside the autotuner's own benchmark run,
# before any kernel was compiled, with "KeyError: 'Keyword argument SPLIT_K was
# specified but unrecognised'" from triton/spec/ascend/runtime/jit.py:_pack_args.
# That was measured on Ascend910 / CANN 9.0.0 / FlagTree 0.6.2a1+ascend3.5
# against FlagGems d45285ba and 6d31db9aa.
#
# It does not reproduce on the current cohort. Measured on Ascend910C / CANN
# 9.0.0 / FlagTree 0.7.0+ascend3.5 / FlagGems 5.4.0: the yaml still declares
# SPLIT_K for `mm:` (lines 98/103/108) and runtime.get_tuned_config("mm") still
# returns the three Config objects carrying it, but the first call succeeds. The
# key never reaches _pack_args because FlagGems' own LibEntry.run no longer
# forwards **kwargs to the kernel at all -- it collects the arguments from the
# wrapped kernel's signature (self.jit_function.params, then
# self.signature.parameters) and launches positionally as
# `kernel[grid[0:3]](*all_args)` (flag_gems/utils/libentry.py:1472-1622), so a
# config key the kernel does not declare is dropped instead of passed to the
# argument packer. Confirmed with a cold cache -- a fresh HOME, which leaves
# both the triton compile cache and ~/.flaggems/config_cache empty -- so a
# cached tuned config is not the reason, and in a fresh process whose
# `[flagos dispatch]` line reads `mm -> flaggems`.
#
# Routing them here also fixes numerics the runtime-conformance suite reads. At
# 64x64 with a fixed seed, FlagGems matches an fp32 CPU reference to 8.6e-6
# max-abs; the aclnn route returns 4.7e-3, which misses a rtol=atol=1e-3
# comparison at the near-zero entries. The aclnn cube ops run fp32 operands
# through `get_cube_math_type(true)`, i.e. cubeMathType=1, which downcasts them
# to HF32; ALLOW_HF32=0 does not change the result, so the downcast is the mode
# the kernel asks for and not a CANN default to tune away.
# The rest of the family is unaffected: bmm, bmm.out and addmm already run on
# FlagGems and match a float64 CPU reference to 1.6e-7 relative, three orders
# tighter than the aclnn path's 1.7e-4. matmul is not in the FlagGems coverage
# this conf is generated from and still routes to `ascend`. The FlagGems tuner
# defect itself is still unfiled upstream.
#
# ascend: topk returns all zeros -- both the values and the indices -- and does it
# silently, for every shape measured. On Ascend910 with CANN 9.0.0, FlagTree
# 0.6.2a1+ascend3.5 and FlagGems 6d31db9aa, `torch.topk(randn(128, device=...), 5)`
# answers [0.0, 0.0, 0.0, 0.0, 0.0] / [0, 0, 0, 0, 0] against a CPU reference of
# [3.4105, 2.5672, 2.3025, 2.3022, 1.9218] / [59, 69, 89, 45, 122]. The FlagGems
# body is reached -- its `GEMS TOPK` debug line fires -- and calling
# flag_gems.ops.topk directly on the same operand returns the same zeros in 0.02s,
# so the shared entry point is what is wrong, not this conf's plumbing.
#
# The indices are what identifies the failure. topk_single_stage_kernel stores
# `sorted_idx`, which is built from `tl.where(mask, cols, mask_index_val)` -- for
# this shape every element of it is a distinct value in 0..127, and the pad is
# INT32_MIN. Five zeros cannot be sorted output from that. The values buffer is
# built the same way, from `x_val` padded with float("-inf"). Neither buffer holds
# anything the kernel could have put there, so the store never lands: this is an
# empty output, not a mis-sorted one. (N=128 with k=5 has HAS_TLE False and
# x.is_cuda False on this backend, so topk.py's radix-TLE fast path is skipped and
# the single-stage bitonic kernel above is the one that runs.) The compile is also
# pathological rather than merely wrong at the margin -- bishengir-compile spends
# minutes on a single (shape, k) pair, and the large-N shape in the same family is
# already recorded as blowing the unified-buffer budget (`randperm(2000)` through
# topk.py, "ub overflow, requires 10092544 bits").
#
# Ascend implements topk through aclnnTopk (csrc/aten/backends/ascend/topk.cc), so
# the route back is free -- and it is exact: with FLAGOS_FORCE_BACKEND=vendor the
# same call returns the CPU reference values and indices, 0 zeros. This is what
# un-blocks tests/integration/test_compute_device_index.py::test_tuple_returning_op,
# which is itself blocked from CI by issue #391. Not yet filed upstream.
#
# ascend: sum.dim_IntList, because the FlagGems kernel reduces a bool operand in
# i1 -- which turns the reduction into an "any" and makes the count wrong. Two
# defects sit behind it, both in the non-fp16 branch of `sum_dim_kernel`
# (flag_gems/ops/sum.py:229):
#
#   * `cdtype = inp.dtype.element_ty` (:242) leaves a bool input accumulating in
#     i1, where `_sum += a` (:257) is a bitwise OR and `tl.sum` (:258) over i1 is
#     an "any". Every row holding at least one True reduces to `1` and every row
#     holding none to `0`, in the correct int64 dtype, so nothing downstream can
#     tell the count is wrong. Measured on Ascend with the FlagGems route forced
#     back: a `(1, 52)` bool tensor with 52 Trues returns `[1]`, with 40 Trues
#     `[1]`, with 1 True `[1]`, with no Trues `[0]`; a `(3, 52)` tensor whose rows
#     hold 52/0/52 Trues returns `[1, 0, 1]` along dim=1 and `[1, 1, 1, 1]` along
#     dim=0, where the CPU returns `[1, 0, 1]` and `[2, 2, 2, 2]`.
#   * the same i1 accumulate is rejected outright by BiShengIR on the
#     multi-dimension path, so `mask.sum(dim=(0, 1))` is a hard RuntimeError
#     rather than a wrong answer:
#         loc("_sum"(.../flag_gems/ops/sum.py:257:16)): error: 'hivm.hir.vadd' op
#         failed to verify that operand at idx 0 and 1 should have element type
#         16-bit signless integer or 32-bit signless integer or 16-bit float or
#         32-bit float or 64-bit signless integer
#
# The float, fp16 and int64 paths are numerically correct, so this is a dtype
# gap that a per-op conf entry spends more than it needs to. It is written that
# way anyway because the mechanism this repo has for a dtype gap
# (`FlagGemsRejectsOpDtype` in csrc/aten/common.cc) exists to keep an op on
# FlagGems for the dtypes that work, and the only thing keeping `sum.dim_IntList`
# there for float would be an unmeasured speed claim -- aclnnReduceSum is exact
# for every dtype. The survey's 2d-f32, 4d-f32, 1d-f32, 2d-f16, 2d-i64 and
# 2d-f32-strided profiles all PASS; the 2d-bool profile is the one that is WRONG.
#
# `mask = row_mask and col_mask` (:254) is not part of this. It reads like a
# Python `and` that would return the second operand and drop the row check, and
# Triton warns about it on every launch ("Logical operators 'and' and 'or' are
# deprecated for non-scalar tensors"), but the front end lowers a BoolOp between
# block tensors to `logical_and` (`visit_BoolOp`, folding its operands through
# `_apply_binary_method`; triton/compiler/code_generator.py:1575-1587), so the
# mask is correct and only the spelling is deprecated.
#
# The full reduction is unaffected: `mask.sum()` routes through `sum`, not
# `sum.dim_IntList`, and returns `tensor([52], dtype=torch.int64)` correctly.
#
# This is what blocks the Qwen-Image text encoder on Ascend. diffusers'
# `QwenImagePipeline._extract_masked_hidden` does `bool_mask.sum(dim=1)` and
# feeds the result to `torch.split`, so the wrong [1] becomes
# `RuntimeError: split_with_sizes: split sizes sum to 1 but tensor has 52
# elements along dim 0` -- a crash three frames away from its cause.
#
# Ascend implements sum.dim_IntList through aclnnReduceSum
# (SumDimIntlistKernelAscend in csrc/aten/backends/ascend/generated/), so the
# route back is free. That kernel needed a fix of its own before the move was
# correct -- it did not apply torch's no-dtype promotion and returned the input
# dtype, so a bool sum came back `tensor([True])` rather than int64 counts; the
# promotion now lives in codegen_ascend.py's `_reduce_dtype_prologue`. With both
# in place `mask.sum(dim=1)` returns `[52]` and `mask.sum(dim=(0,1))` returns
# `[52]`, bit-identical to the CPU. Not yet filed upstream.
#
# ascend: gelu and gelu_backward, both `approximate` spellings and every dtype,
# because the failure is in the kernels rather than at a dispatch boundary.
# Three of flag_gems/ops/gelu.py's four kernels evaluate `pow(x_fp32, 2)` --
# `gelu_tanh` (:43), `gelu_backward_none` (:54) and `gelu_backward_tanh` (:67,
# :69) -- and `pow` there is `tl_extra_shim.pow`. The Python int literal `2`
# reaches triton-ascend's libdevice binding as an int32 operand, and the table
# that resolves it has no `(float32, int32)` overload:
#
#     triton.compiler.errors.CompilationError: at 3:71:
#     def gelu_tanh(x):
#         x_fp32 = x.to(tl.float32)
#         output = 0.5 * x * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32, 2))))
#     KeyError((triton.language.float32, triton.language.int32))
#
# The column is the `pow(` call. `gelu_none` is the one kernel in the file with
# no `pow` and the only one that compiles, so `approximate="none"` runs and
# `approximate="tanh"` cannot launch at all; `gelu_backward_none` uses the same
# construct, so a fwd+bwd graph is unbuildable whichever forward spelling is
# chosen. This is what blocks the Qwen-Image transformer on Ascend, and it is not
# avoidable there: diffusers' `QwenImageFeedForward` builds its activation with
# `approximate="tanh"` (models/activations.py:85, reached from
# transformer_qwenimage.py), so all 60 blocks take the failing route.
#
# Measured with tests/manual/flaggems_overload_survey.py against the pre-fix conf
# (SHA-256 9d24378804775bda932f4572f94b1984b88fd069d659e16187c5ce8dab80918e), which
# is where the two ops part company: `gelu_backward` FAILED on every float
# profile -- 2d-f32, 4d-f32, 1d-f32, 2d-f16, 2d-f32-strided -- with the KeyError
# above, so no per-dtype escape could state it, while `gelu` came back STRICT
# because the survey drives the default `approximate="none"` spelling only. The
# gap in `gelu` is argument-valued, and the conf keys on the ATen overload, so
# the only way to route the tanh call away from FlagGems is to move the overload.
#
# Ascend implements both ops through aclnnGeluV2 / aclnnGeluBackwardV2
# (GeluKernelAscend / GeluBackwardKernelAscend in
# csrc/aten/backends/ascend/generated/), so the route back is free. Measured on
# Ascend910 with CANN 9.0.0, FlagTree 0.6.2a1+ascend3.5 and FlagGems 6d31db9aa
# over the three gate shapes the transformer actually feeds it -- (2, 4096, 24),
# (1, 32, 3584), (2, 8, 256) -- forward and backward, both `approximate`
# spellings: max|d| 4.8e-07 forward and 1.3e-06 backward against a float64 CPU
# reference, all finite. Not yet filed upstream.
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
# A gap entry only takes the op off FlagGems; where it goes instead is the native
# side's decision, so the entry stays useful after codegen_gcu.py learns the op.
# `arange` and `zero_` are the two here that have since been claimed by
# codegen_gcu.py, and they route to `gcu` rather than `none` because of it.
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
# Several entries have no topsaten kernel claimed by codegen_gcu.py (`linspace`,
# `full`, `full_like`, `scalar_tensor`, `constant_pad_nd`, `embedding`,
# `nll_loss_*`, `floor_divide*`, `index_add*`, `sort*`). There the gap drops them
# from both the conf's FlagGems route and the generated registration, so route()
# falls through to `none` and the call reaches ATen's cpu_fallback. That is the
# route these ops already took before GCU had a FlagGems path, so nothing
# regresses; it is not a vendor kernel because no vendor kernel is claimed for
# them yet. `arange` (all three overloads) and `zero_` used to be on that list
# and are not any more: codegen_gcu.py claims a topsaten kernel for both, so the
# gap now hands them to `gcu`. The entry still has to stay, because deleting it
# would put `arange` back on the FlagGems path that fails to compile -- for the
# factory family the gap is what selects the vendor, not what excludes it.
#
# Measured on S60 with FlagGems master 3c6f7537d2d5d3aa680c55bbee5c70f2100c5b85
# (5.4.0.dev0) + flagtree 0.6.1+enflame3.6.
NATIVE_TRITON_GAPS = {
    "ascend": {
        # `cumsum` is an Ascend-backend kernel defect in the installed
        # FlagGems, not a triton-lowering gap: its aclnn kernel is registered
        # below (ascend_register.inc: WrapperCumsum; ascend_kernels.cc:
        # CumsumKernelAscend -> aclnnCumsum), and the conf therefore annotates
        # the FlagGems route `# ascend` -- the kernel is reachable, it just
        # must not be the one a run reaches first.
        #
        #   cumsum -- flag_gems/runtime/backend/_ascend/ops/cumsum.py:225
        #     sizes its reduction with `K = inp.numel() // M // N`, where
        #     `N = shape[dim]`. For an empty reduction dim (e.g. a bool
        #     tensor of shape (1, 0), the input `_assisted_decoding` builds at
        #     transformers/generation/utils.py:4107) that is `0 // 1 // 0` ->
        #     ZeroDivisionError. An empty dim is a legitimate input -- ATen's
        #     own cumsum returns an empty tensor -- and the generic
        #     flag_gems/ops/cumsum.py does not divide by `N`.
        #
        # This is the third instance of the same `_ascend` override defect
        # class on this backend (cf. `ones`/`ones_like` in the sibling change
        # and the annotated entries below): the override does shape arithmetic
        # that divides by a dimension size, and an empty dim makes it raise.
        # The torch_fl lever is the route, and the aclnn kernel that serves it
        # already exists.
        "cumsum",
        "eq.Scalar",
        "eq.Tensor",
        "exponential_",
        # `flip` is an Ascend-backend kernel defect in the installed FlagGems,
        # not a triton-lowering gap: its aclnn kernel is registered below
        # (ascend_register.inc: WrapperFlip; ascend_kernels.cc:
        # FlipKernelAscend -> aclnnFlip), and the conf therefore annotates the
        # FlagGems route `# ascend` -- the kernel is reachable, it just must
        # not be the one a run reaches first.
        #
        #   flip -- flag_gems/runtime/backend/_ascend/ops/flip.py:71
        #     realizes the reversal as a strided `copy_func` pointwise kernel
        #     (`overload = copy_func.instantiate(A.ndim)`), and the rank-2
        #     instantiation that FlagTree 0.7.0's Ascend backend generates
        #     from it fails to lower:
        #
        #       /root/.flaggems/code_cache/pointwise_dynamic_..._copy_func_
        #       kernel_rank_2__t2048.py:59:0: error: Failures have been
        #       detected while processing an MLIR pass pipeline
        #       [ConvertTritonIRToLinalgIR] encounters error:
        #       PassManager::run failed
        #
        #     The failure is in the FlagTree MLIR pipeline, not in the torch_fl
        #     route, but the route is the lever: the aclnn kernel that serves
        #     `flip` already exists and returns correct values. Measured on
        #     Ascend 910C (`bm-jn-zs-zone1-910C-64G-10-108`, torch_fl
        #     `c2ded4b`): a plain `torch.arange(24).reshape(4, 6).to("flagos")
        #     .flip(1)` raises `MLIRCompilationError` with the FlagGems route
        #     and returns the correct tensor with the aclnn route
        #     (`[flagos dispatch] flip -> ascend`). On the transformers-5.19.0
        #     bert suite this is worth three ids: the two
        #     `*_padding_matches_padding_free_with_position_ids` tests
        #     (`tests/generation/test_utils.py:2341` calls `attention_mask
        #     .flip(1)`) and `test_generate_with_and_without_position_ids`.
        "flip",
        "ge.Scalar",
        "ge.Tensor",
        "gelu",
        "gelu_backward",
        "gt.Scalar",
        "gt.Tensor",
        "le.Scalar",
        "le.Tensor",
        "lt.Scalar",
        "lt.Tensor",
        # `max`/`max.dim` and the `ones` family are Ascend-backend kernel
        # defects in the installed FlagGems, not triton-lowering gaps: their
        # aclnn kernels are registered below (ascend_register.inc:
        # WrapperMax/WrapperMaxDim/WrapperOnes/WrapperOnesLike), and the conf
        # therefore annotates the FlagGems route `# ascend` -- the kernel is
        # reachable, it just must not be the one a run reaches first.
        #
        #   max/max.dim — flag_gems/runtime/backend/_ascend/ops/max.py:100
        #     builds its block mask with Python `and` on non-scalar tensors
        #     (`mask = m_offset[:, None] < M and n_offset[None, :] < N`), the
        #     deprecated form that does not produce the intended elementwise
        #     mask. `torch.max(t)` measured 43.4% mismatched elements
        #     (558/1287) on bert.
        #   ones/ones_like — flag_gems/runtime/backend/_ascend/ops/ones.py:56
        #     sizes its grid with triton.cdiv(N, BLOCK_SIZE) where
        #     BLOCK_SIZE = next_power_of_2(ceil(sqrt(volume))). For a
        #     zero-element shape (e.g. (2, 0)) that is cdiv(0, 0) →
        #     ZeroDivisionError; the generic flag_gems/ops/ones.py hardcodes
        #     1024 and does not have the bug.
        #
        # Both are upstream FlagGems bugs on the `_ascend` backend override,
        # reproducible with the lines above; the torch_fl lever is the route,
        # and the aclnn kernels that serve it already exist.
        "max",
        "max.dim",
        "mul_.Tensor",
        "native_dropout",
        "native_dropout_backward",
        "ne.Scalar",
        "ne.Tensor",
        "ones",
        "ones_like",
        "rand",
        "rand_like",
        "randperm",
        "sort",
        "sort.stable",
        "sum.dim_IntList",
        "topk",
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
        # `rsub` is the reflected `other - self` and belongs to the same family:
        # its FlagGems kernel is the same pointwise template and fails on an
        # integral operand for the same reason. Measured on the S60 through the
        # HF bert cohort, which is what made it visible --
        # `1 - unfinished_sequences` over the int64 mask transformers builds at
        # generation/utils.py:3257 reaches `Tensor.__rsub__` ->
        # flag_gems/ops/rsub.py:45 -> GCU300's
        # `64-bit data type not supported on GCU300!` at
        # triton/backends/enflame/toolkit.py:146. Four bert tests
        # (assisted-decoding x2, assisted_decoding_sample,
        # generate_continue_from_inputs_embeds) failed there and nowhere else.
        #
        # Unlike the rest of this group, `rsub` has a topsaten entry point of its
        # own -- `topsatenRsub`, with the reflected operand order -- so gapping it
        # routes to `gcu` rather than to `none`. That kernel is what
        # codegen_gcu.py's T_BINARY_RSUB_SCALAR emits; the two changes are one
        # fix and neither half works alone. Gapping without the kernel raises
        # `rsub.Scalar: backend not registered` (measured with
        # `FLAGOS_OP_rsub__Scalar=gcu`), and the kernel without the gap is never
        # reached because FlagGems wins the route.
        "rsub.Scalar",
        "rsub.Tensor",
        # Factory / creation ops that widen to a 64-bit element type.
        #
        # `new_ones` is here because flag_gems' `new_ones` is a thin wrapper over
        # its `ones` kernel: `flag_gems/ops/new_ones.py:51` runs
        # `ones_kernel[grid_fn](out, N, BLOCK_SIZE=1024)`, whose i64 instantiation
        # is the one at `flag_gems/ops/ones.py:32` that GCU300 cannot lower. It is
        # the only member of the `new_*` family that reaches the generated op list
        # at all -- `new_empty`, `new_zeros` and `new_full` have no conf entry and
        # no registration, so they were never routed and were always correct --
        # and that left it as the one factory a model could not call on an int64
        # tensor. `_update_model_kwargs_for_generation` makes that call on every
        # generation step, at `transformers/generation/utils.py:991`
        # (`attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))`),
        # so the generate half of the BERT cohort failed there with
        # `RuntimeError: Pipeline run failed: PassManager execution failed`.
        #
        # Unlike its siblings, `new_ones` does not stop here. `ones`, `zeros` and
        # `full` have no vendor entry point that takes a caller-supplied shape, so
        # taking them off the FlagGems route leaves the ATen composite to
        # decompose on the device into the `empty` + `fill_`/`zero_` pair
        # codegen_gcu.py claims. `topsatenNewOnes` does take the shape, and
        # codegen_gcu.py emits a kernel for it, so `new_ones` is served natively:
        # membership in this set is what removes it from FlagGems coverage, and
        # the native kernel behind it is what `route()` then falls through to.
        # Drop the entry and the route returns to `flaggems` even with the kernel
        # registered; keep the entry and remove the kernel and it falls back to
        # `none`.
        "arange",
        "arange.start",
        "arange.start_step",
        "constant_pad_nd",
        "full",
        "full_like",
        "linspace",
        "new_ones",
        "ones",
        "ones_like",
        "scalar_tensor",
        "sum",
        "zero_",
        "zeros",
        "zeros_like",
        # The rest of this group is here because what these ops legalize to
        # cannot run on GCU either, but the two routes they land on are not the
        # same one, so they are listed by what they fall through to.
        #
        # `embedding` is served natively: `topsatenEmbedding` is a pure gather
        # and codegen_gcu.py emits the kernel for it, so its membership here is
        # what keeps FlagGems off the route and lets route() fall through to
        # `gcu` -- the same pairing as `new_ones` above. Before the native kernel
        # existed the set closed the FlagGems route with nothing behind it, so
        # the op routed to `none` and every lookup in a model reached the
        # cpu_fallback.
        #
        # `floor_divide`, `index_add`, `nll_loss_backward`, `nll_loss_forward`
        # and `sort` have no topsaten entry point, so they stay on `none` and
        # reach that same cpu_fallback deliberately.
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
        # the int64 case. The in-place writes do too -- `fill_.Scalar`,
        # `fill_.Tensor`, `masked_fill_.Scalar`, `masked_fill_.Tensor` are all
        # claimed by codegen_gcu.py now, which matters more here than for the
        # comparisons: they are what `torch.zeros` / `torch.full` decompose into
        # on a device, so routing them to `gcu` is what keeps the int64 raise
        # from turning every factory call in a model into a host round trip. The
        # rest of the group -- the out-of-place `fill.*` and `masked_fill.*`,
        # `masked_select`, `any` and the bitwise family -- has no topsaten
        # kernel, so it lands on `none` and the call reaches cpu_fallback, the
        # same route these ops already took before GCU had a FlagGems path.
        #
        # `where.self` and the whole-tensor `all` were in that list and are not
        # any more: codegen_gcu.py claims both now, so they route to `gcu`. The
        # entries stay in the set anyway, and dropping either one would be a
        # regression rather than a cleanup -- the set is what closes the FlagGems
        # route, and FlagGems still wins the route whenever both exist (see the
        # i64 note above). Which overload `all` is matters here: only the
        # no-dim schema is claimed, while `all.dim`/`all.dims` stay on FlagGems
        # and `all.out`/`all.all_out` stay on `none`.
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
        # The out= spelling is gapped for its own reason, and it is not a
        # correctness one: FlagGems' `where_self_out` is correct but 46x slower
        # than the vendor's op when the second operand is 0-dim, which is
        # exactly how ATen's `_safe_softmax` calls it. Measured at
        # (1, 24, 4114, 4114) fp32, condition (1, 24, 4114, 1) bool, with a
        # drain on both sides: 1098.6 ms through FlagGems against 23.7 ms for
        # `where.self` at the same shape. `_safe_softmax` is the SDPA math
        # path's softmax and runs once per attention layer, so the gap is
        # ~1.1 s/layer. See T_WHERE_SELF_OUT in codegen_gcu.py.
        "where.self_out",
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
        # `cat` reached FlagGems when the tensorlist_dim codegen category gave
        # `aten::cat(Tensor[] tensors, int dim)` a Python route, and the FlagGems
        # route does not compile on this vendor. Measured 2026-10-12 on Enflame
        # GCU S60, GCU integration suite:
        # tests/integration/ops/test_cat_dispatch.py::TestCatKvCacheAppend,
        # `torch.cat([k_prefill, k_new], dim=-2)` over concatenands of differing
        # sizes, raised
        # `triton.compiler.errors.CompilationError: AssertionError('Mismatched
        # type for dim_size_in between then block (int32) and else block
        # (int64)')` out of
        # flag_gems/runtime/backend/_enflame/gcu300/ops/cat.py:291
        # (cat_copy_row_kernel_4). It is an Enflame-side compile bug in the
        # FlagGems `cat` kernel, not a routing mistake: the kernel merges the two
        # per-concatenand size scalars into one variable and the GCU300 front end
        # rejects the int32/int64 join. Membership here closes the FlagGems route
        # so `cat` falls back to the vendor kernel T_CAT in codegen_gcu.py --
        # the route it took before the reroute, and one this suite measured
        # working. Drop the entry if a later FlagGems fixes the kernel.
        "cat",
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
        # Group C -- correct at every dtype, broken at scale, so the dtype sweep
        # above cannot see it: every one of its profiles is at most 32 rows wide.
        # The enflame backend sizes a grid axis from a data dimension, and GCU's
        # limits are far below CUDA's -- grid.x 65535 against 2**31-1, grid.y 255
        # against 65535 -- so a kernel that is launched once per row (or per
        # 8-row tile) works in the sweep and raises on a real workload.
        #
        # _softmax
        # `softmax_kernel_inner` is launched with `grid = (M, 1, 1)` for a
        # last-dim softmax (flag_gems/runtime/backend/_enflame/gcu300/ops/
        # softmax.py:325), M being the product of the leading dimensions.
        # Measured on the S60 with `torch._softmax(x, -1, False)` on bf16:
        #
        #   (1, 24, 2729, 2729)  M = 65496  PASS
        #   (1, 24, 2731, 2731)  M = 65544  FAIL grid.x Required 65544
        #   (1, 24, 4114, 4114)  M = 98736  FAIL grid.x Required 98736
        #
        # The non-inner kernel has the same defect on the other axis: a softmax
        # that is not over the last dimension launches with
        # `grid = (M, cdiv(K, TILE_K), 1)`, so (24, 4114, 4114) over dim 0 fails
        # with `grid.y Required 2067`. First caller is Qwen-Image-2512's joint
        # attention, `_softmax` on (1, 24, 4114, 4114) reached from
        # `F.scaled_dot_product_attention` in every one of the 60 transformer
        # blocks. topsatenSoftmaxForward serves every shape above -- it has no
        # grid to size -- so gapping routes it back to the vendor kernel.
        #
        # linalg_vector_norm
        # `l2_norm_kernel` is launched with `grid = (cdiv(M, BLOCK_M),)`, and the
        # enflame tune config for it offers only BLOCK_M in {1, 2, 4, 8}
        # (runtime/backend/_enflame/gcu300/tune_configs.yaml:762) -- 8 is the
        # widest, so no tuned config fits M above 65535 * 8 = 524280.
        # Measured on the VAE decode: `F.normalize(x, dim=1)` from
        # diffusers' QwenImageRMS_norm on (1, 128, 1, 1024, 1024), M = 1048576,
        # fails with `grid.x Required 131072` at every candidate.
        #
        # For a while the gap landed on `none` and the op reached cpu_fallback,
        # because no topsaten kernel was claimed for it: correct results at the
        # cost of a host round trip per call. That is no longer the state --
        # codegen_gcu.py now claims `topsatenLinalgVectorNorm`, which takes the
        # dim list and the order directly and has no grid to size, so the gap
        # routes to `gcu` and the 1M-element VAE call above stays on the device.
        "_softmax",
        "linalg_vector_norm",
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
        # addmm/baddbmm: FlagGems' MThreads matmul kernels lose fp64, and they
        # lose it in two different ways. `_mthreads/ops/baddbmm.py` declares an
        # `IS_FP64` constexpr that `_baddbmm_launch` never passes to the kernel,
        # so the accumulator is unconditionally float32 and the fp64 call
        # raises in the Triton front end. `_mthreads/ops/addmm.py` downcasts
        # both operands to float32 before the dot, so an fp64 call returns a
        # float32 answer without saying so. Measured on the MUSA CI runner with
        # flag_gems 5.4.0 and torch 2.10.0+cpu, 64x64x64 against a CPU fp64
        # reference: `baddbmm` raised `CompilationError` and `addmm` returned a
        # float32-grade answer (the gap tracks the fp32 operand rounding, and
        # its size moves with the seed; the downcast is the stable fact). The
        # vendor kernel that takes both over is T_ADDMM
        # in codegen_mudnn.py; the route and this entry are two halves of one
        # claim. FlagGems is told about the two kernels separately, and these
        # entries come off when a release no longer ships the defect.
        "addmm",
        "baddbmm",
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
        # The out= spelling is gapped for ATen's out= contract and not for a
        # throughput one. FlagGems' `where_self_out` computes the broadcast
        # shape only when `out is None`; handed a destination it passes it
        # straight to `where_inner(..., out0=out)`, whose `prepare_args`
        # validates rather than resizes, so `torch.where(c, a, b, out=out)`
        # raises where ATen's own overload grows `out`. Measured on the MUSA CI
        # runner with flag_gems 5.4.0rc2.post1+g437ba3938:
        # `RuntimeError: out tensor at index 0 shape is invalid, should be
        # (2, 4) but is torch.Size([0])!` from flag_gems/utils/
        # pointwise_dynamic.py, raised through flag_gems/ops/where.py:81 for
        # tests/integration/ops/test_where_dispatch.py::TestWhereOutCorrectness::
        # test_out_grows_from_empty -- the only failing case of that group's
        # 508. `where.self` is unaffected and stays on FlagGems, because the
        # wrapper sizes its own destination. The native kernel that takes the
        # overload over is T_WHERE_OUT in codegen_mudnn.py; the route and this
        # entry are two halves of one claim.
        "where.self_out",
    },
    # Measured on a BI-V150 (ub22) with flag_gems 5.3.5 + flagtree/triton 3.6.0
    # (corex 4.5.0), 2026-10-14. Harness: tests/manual/flaggems_overload_survey.py,
    # seven profiles per overload, 639-op cohort (conf_sha256
    # f08d420acf4d39104bb6c49db1c72c76513fc82e0509dba5173d6e484739046d): 503
    # tested, STRICT 379 / BASIC_ONLY 66 / FAILED 58 / UNTESTED 136, plus `cat`
    # measured separately as STRICT on the 640-op conf.
    #
    # Re-measured 2026-10-15 on the same host against THIS conf (conf_sha256
    # 5b2ab727110268857c2e0f27cd7f5f84900f37614be1ae21ba098a2ce6a675bc): 603
    # routes, 467 tested, STRICT 380 / BASIC_ONLY 66 / FAILED 21 / UNTESTED 136.
    # No op newly failed -- the 37 the first run gapped are simply off the route,
    # and `cat` (added to the ceiling by upstream #498 after the first conf) is
    # now in-band. So this set is still the defect subset of the FIRST run's 58
    # FAILED ops, and the second run neither adds to nor subtracts from it.
    #
    # The set below is the subset of the 58 FAILED ops that are defects rather
    # than harness synthesis limits. Three failure families dominate:
    #
    #  - **device-guard mismatch.** The `_iluvatar` backend declares
    #    `device_name="cuda"` (flag_gems/runtime/backend/_iluvatar/__init__.py),
    #    but this platform's device is `flagos`, so every op that asserts
    #    `x.device.type != flag_gems.device` (i0, the special_* Bessel/Chebyshev
    #    family, soft_margin_loss) or that calls `torch_device_fn.get_device_properties`
    #    (masked_scatter_backward, upsample_bicubic2d's `is_cuda` guard) rejects
    #    every tensor the platform can hand it. Filed upstream as FlagGems #7135;
    #    gapped here so the calls fall back to cpu_fallback instead of raising.
    #  - **real kernel/compile defects.** randint / randint_like (Python int
    #    reaches the Triton front end as `int.to()`), norm.Scalar / norm.ScalarOpt_dim,
    #    unique_consecutive (dim set recurses through torch.unique_consecutive),
    #    special_shifted_chebyshev_polynomial_w (a runtime `if` around tl.constexpr),
    #    the wrong-value family (histc, gcd_, elu's iluvatar acos misuse,
    #    special_chebyshev_polynomial_u/v, special_hermite_polynomial_h,
    #    special_shifted_chebyshev_polynomial_u), and the CRASHing batch-norm
    #    arm (native_batch_norm, _batch_norm_no_update,
    #    _native_batch_norm_legit.no_stats).
    #  - **out-variant / shim bugs in the torch_fl Python-kernel path.**
    #    linalg_matrix_exp.out, nansum.out and sum.out do not forward `out` (or
    #    do not resize it); range returns float64 for a float32 request;
    #    mse_loss_backward indexes past the flattened target.
    #
    # Deliberately NOT gapped (harness synthesis limits, op left on FlagGems):
    # `elu`/`elu_`/`elu_backward` (the harness fills elu's `scale`/`input_scale`
    # Scalars with 0.5, a form no torch.nn caller produces -- F.elu passes only
    # `alpha`, and with a coherent triple the kernel matches to 6e-8), `_unique2`
    # (the faithful 3-tuple shim vs the harness's single-tensor compare),
    # `_adaptive_avg_pool3d_backward`/`.grad_input` (only the 4-D profile is
    # contract-valid; a 5-D path would need re-measuring), `lu_unpack.out`
    # (harness synthesis of the schema-default bool flags), `_pdist_backward` /
    # `igamma` / `igamma_` / `logit_backward` (ATen's reference is NaN on the
    # synthesized domain; finite elements agree), `_compute_linear_combination`
    # / `.out` (SPLIT_K kwarg the harness cannot construct), `_cdist_forward`,
    # `dequantize.self` (`int_repr` absent on the CPU reference),
    # `fake_quantize_per_channel_affine_cachemask`, `linalg_eig` (eigenvalue
    # ordering is unspecified), `linalg_lstsq` (metadata shape), `_softmax_backward_data`
    # / `_log_softmax_backward_data` (device returns int8 -- needs one targeted
    # probe before gapping) and `unique_dim` (return-metadata).
    "iluvatar": {
        # device-guard mismatch (cuda vs flagos)
        "i0",
        "i0.out",
        "special_i0e",
        "special_i0e.out",
        "special_i1",
        "special_i1e",
        "special_i1e.out",
        "special_modified_bessel_k0",
        "special_modified_bessel_k0.out",
        "special_scaled_modified_bessel_k1",
        "special_scaled_modified_bessel_k1.out",
        "special_chebyshev_polynomial_w",
        "special_chebyshev_polynomial_w.out",
        "soft_margin_loss",
        "masked_scatter_backward",
        "upsample_bicubic2d",
        # wrong values
        "histc",
        "gcd_",
        "special_chebyshev_polynomial_u",
        "special_chebyshev_polynomial_v",
        "special_hermite_polynomial_h",
        "special_shifted_chebyshev_polynomial_u",
        # Triton compile / structural defects
        "norm.Scalar",
        "norm.ScalarOpt_dim",
        "randint",
        "randint_like",
        "randperm",
        "special_shifted_chebyshev_polynomial_w",
        "unique_consecutive",
        # CRASH (context poison)
        "native_batch_norm",
        "_batch_norm_no_update",
        "_native_batch_norm_legit.no_stats",
        # out-variant / shim bugs in the torch_fl Python-kernel path
        "linalg_matrix_exp.out",
        "nansum.out",
        "sum.out",
        "range",
        "mse_loss_backward",
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
# setup.py force-sets FLAGOS_BUILD_TILEOPS=OFF for every FLAGOS_ACCELERATOR != "cuda" -- so on
# any other platform the slot is empty, Dispatcher::GetFn degrades kTileOps to
# cuda_fn_, and on a native-kernel vendor that is empty too. Naming `tileops` in
# those confs would therefore route a real op at nothing, which is the "backend
# not registered" class of failure this full-coverage rework exists to remove.
# No shipped conf is generated for plain cuda (codegen_ops.py writes
# backends_cuda.conf), so this is currently empty and the key reaches confs only
# through the `# tileops` annotation that FLAGOS_FORCE_BACKEND=tileops reads.
TILEOPS_PLATFORMS = set()

# The FlagGems C++ ops MetaX routes to the C++ path in the build its CI lane and
# its shipped wheel actually produce. Fifteen of the 18 in the shared C++ set;
# `mm` is one deliberate omission and the two `*_softmax`-adjacent drops another,
# both explained below. Recorded here because the conf that used to hold this
# measurement no longer exists separately; see boxing_cpp_ops().
# test_metax_conf_keeps_mm_boxed pins the `mm` exception.
#
# The `mm` omission: gems' `mm` passes a SPLIT_K kwarg triton-metax rejects, so
# `mm` stays on the boxing kernel. Measured by forcing it onto the C++ route with
# `FLAGOS_OP_bmm=flaggems_cpp` on MetaX hardware: it fails on the fp32 inputs
# `aten::bmm` must serve ("soft-lowp matrix kernel requires a low-precision
# input").
#
# Verification on this route is not the same as being routed on it: five of these
# also sit in MetaX's boxing gap set, which route_boxing() checks first, so the
# conf keeps `bmm`, `bmm.out`, `sort`, `sort.stable` and `sum.dim_IntList` on the
# boxing kernel and only 10 of the 15 reach flaggems_cpp. That set is read back
# from backends_metax.conf (see BOXING_PLATFORMS) and its per-op reasons are in
# BOXING_GAP_NOTES["metax"]. `sort`'s pin is its profiler interaction, not its
# values -- it computes the host answer on the C++ route.
#
# The two drops: `_softmax` and `embedding` were measured servable through the
# C++ path on a host whose FlagGems build had the C++ half compiled in, but they
# are two of the ten ops
# tests/integration/runtime/runtime_caps.py:RUNTIME_OPS exercises, and the MetaX
# CI lane installs the pure-Python FlagGems wheel, which ships no
# liboperators.so. With no `flaggems_fn_` the dispatcher degrades a flaggems_cpp
# route to the CUDA boxing kernel, so those two entries named a route neither CI
# nor the shipped build can serve, and the runtime-conformance suite could not go
# green. The honest measured state for the current build is that they are *not*
# C++-servable, so they are listed off. Both are also in FLAGGEMS_PYTHON_OPS, so
# they fall through to `flaggems`; restoring the C++ routing is deferred to the
# follow-up PR that wires `-DFLAGGEMS_BACKEND=MACA
# -DFLAGGEMS_BUILD_C_EXTENSIONS=ON`, `FLAGOS_BUILD_FLAGGEMS_CPP=1` and
# `FLAGGEMS_DIR`/`FLAGGEMS_SOURCE_DIR` into the MetaX CI hook (see the MetaX
# section of docs/reference/operator-support.md).
METAX_CPP_MEASURED = {
    "_softmax_backward_data",
    "addmm",
    "addmm.out",
    "argmax",
    "bmm",
    "bmm.out",
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

# Ops codegen_ops.py's `flaggems_runtime_broken` group (5) holds off the FlagGems
# path because the generation cohort at the time did not define their recorded
# `flag_gems.<name>` entry point. That premise is stale: the kernels are now
# generated from the pinned cohort (FlagGems master @ 5a58df410,
# 5.4.0rc2.post1+g5a58df410), and each of these names resolves at package level
# there. Re-measured on MetaX C550, one input set built on the host and moved
# with `.to("flagos:0")` so both arms see identical values, once per arm with
# `FLAGOS_OP_<op>` forcing the route:
#
#   - all 8 compute the CPU answer on the `flaggems` (flagos_python) route;
#   - 2 of them, `special_bessel_j1` and `linalg_ldl_solve`, FAIL on the cuda
#     boxing route today (maca raises cudaErrorMemoryValueTooLarge, and its
#     cusolverDnXsytrs_bufferSize is unavailable), so the FlagGems route is a
#     fix rather than a preference.
#
# That first probe fed `torch.rand(4, 4) + 0.5`, i.e. strictly positive inputs,
# which is why `igammac_` was promoted and then withdrawn: re-measuring it with
# `tests/manual/flaggems_overload_survey.py` (whose floating profiles are built
# with `torch.randn`, so arguments go negative) and with a direct A/B probe over
# `torch.randn(8, 8)` inputs showed the gems kernel returns finite values where
# ATen's CPU and the cuda boxing route both return NaN -- `a=-1.1524, b=+0.9200`
# gives `0.0275` against NaN, `a=+0.8487, b=-1.4782` and `a=+0.3223, b=-1.6293`
# give `1.0` against NaN, 36 of 64 elements disagreed in that probe and the
# whole of the disagreement was one-sided (no input produced NaN on the device
# that was finite on the host). `igammac_` therefore stays on the boxing kernel,
# which reproduces the CPU NaN mask exactly (51/51 NaN, 0 one-sided). The
# remaining 7 keep their promotion, with one recorded gap: `special_bessel_j1`
# returns the input dtype for integral inputs (`torch.special.bessel_j1` on an
# int64/bool tensor yields int64/bool zeros) where ATen promotes to float32, so
# it is correct for the floating dtypes it exists for and silently wrong-typed
# for the rest. It is still kept on the FlagGems route because the alternative
# is not a correct route but no route at all -- the boxing kernel raises
# `mcErrorMemoryValueTooLarge` for every dtype.
#
# Only the 7 measured-good ops are listed, and only for MetaX. The group (5)
# hold stays in codegen_ops.py for every other platform: the names resolve
# against the generation cohort, not against the FlagGems build a platform
# happens to install, and DCU/Ascend consume vendor-provided FlagGems rather
# than the pinned one. `max_unpool3d` also stays held -- its gems signature is
# `(input, indices, kernel_size, stride, padding, output_size)` while ATen's is
# `(self, indices, output_size, stride, padding)`, so the generated call passes
# the output size as the kernel size and returns a (1,2,6,6,6) tensor where the
# boxing route returns (1,2,4,4,4). `cudnn_batch_norm_backward` was not
# re-measured. See boxing_measured_python_ops().
METAX_FLAGGEMS_MEASURED = {
    "_embedding_bag_per_sample_weights_backward",
    "_native_batch_norm_legit_functional",
    "binary_cross_entropy_with_logits",
    "linalg_ldl_solve",
    "special_bessel_j1",
    "unsqueeze",
    "unsqueeze_",
}

# The hand-registered composite (EXTRA_ROUTED), on the platform that measured
# it. Unlike METAX_FLAGGEMS_MEASURED these ops are not held from the shared
# ceiling -- they are not in it at all, because they are not leaves any
# generated kernel calls. The route is a whole-op override in
# csrc/aten/sdp_choice_stub.cc, and MetaX is the only platform it has been run
# on: measured on Qwen-Image-2512's joint attention (bf16, mask=None,
# q(1,24,4114,128), 120 calls/step) as 78.49 s against 184.02 s for the boxing
# route over a 50-step denoise loop, against a route-off control pair that is
# bit-identical (0.0000/255) and a route-on image delta of 2.0461/255 -- the
# same order as the 1.2575/255 the flagos wheel already differs from vendor
# torch by. Every other platform keeps the boxing route, which is what the stub
# does when the conf does not name FlagGems. See the stub's header for why the
# composite has to be overridden whole rather than routed leaf by leaf.
METAX_COMPOSITE_FLAGGEMS = {"scaled_dot_product_attention"}

# The same override on the second platform that has now measured it, and the
# first whose call carries a mask. DCU's Qwen-Image-2.1 prefill hands
# scaled_dot_product_attention a `(1,1,1,4122)` all-true bool mask -- the joint
# sequence's `key_valid`, which build_token_metadata fills with torch.ones --
# and a mask, any mask, sent the composite to its math decomposition, which
# materializes the fp32 score matrix (32 x 4096 x 4122 x 4 B, 2.16 GB per call,
# 32 calls per forward). csrc/aten/sdp_choice_stub.cc now admits that row -- a
# 4-D bool mask with one entry per key, which the caller's shape pins to
# (1,1,1,KV) -- and converts it to the fp32 additive the kernel reads, which is
# what lets this route reach the model's largest attention. A mask of any other
# shape still keeps the boxing route.
# See the DCU entry in docs/reference/operator-support.md for the measurement and
# for what is still unmeasured on the shapes this route does not take.
DCU_COMPOSITE_FLAGGEMS = {"scaled_dot_product_attention"}

# FlagGems Python routes a native-kernel vendor measured on top of the shared
# coverage ceiling. The native twin of boxing_measured_python_ops(): same rule
# -- a cohort claim only the platform holding the matching FlagGems build can
# discharge -- for a platform whose routes are read from VENDORS rather than
# BOXING_PLATFORMS.
#
# Kunlunxin is empty, and stays empty until a FlagGems build exists for this
# SDK at all. The reason is not "not measured yet" but "not measurable today":
# the vendor torch shim on this SDK answers torch.cuda for flag_gems' runtime
# device probe while the tensors dispatch on PrivateUse1, so every FlagGems
# Python route dies at its first op regardless of what a conf claims, and no
# FlagGems liboperators.so exists for the XPU SDK either (see
# FLAGGEMS_CPP_PLATFORMS). Both are stated on the vendor side of the fleet
# already -- see torch_fl/accelerator/cuda/_cuda_compat.py's _ALIGNED_VENDORS
# and the platform table's FLAGOS_BUILD_FLAGGEMS_CPP pin.
#
# Spelled out rather than left implicit so the extension point is obvious when
# that changes: adding an op here routes it to `flaggems` in
# backends_kunlunxin.conf even though kunlunxin is not in
# FLAGGEMS_PYTHON_PLATFORMS, which is exactly the "route on measured evidence,
# not on the shared ceiling" contract the other measured sets follow. The op
# must also gain an m.impl() line in kunlunxin_register.inc (regenerate with
# scripts/codegen/codegen_kunlunxin.py), or the route names an empty slot.
KUNLUNXIN_FLAGGEMS_MEASURED = set()

NATIVE_FLAGGEMS_MEASURED = {
    "kunlunxin": KUNLUNXIN_FLAGGEMS_MEASURED,
}

# FlagGems coverage that arrived from a discovery widening rather than from a
# measurement on the native-kernel vendors, and the vendors that therefore keep
# the routes their own last FlagGems run measured.
#
# FLAGGEMS_PYTHON_OPS is discovered from the installed flag_gems package and is
# the ceiling every conf is intersected with (codegen_ops.py's
# render_flaggems_coverage writes it), so re-running the discovery against a
# newer FlagGems widens what *every* platform may route to that path. A conf is
# what a run executes, though, and a route is only worth the hardware measured
# behind it, so an op the discovery has just handed to the Python path must not
# move a vendor off its own kernel before that vendor's FlagGems suite has run
# against the new cohort. It is the register-then-measure loop NATIVE_TRITON_GAPS
# documents, entered from the other end: there an op joins the gaps once CI
# fails, here it stays on the vendor kernel until CI runs at all.
#
# The 2026-09-15 run (FlagGems master 5a58df410, whose entry points now resolve
# at package level as `flag_gems.<name>` instead of through `flag_gems.ops.*`)
# took the ceiling from 482 to 639 ops. MetaX measured the widened set --
# its FlagGems suite routes and executes 592 of them on C550 -- and so takes the
# ceiling whole. MUSA, GCU and Ascend have not re-run,
# and their shipped confs were generated against the 482-op cohort, so their
# routes stay where those runs left them.
#
# Deleting a vendor from FLAGGEMS_PENDING_NATIVE_VENDORS once its suite passes
# against the widened cohort is the whole promotion: rerun
# scripts/gen_vendor_confs.py and every op that vendor registers moves to the
# FlagGems path in one step. Deleting an op from the set instead is for the case
# where a vendor has measured that one op alone.
#
# The per-vendor register generators read this set too
# (codegen_gcu_flaggems.py, codegen_musa_flaggems.py). They emit the m.impl()
# lines that give the conf's `flaggems` routes a registered slot, so an op this
# set keeps off the conf must not be registered there either: registering it
# would claim the op on PrivateUse1 for a backend the conf never dispatches to,
# and the vendor kernel that is supposed to serve it would stop being reached.
FLAGGEMS_PENDING_NATIVE_VENDORS = ("ascend", "gcu", "musa")
FLAGGEMS_PENDING_NATIVE_OPS = frozenset({
    "_adaptive_avg_pool2d_backward", "_adaptive_avg_pool3d_backward",
    "_batch_norm_with_update_functional", "_cdist_forward",
    "_cholesky_solve_helper", "_compute_linear_combination",
    "_compute_linear_combination.out", "_conj_copy.out",
    "_convert_weight_to_int4pack", "_dirichlet_grad",
    "_fake_quantize_learnable_per_channel_affine_backward",
    "_fake_quantize_learnable_per_tensor_affine",
    "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
    "_fused_moving_avg_obs_fq_helper", "_fused_rms_norm",
    "_fused_rms_norm_backward", "_native_batch_norm_legit",
    "_native_batch_norm_legit.no_stats",
    "_native_batch_norm_legit.no_stats_out", "_native_batch_norm_legit.out",
    "_native_batch_norm_legit_no_training", "_pdist_forward",
    "_scaled_dot_product_efficient_attention", "_sparse_semi_structured_addmm",
    "_upsample_bilinear2d_aa_backward", "_upsample_nearest_exact1d_backward",
    "_upsample_nearest_exact1d_backward.grad_input", "_weight_int8pack_mm",
    "acos_", "adaptive_avg_pool3d_backward.grad_input", "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward", "adaptive_max_pool3d", "addbmm", "addbmm_",
    "addmv_", "addr_", "alias", "alias_copy.out", "atan2_", "atanh_",
    "baddbmm_", "binary_cross_entropy_backward", "bitwise_right_shift.Tensor",
    "bitwise_right_shift_.Tensor", "blackman_window",
    "blackman_window.periodic", "cholesky_inverse", "cholesky_solve",
    "cholesky_solve.out", "conj_physical_", "convolution_overrideable",
    "convolution_overrideable.out", "cudnn_convolution_transpose", "cumsum_",
    "dist", "embedding_renorm_", "empty_permuted", "eq_.Scalar", "eq_.Tensor",
    "erfc", "erfc_", "fake_quantize_per_channel_affine_cachemask",
    "fake_quantize_per_channel_affine_cachemask.out", "grid_sampler_3d",
    "grid_sampler_3d_backward", "hardshrink", "hardshrink.out",
    "hardsigmoid_backward", "hardswish", "hardswish.out", "hardswish_backward",
    "hardtanh", "hardtanh.out", "hardtanh_", "hash_tensor", "heaviside",
    "heaviside_", "huber_loss", "huber_loss.out", "hypot_", "igamma",
    "igamma_", "index_fill.int_Scalar", "index_fill.int_Tensor",
    "index_fill_.int_Scalar", "index_fill_.int_Tensor", "isposinf",
    "le_.Scalar", "le_.Tensor", "leaky_relu_backward", "linalg_cross",
    "linalg_cross.out", "linalg_eig", "linalg_householder_product",
    "linalg_lstsq", "linalg_lu", "linalg_lu.out", "linalg_lu_factor_ex",
    "linalg_lu_factor_ex.out", "linalg_matrix_exp", "linalg_matrix_exp.out",
    "linalg_qr", "linalg_qr.out", "linalg_solve_triangular",
    "linalg_solve_triangular.out", "log_sigmoid_backward",
    "log_sigmoid_backward.grad_input", "log_sigmoid_forward", "logit_backward",
    "lu_unpack", "lu_unpack.out", "masked_scatter_backward",
    "max_pool2d_with_indices_backward", "max_pool3d_with_indices_backward",
    "miopen_batch_norm", "miopen_batch_norm_backward", "nan_to_num_", "nansum",
    "nansum.out", "ne_.Scalar", "ne_.Tensor", "nextafter.out",
    "nonzero_static", "ormqr", "polygamma", "polygamma.out", "polygamma_",
    "reflection_pad2d_backward", "replication_pad2d", "replication_pad2d.out",
    "replication_pad2d_backward", "replication_pad2d_backward.grad_input",
    "replication_pad3d_backward", "scatter_add", "sgn", "sgn.out", "sign",
    "sign.out", "slice.Tensor", "special_bessel_j0", "special_bessel_y0",
    "special_bessel_y1", "special_chebyshev_polynomial_w.out", "special_erfcx",
    "special_hermite_polynomial_h.n_scalar", "special_i0e.out", "special_i1e",
    "special_i1e.out", "special_legendre_polynomial_p", "special_log_ndtr",
    "special_ndtri", "special_shifted_chebyshev_polynomial_t", "take",
    "take.out", "view_as_complex", "xlogy_.Scalar_Other", "xlogy_.Tensor",
})  # fmt: skip

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

    An empty ``inc_path`` (a FlagGems-primary vendor with no native kernels, see
    VENDORS) claims nothing, the same as a file that is not there.
    """
    if not inc_path.name or not inc_path.is_file():
        return set()
    return set(re.findall(r'm\.impl\(\s*"([^"]+)"', inc_path.read_text()))


def vendor_native_ops(vendor: str) -> set:
    """Ops with a vendor-native kernel, read from that vendor's own `.inc`.

    Earlier this was recovered from the conf being rewritten, which needed a
    trailing `# <vendor>` annotation to survive the round trip. Reading the
    generated C++ instead removes that dependency: the kernel set cannot drift
    from what is compiled, and a conf edit cannot invent or lose a kernel.

    The annotation is still *emitted* -- it is what tells a reader (and
    FLAGOS_FORCE_BACKEND=vendor) that a kernel exists behind an op FlagGems
    currently wins. ESCAPE_NATIVE ops (a kernel with no `m.impl` of its own) are
    annotated by render(), not added here: this function is what each vendor's
    own generator subtracts from its registration set, so it must stay equal to
    the ops that generator actually emits an `m.impl` for.
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


def boxing_measured_python_ops(filename: str) -> set:
    """Extra FlagGems Python ops a boxing platform routes on measured evidence.

    These ops are outside the shared coverage ceiling (``FLAGGEMS_PYTHON_OPS``)
    because codegen_ops.py's ``flaggems_runtime_broken`` holds them for every
    platform -- a cohort claim that only the platform holding the matching
    FlagGems build can discharge. MetaX has, so it routes them; see
    METAX_FLAGGEMS_MEASURED for the measurement and for why the hold stays
    elsewhere. METAX_COMPOSITE_FLAGGEMS and DCU_COMPOSITE_FLAGGEMS are outside
    the ceiling for a different reason -- they are composites no coverage scan
    could see -- but they are measurements held the same way, each on its own
    platform.

    Returning them from here rather than widening the coverage set is what keeps
    a measurement taken on one platform from becoming a route on another: a
    platform not named below gains nothing from it.
    """
    if filename == "backends_metax.conf":
        return METAX_FLAGGEMS_MEASURED | METAX_COMPOSITE_FLAGGEMS
    if filename == "backends_dcu.conf":
        return DCU_COMPOSITE_FLAGGEMS
    return set()


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
    kernel stays reachable through the `# <vendor>` annotation and
    FLAGOS_FORCE_BACKEND=vendor.

    An op the vendor implements but FlagGems or TileOps wins gets a trailing
    `# <vendor>` annotation. That is what tells a reader a kernel exists behind
    the winning route, and what makes FLAGOS_FORCE_BACKEND=vendor able to move
    the op.
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


def render(
    platform: str,
    vendor: str,
    routes: dict,
    boxing: bool = False,
    escape: set = frozenset(),
) -> str:
    """Build the conf text for one platform.

    `platform` names the conf (backends_<platform>.conf); `vendor` is the key in
    the third slot -- a native kernel name, or "cuda" for a boxing platform.
    `escape` is ESCAPE_NATIVE[vendor]: ops with a vendor kernel but no `m.impl`
    of their own, which route() cannot see. They are backfilled with the vendor
    annotation here, so the line still says a kernel exists behind the winning
    route -- and FLAGOS_FORCE_BACKEND=vendor can still reach it -- without
    widening `natives[vendor]`, which the routing priority and each vendor's own
    registration generator both depend on.
    """
    if escape:
        routes = {
            op: (f"{value}  # {vendor}" if op in escape else value)
            for op, value in routes.items()
        }
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

    # A FlagGems-primary platform with no vendor kernel library (iluvatar) has
    # no native `.inc`, so no op routes to the vendor slot. Its conf must not
    # advertise a backend nothing selects -- the legend and the routing-priority
    # line are the only places the vendor name appears, and both would otherwise
    # describe a route that does not exist. Every other vendor (and every boxing
    # platform) has at least one native route, so this only changes that conf.
    has_vendor_slot = boxing or counts.get(vendor, 0) > 0

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
        "#   tileops       TileOPs Triton shims (needs FLAGOS_BUILD_TILEOPS=ON + SM90)",
    ]
    if has_vendor_slot:
        lines.append(f"#   {vendor:<13} {vendor_desc}")
    if boxing:
        lines += [
            "#",
            f"# Routing priority: flaggems_cpp > flaggems > tileops > {vendor}.",
            "# Boxing reuses the CUDA kernels for every op, so nothing routes to",
            "# 'none' here.",
            "#",
        ]
    else:
        lines.append(
            "#   none          no accelerated impl here -> reaches cpu_fallback"
        )
        lines += [
            "#",
            "# Routing priority: flaggems_cpp > flaggems > tileops > "
            + (f"{vendor} > none." if has_vendor_slot else "none."),
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
        "# Collapse the whole table onto one backend family for A/B measurement",
        "# with FLAGOS_FORCE_BACKEND=flaggems|vendor|tileops. An op only moves if",
        "# that backend implements it -- known from the routed value plus its",
        "# `# <backend>` annotation. The rest are reported on stderr and stay as",
        "# configured, so FLAGOS_FORCE_BACKEND=vendor is partial by nature: a",
        "# vendor implements far fewer ops than FlagGems.",
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

    # A missing .inc is a silent disaster. registered_impls() returns the empty
    # set for a path that does not exist, so that vendor's whole coverage set
    # reads as empty, every one of its ops routes to `none`, and --check reports
    # success -- while the C++ build fails on the very same #include register.cc
    # emits. Fail loudly here instead: the VENDORS table names files that
    # register.cc actually includes, so a name with no file behind it is always
    # a bug, never a platform that happens to cover nothing. (A platform that
    # genuinely covers nothing belongs in BOXING_PLATFORMS or has no row at all.)
    #
    # The one documented exception is a FlagGems-primary vendor: iluvatar has no
    # vendor kernel library, so its native slot is the empty string -- an explicit
    # "no native kernels" declaration, not a file that should exist. Only the
    # extra (FlagGems) paths are real includes for such a vendor, and they are
    # checked like any other.
    for _vendor, (_native_inc, _extra_incs) in VENDORS.items():
        for _inc in (_native_inc, *_extra_incs):
            if not _inc:
                continue
            if not (CSRC_DIR / _inc).exists():
                raise SystemExit(
                    f"error: {_vendor}: {_inc} does not exist. Its coverage set "
                    "would read as empty and every op would route to `none`, "
                    "silently -- while register.cc fails to compile on the same "
                    "include. Generate it (see scripts/codegen/) or remove the "
                    "entry from VENDORS."
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

    # Same argument as the vendor-only widening above, for an op a hand-written
    # wrapper owns on every platform; see EXTRA_ROUTED.
    all_ops |= EXTRA_ROUTED

    out = {}
    for vendor in VENDORS:
        # A vendor build is FLAGOS_BUILD_FLAGGEMS=OFF, so the flaggems_cpp slot has no
        # kernel -- withhold the key and let those ops take the Python path.
        cpp_here = fg_cpp if vendor in FLAGGEMS_CPP_PLATFORMS else set()
        py_here = fg_py if vendor in FLAGGEMS_PYTHON_PLATFORMS else set()
        py_here = py_here - NATIVE_TRITON_GAPS.get(vendor, set())
        # Coverage the shared ceiling gained since this vendor's last FlagGems
        # run stays on the vendor kernel; see FLAGGEMS_PENDING_NATIVE_OPS.
        if vendor in FLAGGEMS_PENDING_NATIVE_VENDORS:
            py_here = py_here - FLAGGEMS_PENDING_NATIVE_OPS
        # Native kernel wins wherever both exist; FlagGems keeps the ops the
        # vendor has no kernel for. See NATIVE_KERNEL_PREFERRED.
        if vendor in NATIVE_KERNEL_PREFERRED:
            py_here = py_here - natives[vendor]
        # A measured FlagGems route this vendor owns beyond the shared ceiling
        # outranks both holds above -- it is a hardware measurement, and the
        # holds are heuristics about cohorts. See NATIVE_FLAGGEMS_MEASURED.
        # Intersected with fg_py so an op dropped from the FlagGems set cannot
        # linger here, the same guard boxing_cpp_ops() applies.
        py_here |= NATIVE_FLAGGEMS_MEASURED.get(vendor, set()) & fg_py
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
        out[vendor] = (
            render(
                vendor,
                vendor,
                routes,
                escape=ESCAPE_NATIVE.get(vendor, frozenset()),
            ),
            routes,
            vendor,
        )

    for platform, sparse_conf in BOXING_PLATFORMS.items():
        gaps = boxing_triton_gaps(conf_dir, sparse_conf, fg_py, fg_cpp)
        # A measured FlagGems route outranks the gap this file's own `cuda`
        # entries would otherwise recover: the op is spelled `cuda` today
        # because the shared coverage set holds it, not because this platform
        # measured a triton gap, so it has to come off `gaps` as well as onto
        # `py_here` -- route_boxing() checks the gap set first.
        measured = boxing_measured_python_ops(sparse_conf)
        gaps -= measured
        # Same gate as the native vendors below: a platform whose FlagGems
        # Python routes are not measured yet is not in FLAGGEMS_PYTHON_PLATFORMS,
        # so its conf routes nothing to flaggems and every covered op stays on
        # the boxing kernel. It is a no-op for the boxing platforms we build
        # today (metax/dcu/ppu, all in the set); it exists so that a new boxing
        # platform can join without claiming FlagGems coverage it has not run.
        # Routing the shared ceiling here instead would claim exactly that.
        py_here = (fg_py if platform in FLAGGEMS_PYTHON_PLATFORMS else set()) | measured
        # Only the C++-runtime conf routes to flaggems_cpp; elsewhere the slot is
        # unused and every C++ op is reached through the Python path instead.
        cpp_here = (
            boxing_cpp_ops(conf_dir, sparse_conf, fg_cpp)
            if platform in FLAGGEMS_CPP_PLATFORMS
            else set()
        )
        tileops_here = tileops if platform in TILEOPS_PLATFORMS else set()
        routes = {
            op: route_boxing(op, cpp_here, py_here, gaps, tileops_here)
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
