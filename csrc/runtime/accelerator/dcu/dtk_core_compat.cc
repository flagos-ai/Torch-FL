// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdexcept>

// DTK 2604's libtorch_hip.so imports the generated at::_ops::*::call wrappers
// below from its forked libtorch_cpu.so. Official PyTorch does not contain the
// DTK-only schemas. Defining the symbols lets every upstream-schema HIP kernel
// register against the official core; calling a DTK-only schema fails cleanly
// rather than continuing with an invalid return value.
//
// Two families, both traced to the same DTK patch to ATen/autocast_mode.h,
// which adds the fused ops to the AT_FORALL_FP32 macro lists:
//   * native_fuse_* -- 16 imports of DTK's own libtorch_hip.so
//   * fuse_*        -- 16 imports of libtorch_fl.so, emitted because the plugin
//                      compiles against DTK's patched headers
// Both must be defined here: the second family makes `import torch_fl._C`
// itself fail with an undefined-symbol ImportError otherwise.
#define FLAGOS_DTK_COMPAT_STUB(identifier, symbol, op)                         \
  extern "C" [[noreturn]] __attribute__((visibility("default"), noinline))   \
  void identifier() __asm__(symbol);                                           \
  extern "C" [[noreturn]] __attribute__((visibility("default"), noinline))   \
  void identifier() {                                                          \
    throw std::runtime_error(                                                   \
        "DTK-only operator aten::" op                                          \
        " is unavailable with the official PyTorch core. Rebuild and install " \
        "with FLAGOS_DCU_VENDOR_CORE=1 to use DTK-private schemas.");          \
  }

FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_celu_gn,
    "_ZN2at4_ops19native_fuse_celu_gn4callERKNS_6TensorES4_S4_llddb",
    "native_fuse_celu_gn")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_gn_silu,
    "_ZN2at4_ops19native_fuse_gn_silu4callERKNS_6TensorES4_S4_lldb",
    "native_fuse_gn_silu")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_rmsnorm,
    "_ZN2at4_ops19native_fuse_rmsnorm4callERKNS_6TensorES4_db",
    "native_fuse_rmsnorm")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_ctc_loss,
    "_ZN2at4_ops20native_fuse_ctc_loss4callERKNS_6TensorES4_S4_S4_ldbbbb",
    "native_fuse_ctc_loss")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_bias_gelu,
    "_ZN2at4_ops21native_fuse_bias_gelu4callERKNS_6TensorES4_",
    "native_fuse_bias_gelu")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_group_norm,
    "_ZN2at4_ops22native_fuse_group_norm4callERKNS_6TensorES4_S4_lldb",
    "native_fuse_group_norm")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_layer_norm,
    "_ZN2at4_ops22native_fuse_layer_norm4callERKNS_6TensorERKSt8optionalIS2_ES8_db",
    "native_fuse_layer_norm")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_ln_dropout,
    "_ZN2at4_ops22native_fuse_ln_dropout4callERKNS_6TensorERKSt8optionalIS2_ES8_ddb",
    "native_fuse_ln_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_rn_dropout,
    "_ZN2at4_ops22native_fuse_rn_dropout4callERKNS_6TensorES4_ddb",
    "native_fuse_rn_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_bias_swiglu,
    "_ZN2at4_ops23native_fuse_bias_swiglu4callERKNS_6TensorES4_",
    "native_fuse_bias_swiglu")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_dropout_add,
    "_ZN2at4_ops23native_fuse_dropout_add4callERKNS_6TensorES4_db",
    "native_fuse_dropout_add")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_ln_add_dropout,
    "_ZN2at4_ops26native_fuse_ln_add_dropout4callERKNS_6TensorES4_RKSt8optionalIS2_ES8_ddb",
    "native_fuse_ln_add_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_rn_add_dropout,
    "_ZN2at4_ops26native_fuse_rn_add_dropout4callERKNS_6TensorES4_S4_ddb",
    "native_fuse_rn_add_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_bias_dropout_add,
    "_ZN2at4_ops28native_fuse_bias_dropout_add4callERKNS_6TensorES4_S4_db",
    "native_fuse_bias_dropout_add")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_ln_add_dropout_bias,
    "_ZN2at4_ops31native_fuse_ln_add_dropout_bias4callERKNS_6TensorES4_S4_RKSt8optionalIS2_ES8_ddb",
    "native_fuse_ln_add_dropout_bias")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_native_fuse_rn_add_dropout_bias,
    "_ZN2at4_ops31native_fuse_rn_add_dropout_bias4callERKNS_6TensorES4_S4_S4_ddb",
    "native_fuse_rn_add_dropout_bias")

// The public fuse_* family. Same schemas, one extra trailing bool arg in most
// signatures (the autocast wrapper's), imported by libtorch_fl.so.
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_celu_gn,
    "_ZN2at4_ops12fuse_celu_gn4callERKNS_6TensorES4_S4_llddbb",
    "fuse_celu_gn")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_gn_silu,
    "_ZN2at4_ops12fuse_gn_silu4callERKNS_6TensorES4_S4_lldbb",
    "fuse_gn_silu")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_rmsnorm,
    "_ZN2at4_ops12fuse_rmsnorm4callERKNS_6TensorES4_db",
    "fuse_rmsnorm")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_ctc_loss,
    "_ZN2at4_ops13fuse_ctc_loss4callERKNS_6TensorES4_S4_S4_ldbbbb",
    "fuse_ctc_loss")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_bias_gelu,
    "_ZN2at4_ops14fuse_bias_gelu4callERKNS_6TensorES4_",
    "fuse_bias_gelu")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_group_norm,
    "_ZN2at4_ops15fuse_group_norm4callERKNS_6TensorES4_S4_lldbb",
    "fuse_group_norm")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_layer_norm,
    "_ZN2at4_ops15fuse_layer_norm4callERKNS_6TensorERKSt8optionalIS2_ES8_dbb",
    "fuse_layer_norm")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_ln_dropout,
    "_ZN2at4_ops15fuse_ln_dropout4callERKNS_6TensorERKSt8optionalIS2_ES8_ddbb",
    "fuse_ln_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_rn_dropout,
    "_ZN2at4_ops15fuse_rn_dropout4callERKNS_6TensorES4_ddb",
    "fuse_rn_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_bias_swiglu,
    "_ZN2at4_ops16fuse_bias_swiglu4callERKNS_6TensorES4_",
    "fuse_bias_swiglu")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_dropout_add,
    "_ZN2at4_ops16fuse_dropout_add4callERKNS_6TensorES4_db",
    "fuse_dropout_add")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_ln_add_dropout,
    "_ZN2at4_ops19fuse_ln_add_dropout4callERKNS_6TensorES4_RKSt8optionalIS2_ES8_ddbb",
    "fuse_ln_add_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_rn_add_dropout,
    "_ZN2at4_ops19fuse_rn_add_dropout4callERKNS_6TensorES4_S4_ddb",
    "fuse_rn_add_dropout")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_bias_dropout_add,
    "_ZN2at4_ops21fuse_bias_dropout_add4callERKNS_6TensorES4_S4_db",
    "fuse_bias_dropout_add")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_ln_add_dropout_bias,
    "_ZN2at4_ops24fuse_ln_add_dropout_bias4callERKNS_6TensorES4_S4_RKSt8optionalIS2_ES8_ddbb",
    "fuse_ln_add_dropout_bias")
FLAGOS_DTK_COMPAT_STUB(
    flagos_dtk_fuse_rn_add_dropout_bias,
    "_ZN2at4_ops24fuse_rn_add_dropout_bias4callERKNS_6TensorES4_S4_S4_ddb",
    "fuse_rn_add_dropout_bias")

#undef FLAGOS_DTK_COMPAT_STUB
