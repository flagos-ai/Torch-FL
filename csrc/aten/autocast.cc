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

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace at::flagos {
namespace {

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
#define FLAGOS_AUTOCAST_LOWER_PRECISION_FP(...) \
  KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)
  AT_FORALL_LOWER_PRECISION_FP(FLAGOS_AUTOCAST_LOWER_PRECISION_FP)
#undef FLAGOS_AUTOCAST_LOWER_PRECISION_FP

#define FLAGOS_AUTOCAST_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)
  AT_FORALL_FP32(FLAGOS_AUTOCAST_FP32)
#undef FLAGOS_AUTOCAST_FP32

#define FLAGOS_AUTOCAST_FP32_SET_OPT_DTYPE(...) \
  KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)
  AT_FORALL_FP32_SET_OPT_DTYPE(FLAGOS_AUTOCAST_FP32_SET_OPT_DTYPE)
#undef FLAGOS_AUTOCAST_FP32_SET_OPT_DTYPE

  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE)

#define FLAGOS_AUTOCAST_PROMOTE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)
  AT_FORALL_PROMOTE(FLAGOS_AUTOCAST_PROMOTE)
#undef FLAGOS_AUTOCAST_PROMOTE

}

}  // namespace
}  // namespace at::flagos
