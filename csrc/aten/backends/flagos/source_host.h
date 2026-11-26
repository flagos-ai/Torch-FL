// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include "python_op_caller.h"

namespace at::native::flagos {

// ATen builds the derivative outside the backend kernel and excludes the
// Autograd keys while it redispatches. The compiled host may run in that
// region even if GradMode remains enabled for the caller.
inline bool NativeSourceCanRun() {
  return !c10::GradMode::is_enabled() ||
      c10::impl::tls_is_dispatch_keyset_excluded(c10::autograd_dispatch_keyset);
}

// Registration is per function; shape handling stays in the generated body.
using SourceHostEntry = bool (*)(const std::vector<c10::IValue>&, const std::vector<PyKwarg>&,
                                 std::vector<at::Tensor>&);
using SourceHostTensorEntry = bool (*)(c10::ArrayRef<c10::IValue>, c10::ArrayRef<PyKwarg>,
                                       at::Tensor&);

bool TrySourceHostTensor(const char* name, c10::ArrayRef<c10::IValue> args,
                         c10::ArrayRef<PyKwarg> kwargs, at::Tensor& result);

bool TrySourceHost(const char* name, const std::vector<c10::IValue>& args,
                   const std::vector<PyKwarg>& kwargs, int64_t outputs,
                   std::vector<at::Tensor>& result);

}  // namespace at::native::flagos
