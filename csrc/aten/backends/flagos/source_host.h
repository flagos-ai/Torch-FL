// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "python_op_caller.h"

namespace at::native::flagos {

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
