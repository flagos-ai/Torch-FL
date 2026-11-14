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

#pragma once

#include "python_op_caller.h"

namespace at::native::flagos {

bool HostProgramsEnabled();
std::string HostProgramKey(const char* name, const std::vector<c10::IValue>& args,
                           const std::vector<PyKwarg>& kwargs, int64_t outputs);
bool TryHostProgram(const char* name, const std::vector<c10::IValue>& args,
                    const std::vector<PyKwarg>& kwargs, int64_t outputs,
                    std::vector<at::Tensor>& result);

} // namespace at::native::flagos
