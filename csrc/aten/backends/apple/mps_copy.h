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

#include <ATen/ATen.h>

namespace at::native::flagos::mps {

// Gathers a non-contiguous `src` into a dense, same-dtype `dst` on the current
// stream. Both tensors belong to this runtime's flagos device; their layouts
// may differ, but their shapes, dtype and element count must be equal.
//
// Uses PyTorch's Metal gather on the storage bits so it also handles
// subword and non-numerical dtypes without a value conversion. Returns false
// for layouts the shader cannot index or memory outside the Metal runtime.
bool CopyStridedToDense(const at::Tensor& src, const at::Tensor& dst);

} // namespace at::native::flagos::mps
