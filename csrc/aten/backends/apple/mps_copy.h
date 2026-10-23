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
// The source's coordinates are built in an MPSGraph and gathered from its
// storage; a negative-axis coordinate is adjusted to start at that axis' last
// element. The whole storage is described to the graph, so the view's offset is
// applied without reading bytes outside it. A scalar copy is a Metal blit.
// Neither path moves tensor bytes through host memory.
//
// False for an empty tensor, a dtype without an MPSGraph representation, or
// memory outside the runtime's Metal allocations. The caller takes its existing
// fallback.
//
// MPSGraph scatter has not provided a reliable general write into a strided
// destination, so this header deliberately exposes only the gather direction.
bool CopyStridedToDense(const at::Tensor& src, const at::Tensor& dst);

} // namespace at::native::flagos::mps
