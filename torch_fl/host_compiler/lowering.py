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

"""Shared allocation/copy lowering inside captured FlagGems host programs."""

import flag_gems
import torch


def clone(source, *, memory_format=torch.preserve_format):
    result = torch.empty_like(source, memory_format=memory_format)
    return flag_gems.copy_(result, source)


def to_copy(
    source,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    non_blocking=False,
    memory_format=None,
):
    if device is not None and torch.device(device) != source.device:
        raise ValueError("host capture cannot move data between devices")
    result = torch.empty_like(
        source,
        dtype=dtype,
        layout=layout,
        pin_memory=pin_memory,
        memory_format=memory_format or torch.preserve_format,
    )
    return flag_gems.copy_(result, source, non_blocking=non_blocking)


DECOMPOSITIONS = {
    torch.ops.aten.clone.default: clone,
    torch.ops.aten._to_copy.default: to_copy,
}
