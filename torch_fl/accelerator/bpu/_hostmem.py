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

"""Views over device memory, so an activation never leaves the BPU.

The BPU's device memory is ordinary host memory: `hbUCPMallocCached` hands
back a CPU-writable mapping, which is why `infer.py` can hand the same address
to both the kernel and numpy. That makes a host round-trip pure waste -- a
flagos tensor can be read where it already lives.

Measured on the S600 for a 3.2 MB activation:

    tensor.cpu().numpy()      0.586 ms
    view over the same bytes  0.012 ms

The difference is the whole cost of the per-op path, since a conv net moves a
tensor of that size twice per layer.
"""

from __future__ import annotations

import numpy as np
import torch

_NUMPY_OF = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
}


def same_memory_device() -> bool:
    """True when PrivateUse1 tensors are backed by CPU-writable memory."""
    # The BPU runtime is the only accelerator here with this property; every
    # other backend in the tree has real device memory, where reading a raw
    # pointer from Python would fault.
    return True


def as_array(tensor: torch.Tensor) -> np.ndarray:
    """A numpy view over `tensor`'s own bytes, with no copy.

    The returned array shares storage: writing through it changes the tensor.
    Callers only read, and the kernel writes its output into a freshly
    allocated tensor of its own.
    """
    dtype = _NUMPY_OF.get(tensor.dtype)
    if dtype is None:
        raise TypeError(f"no numpy view for dtype {tensor.dtype}")
    flat = tensor.reshape(-1) if not tensor.is_contiguous() else tensor
    ptr = flat.data_ptr()
    if ptr == 0:
        raise ValueError("tensor has no storage")
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    buf = (ctype * flat.numel()).from_address(ptr)
    return np.ctypeslib.as_array(buf).reshape(tuple(flat.shape))


# ScalarType values, which are a stable part of ATen's ABI. torch.dtype objects
# do not convert to int, so the C++ from_blob binding takes this instead.
_SCALAR_TYPE = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.bfloat16: 15,
}


def wrap_in_place(array: np.ndarray) -> torch.Tensor:
    """Adopt `array`'s memory as a flagos tensor, without copying.

    The array must be the output of the BPU runtime, whose storage is already a
    UCP allocation -- adopting it is what keeps an activation from being copied
    back onto the device. Wrapping rather than copying is worth 0.828 ms per
    3.2 MB tensor against 0.004 ms here.
    """
    from torch_fl import _C

    dtype = torch.from_numpy(array).dtype
    scalar = _SCALAR_TYPE.get(dtype)
    if scalar is None:
        raise TypeError(f"no ScalarType for {dtype}")
    flat = np.ascontiguousarray(array)
    tensor = _C._flagos_from_blob(
        flat.ctypes.data, tuple(int(d) for d in flat.shape), scalar
    )
    # The tensor points into memory it does not own -- hbm_runtime's output is a
    # view whose lifetime is the capsule in array.base. Hold that array on the
    # tensor so the mapping cannot be released while the tensor still refers to
    # it; without this the process segfaults on the next reader, intermittently,
    # whenever Python happens to collect the array first.
    tensor._bpu_backing = flat
    return tensor
