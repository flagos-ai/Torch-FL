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
Triton kernels used by tests/integration/test_compile.py.

They live in their own module because `triton.jit` compiles a kernel from its own
source text, read back from the file by line number. A kernel therefore has to be
a real module-level function: defining one inside a test body (or via `exec`)
either hides the `tl` import from the generated kernel or makes Triton read the
wrong lines out of the file.
"""

import triton
import triton.language as tl


@triton.jit
def masked_byte_row_load(
    in_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr
):
    """Masked 2-D strided load of an 8-bit dtype, widened to fp32 on store.

    This is the access pattern inductor emits when reducing over a boolean mask,
    e.g. `threshold_backward` for `relu`, and the one triton-ascend miscompiles
    without the workarounds in torch_fl/compile/triton_byte_loads.py.
    """
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    offsets = xindex + 128 * r0_index
    value = tl.load(in_ptr + offsets, xmask, other=0)
    tl.store(out_ptr + offsets, value.to(tl.float32), xmask)
