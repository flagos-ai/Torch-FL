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

"""Build-time ABI smoke test for the CUDA base image.

Runs during `docker build`, so it must NOT touch the GPU (no CUDA context
init, no `torch.cuda.is_available()`). It only checks that the torch / triton /
flag_gems / flagcx packages import together against a single torch ABI -- the
cheapest way to catch the c10::MessageLogger-style undefined-symbol failures
that appear when a FlagGems C++ extension built against one torch version is
loaded into another. GPU availability is verified at CI runtime instead.

Override the expected versions via env vars when bumping the torch line
(see docker/cuda/Dockerfile ARGs).
"""

import os
import sys

import torch

EXPECTED_TORCH = os.environ.get("VERIFY_TORCH_VERSION", "2.9.0")
EXPECTED_CUDA = os.environ.get("VERIFY_CUDA_VERSION", "13.0")

base_version = torch.__version__.split("+", 1)[0]
assert base_version == EXPECTED_TORCH, (
    f"torch base version mismatch: got {base_version}, expected {EXPECTED_TORCH}"
)
assert torch.version.cuda == EXPECTED_CUDA, (
    f"torch CUDA runtime mismatch: got {torch.version.cuda}, expected {EXPECTED_CUDA}"
)

# Importing flag_gems pulls liboperators.so / libtriton_jit.so, which were
# compiled against this image's torch. An undefined-symbol error here means
# the FlagGems C++ extension's ABI drifted from the active torch.
import flag_gems  # noqa: E402, F401

# Importing flagcx pulls flagcx._C, which needs libc10_cuda.so. A CPU-only
# torch would not ship it; a full GPU torch (this image) does.
import flagcx  # noqa: E402, F401

print(
    f"verify ok: torch={torch.__version__} cuda={torch.version.cuda} "
    f"python={sys.version.split()[0]}"
)
