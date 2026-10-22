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
torch.compile backend for flagos device.

Enables inductor-based kernel fusion for models running on the flagos device:
    model = torch.compile(model, backend="flagos")

The backend internally uses TorchInductor for graph optimization and kernel
fusion, with device context patched to target the flagos device. Generated
Triton kernels dispatch through the existing flagos routing infrastructure
(kFlagGems/kFlagGemsCpp/cuda boxing).

FlagTree compiles these kernels instead of OpenAI Triton when it is installed,
which requires no change here: FlagTree replaces the `triton` module at install
time. FLAGOS_USE_FLAGTREE=1 asserts that it really is the active Triton.
"""

from torch_fl.compile.inductor_backend import flagos_compile_backend

__all__ = ["flagos_compile_backend"]
