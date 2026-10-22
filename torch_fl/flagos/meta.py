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

# Meta functions for FlagOS device
# These are used by torch.compile and other meta-dispatch mechanisms

"""
Meta tensor implementations for shape inference during torch.compile tracing.

When torch.compile traces a model with flagos tensors, it needs to infer output
shapes without executing kernels. We register meta implementations that compute
output shapes/dtypes for ops that don't have default meta kernels.

Most ops inherit meta kernels from their CPU/CUDA implementations. We only need
to register meta kernels for:
1. Custom ops specific to flagos
2. Ops where the default meta kernel is incorrect for our backend
3. Ops that torch.compile explicitly requires but are missing

Start with a minimal set and expand as needed based on compile errors.
"""


# Meta kernels are registered via torch.library.impl
# Format: @torch.library.impl("aten::op_name", "Meta")

# Example: if we had a custom flagos-specific op
# @torch.library.impl("flagos::custom_op", "Meta")
# def custom_op_meta(input: Tensor, alpha: float) -> Tensor:
#     return torch.empty_like(input)


# Most standard ops (mm, add, conv, etc.) already have meta kernels from
# torch's default registrations. We inherit those automatically.

# Placeholder for future custom meta implementations as needed
# (torch.compile will error with specific op names if meta kernels are missing)
