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

"""D-Robotics RDK BPU support: a torch.compile backend, not per-op kernels.

The BPU executes whole compiled graphs. An aten subgraph is exported to ONNX,
quantized to int8 (a precondition, not an optimization -- hbdk4 lowers float
conv to the CPU), compiled by hbdk4 into a `.hbm`, and executed through the
Horizon runtime. Ops outside a compilable partition keep running eagerly on the
CPU, so a missing hbdk4 costs performance and never correctness.

Usage:

    import torch, torch_fl
    compiled = torch.compile(model.eval(), backend="bpu")

`register()` is called by `torch_fl` when the build targets bpu, so importing
this package a second time is not required.
"""

from __future__ import annotations

from .backend import bpu_backend, register

__all__ = ["bpu_backend", "register"]
