#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ascend NPU environment for the common build/integration workflow.
#
# Ascend is a standalone vendor backend (pure CANN ACLNN). Unlike CUDA it has no
# boxing layer, and unlike MetaX it does not shim a CUDA runtime. The wheel is
# built against a stock CPU PyTorch (2.10.0+cpu); the ACLNN shared libraries
# from the CANN toolkit are linked at runtime via LD_LIBRARY_PATH.
#
# FlagGems runs here on FlagTree, not on triton-ascend. triton-ascend 3.2.x was
# the previous provider and is gone: it lags the Triton APIs current FlagGems
# uses, its task-queue launch path calls at_npu::native::OpCommand (a torch_npu
# symbol torch_fl must not link), and the exact-string patch that used to strip
# those calls stopped matching on 3.2.2 and silently no-opped, which broke 11
# operator tests in run 34786387238. FlagTree's ascend3.5 wheel is the Triton
# 3.5 build for this backend, and torch_fl carries a torch_npu-free backend
# policy for it (torch_fl/compile/flagtree_ascend_policy.py), so no torch_npu
# appears anywhere in this environment. See docs/vendors/ascend/installation.md.

# Thin wrapper: the parameterized entrypoint is set_env.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh" --platform ascend
