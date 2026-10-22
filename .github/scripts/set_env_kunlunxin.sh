#!/usr/bin/env bash
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

# Kunlun P800 XPU environment for the common build/integration workflow.
#
# Kunlunxin is a native PrivateUse1 backend: the port's own runtime sources
# under csrc/runtime/accelerator/kunlunxin/ call the vendor libxpurt runtime
# for device discovery, allocation, copies and streams, and the compute
# kernels come from the vendor xdnn op library behind the native route key.
# There is no CUDA boxing, so nothing here needs a CUDA toolkit.
#
# The vendor torch package that may be present in the base image is not
# copied into the isolated environment. The build never imports or links that
# torch: setup.py only locates the two xdnn files (libxdnn_pytorch.so and
# include/xdnn_pytorch/xdnn_pytorch.h) by presence, so a stock CPU PyTorch
# wheel is enough to build and run the backend.
#
# The isolated venv needs a Triton build carrying the "xpu" backend (flagtree)
# for the XPU kernels. Kunlunxin is not yet in the FlagGems Python platform
# allowlist, so the generated configs/backends_kunlunxin.conf routes nothing
# to FlagGems; see docs/vendors/kunlunxin/installation.md.

# Thin wrapper: the parameterized entrypoint is set_env.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh" --platform kunlunxin
