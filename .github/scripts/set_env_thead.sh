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

# PPU (T-Head Jianwu ZW810E) environment bootstrap for CI.
#
# PPU is a CUDA-ABI boxing backend with its own FLAGOS_ACCELERATOR value: the toolchain
# is CUDA (PPU torch is a local USE_CUDA=1 build whose libtorch_cpu.so provides
# ~2092 undefined symbols of libtorch_fl.so), but the vendor is PPU, so the
# build bundles its libtorch into lib_ppu/. The stock CPU wheel's core libs must
# be replaced by the PPU build at import time. This script:
#   1. Validates PPU_SDK and the vendor torch assets.
#   2. Builds an isolated venv with stock CPU torch 2.10.0 (link target).
#   3. Installs the published FlagTree and FlagGems wheels into it.
#   4. Exports FLAGOS_ACCELERATOR=ppu + PPU_SDK + the two CUDA-assets kill switches.
#   5. build_ext --inplace, then bundles PPU core/CUDA/MKL .so into
#      torch_fl/lib_ppu/ via bundle_ppu_libtorch.sh (setup.py does not call it).
#
# Nothing here reads a host bind mount, so the environment is fully reproducible
# from the image plus this script's indexes.
#
# Core replacement itself is automatic at `import torch_fl` time, gated on
# lib_ppu/libtorch_cuda.so existing; the FLAGOS_ACCELERATOR value is what selects
# backends_ppu.conf. No mode env var is involved.


# Thin wrapper: the parameterized entrypoint is set_env.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh" --platform ppu
