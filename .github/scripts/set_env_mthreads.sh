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

# Moore Threads MUSA environment for the common build/integration workflow.
#
# MUSA runs native mudnn operator kernels, not CUDA boxing: the toolkit ships no
# CUDA runtime and there is no vendor dispatch key to box into. mudnn links
# against musart only and pulls in no torch symbols, so this backend builds
# against a stock CPU PyTorch.
#
# The vendor torch_musa package present in the base image is deliberately never
# imported, linked, or copied into the isolated environment: its libtorch is a
# 2.9.1 build whose C++ object layout differs from 2.10 (sizeof(c10::MessageLogger)
# 408 -> 400), so mixing the two ABIs corrupts memory at runtime rather than
# failing to link. See docs/vendors/musa/installation.md.

# Thin wrapper: the parameterized entrypoint is set_env.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh" --platform musa
