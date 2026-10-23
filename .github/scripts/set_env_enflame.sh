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

# Enflame GCU environment for the common build/integration workflow.
#
# GCU uses the TopsRider runtime directly. The vendor torch-gcu package that
# may be present in the base image is intentionally not copied into the
# isolated environment: this backend builds against stock CPU PyTorch and
# links TopsRider's libtopsrt/libtopsaten libraries instead.
#
# The FlagGems Python path is provisioned here the same way set_env_musa.sh
# provisions it: the generated configs/backends_gcu.conf routes most overloads
# to a FlagGems Triton kernel and the rest to the native topsaten kernel, so the
# isolated venv needs a Triton build carrying the "enflame" backend (flagtree)
# plus FlagGems itself. See docs/vendors/gcu/flaggems-setup.md.

# Thin wrapper: the parameterized entrypoint is set_env.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh" --platform gcu
