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

"""Apple Silicon (M-series) Metal integration.

There is no CUDA runtime and no vendor SDK on this platform: the device runtime
in ``csrc/runtime/accelerator/mps/`` talks to Metal directly, and
``torch_fl.flagos.Stream``/``Event`` dispatch here for the stream and event
wrappers ``mps_stream.py`` provides. Like MUSA's, this module holds only those
wrappers -- there is no ``_mps_compat.py``, because Metal needs no CUDA-name
shim.
"""
