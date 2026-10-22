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

import ctypes
import os


def _load_dll_libraries():
    """Load required DLL libraries on Windows."""
    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    if os.path.exists(lib_dir):
        for filename in os.listdir(lib_dir):
            if filename.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(lib_dir, filename))
                except OSError:
                    pass
