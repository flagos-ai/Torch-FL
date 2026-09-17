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

"""Load the MetaX cudart shim before torch is imported.

On MetaX hardware, PyTorch's .so files require CUDA runtime symbols
with @@libcudart.so.12 version tags.  MetaX's libsymbol_cu.so provides these
symbols but without version tags.  We build and load a shim library
(csrc/runtime/accelerator/metax/cudart_shim.c) that re-exports all needed symbols with the correct
version tags.
"""

import ctypes
import os
import subprocess

_loaded = False


def ensure_cudart_shim():
    """Build (if needed) and load the cudart shim into the process."""
    global _loaded
    if _loaded:
        return

    # Detect MetaX environment. Candidate order matches _metax_compat and
    # setup.py: the vendor's MACA_PATH, then MACA_HOME.
    metax_path = os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME")
    if not metax_path:
        for candidate in ["/opt/maca", "/opt/maca-3.3.0"]:
            if os.path.isdir(candidate):
                metax_path = candidate
                break
    if not metax_path or not os.path.isdir(metax_path):
        return  # Not a MetaX environment

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(pkg_dir))
    )  # project root
    csrc = os.path.join(base_dir, "csrc", "runtime", "accelerator", "metax")
    build_dir = os.path.join(base_dir, "build")

    shim_src = os.path.join(csrc, "cudart_shim.c")
    version_script = os.path.join(csrc, "libcudart.version")

    if not os.path.exists(shim_src):
        # Source not available (installed wheel without csrc)
        return

    os.makedirs(build_dir, exist_ok=True)
    shim_so = os.path.join(build_dir, "libcudart_shim.so")

    # Rebuild if source is newer
    inputs = [shim_src, version_script]
    if not os.path.exists(shim_so) or any(
        os.path.exists(s) and os.path.getmtime(s) > os.path.getmtime(shim_so)
        for s in inputs
    ):
        subprocess.check_call(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-o",
                shim_so,
                shim_src,
                f"-Wl,--version-script={version_script}",
                "-Wl,-soname,libcudart.so.12",
                "-ldl",
            ]
        )

    ctypes.CDLL(shim_so, mode=ctypes.RTLD_GLOBAL)
    _loaded = True
