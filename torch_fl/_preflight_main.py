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

"""Entry point behind the wheel's ``torch-fl-<platform>`` console script.

setup.py ships one per build (``torch-fl-mps`` installs beside ``torch-fl-mps``
wheels, ``torch-fl-cuda`` beside ``torch-fl-2.10.0+cuda13.3`` wheels), so a
pipeline can name the tool after the wheel it is checking instead of the other
way round.

The tool itself is ``scripts/tools/torch-fl-preflight``, installed as data: it
must not import ``torch_fl`` (or torch), which is the whole point of inspecting
a wheel without loading it. A console script is a Python module, so this shim
is what makes it callable as one; it loads the script by path and hands over
the arguments unchanged.
"""

import runpy
import sys
from pathlib import Path


def _preflight_script():
    """The installed ``scripts/tools/torch-fl-preflight``.

    The wheel places it under a ``scripts`` data directory; a source checkout
    falls back to the same relative path.
    """
    package = Path(__file__).resolve().parent
    for candidate in package.parent.rglob("tools/torch-fl-preflight"):
        if candidate.is_file():
            return candidate
    return package.parent / "scripts" / "tools" / "torch-fl-preflight"


def main(argv=None):
    script = _preflight_script()
    if not script.is_file():
        sys.exit(f"torch-fl-preflight: tool script not found at {script}")
    sys.argv = [str(script)] + list(sys.argv[1:] if argv is None else argv)
    # The script exits the interpreter itself (0 ok, 1 findings, 2 usage).
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    main()
