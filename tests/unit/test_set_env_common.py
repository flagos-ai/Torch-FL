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

"""The set_env_*.sh scripts share one helper library.

`pip_retry`, `flag_gems_installed`, `install_flag_gems`, `strip_vendor_paths`
and `venv_is_usable` used to be copied into every platform script -- `pip_retry`
in five slightly different spellings -- so a fix had to be applied seven times
and was easy to miss. They now live in `.github/scripts/lib/set_env_common.sh`,
which every script sources. This keeps it that way: the library defines them,
no script redefines them, and every script sources the library.

Pure text: the scripts cannot be executed here (they provision vendor
environments), but the wiring is fully visible in their source.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
LIB = SCRIPTS_DIR / "lib" / "set_env_common.sh"

SHARED_FUNCS = [
    "pip_retry",
    "flag_gems_installed",
    "install_flag_gems",
    "strip_vendor_paths",
    "venv_is_usable",
]

SOURCE_LINE = 'source "${REPO_ROOT}/.github/scripts/lib/set_env_common.sh"'


def _scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("set_env_*.sh"))


def test_the_library_defines_every_shared_function():
    text = LIB.read_text(encoding="utf-8")
    for fn in SHARED_FUNCS:
        assert re.search(rf"^{fn}\(\) \{{", text, re.M), fn


def test_no_script_redefines_a_shared_function():
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        for fn in SHARED_FUNCS:
            assert not re.search(rf"^{fn}\(\) \{{", text, re.M), f"{script.name}: {fn}"


def test_every_script_sources_the_library():
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        assert SOURCE_LINE in text, script.name


def test_the_cuda_script_still_selects_its_interpreter_per_call():
    """CUDA drives several interpreters, so it must set PIP_RETRY_PYTHON.

    The shared pip_retry defaults to $VENV_PYTHON; CUDA's bootstrap/vendor steps
    need a different one, so every call has to name it or the wrong interpreter
    is used silently.
    """
    text = (SCRIPTS_DIR / "set_env_cuda.sh").read_text(encoding="utf-8")
    calls = re.findall(r"pip_retry ", text)
    prefixed = re.findall(r'PIP_RETRY_PYTHON="\$[A-Za-z_]+" pip_retry ', text)
    assert len(calls) == len(prefixed) == 7, (len(calls), len(prefixed))
