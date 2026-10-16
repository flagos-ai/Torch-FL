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
HOOKS_DIR = SCRIPTS_DIR / "hooks"
RUNNER = SCRIPTS_DIR / "set_env.sh"

SHARED_FUNCS = [
    "pip_retry",
    "flag_gems_installed",
    "install_flag_gems",
    "strip_vendor_paths",
    "venv_is_usable",
]

#: The provisioning phases that were byte-for-byte the same modulo the pins.
SHARED_PHASES = [
    "install_cpu_torch",
    "install_flagtree",
    "install_flagcx",
    "build_flagos_inplace",
    "export_ci_env",
]

SOURCE_LINE = 'source "${REPO_ROOT}/.github/scripts/lib/set_env_common.sh"'


def _scripts() -> list[Path]:
    """The per-platform hook bodies (the old script bodies)."""
    return sorted(HOOKS_DIR.glob("set_env_*.sh"))


def test_the_library_defines_every_shared_function():
    text = LIB.read_text(encoding="utf-8")
    for fn in SHARED_FUNCS:
        assert re.search(rf"^{fn}\(\) \{{", text, re.M), fn


def test_no_script_redefines_a_shared_function():
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        for fn in SHARED_FUNCS:
            assert not re.search(rf"^{fn}\(\) \{{", text, re.M), f"{script.name}: {fn}"


def test_the_runner_sources_the_library():
    """One entrypoint sources the library; the hooks inherit it."""
    assert SOURCE_LINE in RUNNER.read_text(encoding="utf-8")


def test_the_cuda_script_still_selects_its_interpreter_per_call():
    """CUDA drives several interpreters, so it must set PIP_RETRY_PYTHON.

    The shared pip_retry defaults to $VENV_PYTHON; CUDA's bootstrap/vendor steps
    need a different one, so every call has to name it or the wrong interpreter
    is used silently.
    """
    text = (HOOKS_DIR / "set_env_cuda.sh").read_text(encoding="utf-8")
    calls = re.findall(r"pip_retry ", text)
    prefixed = re.findall(r'PIP_RETRY_PYTHON="\$[A-Za-z_]+" pip_retry ', text)
    assert calls and len(calls) == len(prefixed), (len(calls), len(prefixed))


def test_the_library_defines_every_shared_phase():
    text = LIB.read_text(encoding="utf-8")
    for fn in SHARED_PHASES:
        assert re.search(rf"^{fn}\(\) \{{", text, re.M), fn


def test_no_script_redefines_a_shared_phase():
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        for fn in SHARED_PHASES:
            assert not re.search(rf"^{fn}\(\) \{{", text, re.M), f"{script.name}: {fn}"


def test_every_script_calls_the_common_phases():
    """The phases every platform runs, called by name.

    install_flagcx is defined for the platforms whose hook installs a published
    FlagCX wheel, which is not every one of them, so it is not asserted here.
    install_flagtree is the other one: it exists for platforms whose operators
    come from a FlagTree/FlagGems cohort, and Apple Silicon has none -- PyPI
    publishes no `triton` wheel for macOS and there is no Metal FlagTree line --
    so MPS is the one hook that installs no Triton build. The absence is
    asserted directly rather than left implicit, so a hook that silently
    dropped its FlagTree install would not pass by being here.
    """
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        phases = ["install_cpu_torch", "build_flagos_inplace", "export_ci_env"]
        if script.name == "set_env_mps.sh":
            assert not re.search(r"^[ \t]*install_flagtree\b", text, re.M), (
                f"{script.name}: install_flagtree is not available on macOS"
            )
        else:
            phases.append("install_flagtree")
        for phase in phases:
            assert re.search(rf"^[ \t]*{phase}\b", text, re.M), (
                f"{script.name}: {phase}"
            )


def test_every_local_in_the_library_is_initialized():
    """Every `local` in the library names a value, not a slot.

    The scripts run under `set -u` (`.github/scripts/set_env.sh`), and bash 5 --
    what the runners use -- reports a bare `local` that has not been assigned
    yet as an unbound variable the moment it is expanded. `local a b="" c` is
    therefore safe for `b` and `c` and a landmine for `a`: the first PR to add
    `pip_index_host` declared `local arg prev="" url host` and assigned `url`
    inside a `case` branch, so the Ascend build job died with
    `line 54: url: unbound variable` (run 38491036055) while the same code
    passed on macOS, whose bash 3.2 expands an unassigned local to the empty
    string instead. A name the `for` statement assigns every iteration is the
    one shape a bare declaration may have.
    """
    text = LIB.read_text(encoding="utf-8")
    functions = re.split(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(\) \{", text, flags=re.M)
    assert functions, "no functions parsed"
    for name, body in zip(functions[1::2], functions[2::2], strict=True):
        # A function body ends at the first closing brace in column 0.
        body = body.split("\n}", 1)[0]
        for declaration in re.finditer(r"^[ \t]*local ([^\n]*)$", body, re.M):
            for word in declaration.group(1).split():
                variable = word.split("=", 1)[0]
                if "=" in word or not variable.isidentifier():
                    continue
                assert re.search(rf"\bfor\s+{variable}\s+in\b", body), (
                    f"{name}: `local {variable}` has no initializer and is not a "
                    f"for-loop variable; under `set -u` bash 5 fails on it"
                )
