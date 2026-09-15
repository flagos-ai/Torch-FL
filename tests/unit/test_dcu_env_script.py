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

"""Static guards for the DCU environment script that provisions FlagTree.

``set_env_dcu.sh`` runs before every DCU job can do anything, so a defect in it
is not a test failure -- it is a job that produces no results at all. That is
what happened when the FlagTree/FlagGems provisioning landed: the triton
cleanup used ``while pip uninstall ...; do :; done``, and ``pip uninstall -y``
exits 0 even when it skips every named package, so the loop never terminated.
The job sat on that line with all output sent to ``/dev/null`` until the
60-minute job timeout cancelled it.

These checks are static on purpose: they must run on any host, without a DCU or
a network.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / ".github/scripts/set_env_dcu.sh"
_SCRIPT = _SCRIPT_PATH.read_text(encoding="utf-8")

# A `while <condition>; do ... done` preamble, across continuation lines.
_WHILE_RE = re.compile(r"\bwhile\b(?P<condition>.*?)\bdo\b", re.DOTALL)


def _while_conditions(script: str) -> list[str]:
    """Every `while` loop condition in the script, continuations joined."""
    return [
        " ".join(match.group("condition").split())
        for match in _WHILE_RE.finditer(script)
    ]


def test_script_exists():
    assert _SCRIPT_PATH.is_file(), _SCRIPT_PATH


def test_script_is_valid_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash is unavailable")
    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_while_loop_terminates_on_a_pip_command():
    """A `while pip uninstall` loop cannot terminate: pip exits 0 regardless.

    ``pip uninstall -y`` reports "WARNING: Skipping <pkg> as it is not
    installed" and exits 0, so keying the loop on the command itself spins
    forever once the package is gone. The condition has to test something that
    actually changes -- here, dist-info metadata still being present.
    """
    offenders = [c for c in _while_conditions(_SCRIPT) if "pip" in c]
    assert not offenders, (
        f"while loop keyed on a pip command never terminates; conditions: {offenders}"
    )


def test_triton_is_cleared_by_path_and_the_uninstall_is_bounded():
    """The copied vendor triton ships no dist-info, so it is removed by path.

    Without the path removal pip has nothing to act on and the flagtree install
    leaves two Tritons in the venv; without the dist-info bound the uninstall
    step is unbounded again.
    """
    assert 'rm -rf "$VENV_SITE/triton" "$VENV_SITE/triton_kernels"' in _SCRIPT
    assert 'compgen -G "$VENV_SITE/triton-*.dist-info"' in _SCRIPT
    assert 'compgen -G "$VENV_SITE/triton_kernels-*.dist-info"' in _SCRIPT


def test_uninstall_is_retried_at_most_once_per_present_metadata():
    """The loop body still runs pip, but only while metadata remains.

    Guards against 'fixing' the hang by dropping the uninstall entirely: a
    stale dist-info would then survive into the flagtree install.
    """
    body = _SCRIPT.split('while compgen -G "$VENV_SITE/triton-*.dist-info"', 1)
    assert len(body) == 2, "dist-info bound not found"
    loop = body[1].split("done", 1)[0]
    assert "pip uninstall" in loop
    assert "|| break" in loop
