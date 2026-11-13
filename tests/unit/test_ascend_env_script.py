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

"""Keep Ascend runtime initialization and package selection isolated in CI."""

import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
HOOK = (ROOT / ".github/scripts/hooks/set_env_ascend.sh").read_text()
COMMON = ROOT / ".github/scripts/lib/set_env_common.sh"


def test_ci_python_survives_a_new_workflow_shell(tmp_path):
    """Replay an image's PATH/PYTHONPATH reset, then the real hook's env export."""
    image_bin = tmp_path / "image/bin"
    ci_bin = tmp_path / "ci/bin"
    for directory, name in ((image_bin, "image"), (ci_bin, "ci")):
        directory.mkdir(parents=True)
        python = directory / "python"
        python.write_text(f"#!/bin/sh\nprintf '%s\\n' '{name}'\n")
        python.chmod(0o755)

    marker = tmp_path / "image-starts"
    startup = tmp_path / "bash_env.sh"
    startup.write_text(
        f"printf 'start\\n' >> {shlex.quote(str(marker))}\n"
        f"export PATH={shlex.quote(str(image_bin))}:/usr/bin:/bin\n"
        "export PYTHONPATH=/opt/flagtree:/image/sdk\n"
        "export ASCEND_HOME_PATH=/image/sdk\n"
        "export ASCEND_OPP_PATH=/image/sdk/opp\n"
        "export ASCEND_TOOLKIT_HOME=/image/sdk\n"
        "export TOOLCHAIN_HOME=/image/sdk/toolkit\n"
        "export ATB_HOME_PATH=/image/atb\n"
        "export CMAKE_PREFIX_PATH=/image/sdk/lib/cmake\n"
    )
    env_file = tmp_path / "github-env"
    path_file = tmp_path / "github-path"
    persist = HOOK.split('if [[ -n "${GITHUB_PATH:-}" ]]', 1)[1]
    persist = (
        'if [[ -n "${GITHUB_PATH:-}" ]]' + persist.split('\ncd "$REPO_ROOT"', 1)[0]
    )
    names = persist.split("export_ci_env ", 1)[1].splitlines()[0].split()
    env = {name: "" for name in names}
    env.update(
        PATH="/usr/bin:/bin",
        BASH_ENV=str(startup),
        GITHUB_ENV=str(env_file),
        GITHUB_PATH=str(path_file),
        VENV_ROOT=str(ci_bin.parent),
        VIRTUAL_ENV=str(ci_bin.parent),
        ASCEND_HOME="/image/sdk",
        LD_LIBRARY_PATH="/image/sdk/lib:/host/driver/lib",
    )
    isolation = HOOK.split("# --- Python isolation", 1)[1]
    isolation = (
        "# --- Python isolation" + isolation.split("# --- CANN toolkit root", 1)[0]
    )
    prepared = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            "set -euo pipefail\n"
            + isolation
            + '\nexport PATH="$VENV_ROOT/bin:$PATH"\n'
            + f"source {shlex.quote(str(COMMON))}\n"
            + persist,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not prepared.stderr
    assert marker.read_text().splitlines() == ["start"]
    next_env = {
        **env,
        **dict(line.split("=", 1) for line in env_file.read_text().splitlines()),
    }
    next_env["PATH"] = path_file.read_text().strip() + os.pathsep + next_env["PATH"]
    result = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -euo pipefail; python; test -z "$PYTHONPATH"; '
            'test "$PYTHONNOUSERSITE" = 1; test "$ASCEND_HOME" = /image/sdk; '
            'test "$ASCEND_HOME_PATH" = /image/sdk; '
            'test "$ASCEND_OPP_PATH" = /image/sdk/opp; '
            'test "$ASCEND_TOOLKIT_HOME" = /image/sdk; '
            'test "$TOOLCHAIN_HOME" = /image/sdk/toolkit; '
            'test "$ATB_HOME_PATH" = /image/atb; '
            'test "$CMAKE_PREFIX_PATH" = /image/sdk/lib/cmake; '
            'test "$LD_LIBRARY_PATH" = /image/sdk/lib:/host/driver/lib',
        ],
        env=next_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ci"
    assert marker.read_text().splitlines() == ["start"]


@pytest.mark.parametrize("broken_dependencies", [False, True])
def test_ascend_final_dependency_check_precedes_report_and_build(
    tmp_path, broken_dependencies
):
    script = HOOK.split(
        "# Validate the complete environment after copying and filling dependencies.\n",
        1,
    )[1]
    script = script.split("# --- Verify the isolation held", 1)[0]
    calls = tmp_path / "calls"
    python = tmp_path / "python"
    python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [ "$*" = "-m pip check" ]; then exit "$CHECK_STATUS"; fi\n'
    )
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script + '\necho build >> "$CALL_LOG"'],
        env={
            "PATH": "/usr/bin:/bin",
            "VENV_PYTHON": str(python),
            "REPO_ROOT": str(ROOT),
            "CALL_LOG": str(calls),
            "CHECK_STATUS": "47" if broken_dependencies else "0",
        },
        capture_output=True,
        text=True,
    )
    rows = calls.read_text().splitlines()
    assert rows[0] == "-m pip check"
    assert (result.returncode != 0) == broken_dependencies
    assert any("--report-environment" in row for row in rows) == (
        not broken_dependencies
    )
    assert ("build" in rows) == (not broken_dependencies)
