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

``set_env_hygon.sh`` runs before every DCU job can do anything, so a defect in it
is not a test failure -- it is a job that produces no results at all. That is
what happened when the FlagTree/FlagGems provisioning landed: the triton
cleanup used ``while pip uninstall ...; do :; done``, and ``pip uninstall -y``
exits 0 even when it skips every named package, so the loop never terminated.
The job sat on that line with all output sent to ``/dev/null`` until the
60-minute job timeout cancelled it.

These checks use source inspection and a recording interpreter, without a DCU
or a network.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / ".github/scripts/hooks/set_env_hygon.sh"
)
_SCRIPT = _SCRIPT_PATH.read_text(encoding="utf-8")

# A `while <condition>; do ... done` preamble, across continuation lines.
_WHILE_RE = re.compile(
    r"^[ \t]*while[ \t]+(?P<condition>.*?)\bdo\b", re.MULTILINE | re.DOTALL
)


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


@pytest.mark.parametrize("usable", [True, False])
def test_flagtree_replacement_uninstalls_its_metadata_owner_once(tmp_path, usable):
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == "-" ]]; then cat >/dev/null; exit "$USABLE_STATUS"; fi\n'
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
    )
    python.chmod(0o755)
    log = tmp_path / "calls"
    start = _SCRIPT.index('if "$VENV_PYTHON" - "$FLAGTREE_VERSION"')
    end = _SCRIPT.index("\nHYGON_INDEX_URL=", start)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\ninstall_flagtree() { echo install >> "$CALL_LOG"; }\n'
            + _SCRIPT[start:end],
        ],
        env={
            **os.environ,
            "VENV_PYTHON": str(python),
            "USABLE_STATUS": "0" if usable else "1",
            "FLAGTREE_VERSION": "0.7.0+hcu3.6",
            "CALL_LOG": str(log),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines() if log.exists() else []
    assert calls == (
        []
        if usable
        else ["-m pip uninstall -y flagtree triton triton_kernels", "install"]
    )


@pytest.mark.parametrize("override", [False, True])
def test_vendor_python_survives_sdk_path_changes(tmp_path, override):
    image_python = tmp_path / "python"
    image_python.write_text("#!/bin/sh\nexit 0\n")
    image_python.chmod(0o755)
    explicit_python = tmp_path / "vendor-python"
    explicit_python.write_text("#!/bin/sh\nexit 0\n")
    explicit_python.chmod(0o755)
    capture = _SCRIPT[
        _SCRIPT.index("TORCH_FL_VENDOR_PYTHON=") : _SCRIPT.index("# Locate the DTK")
    ]
    select = _SCRIPT[
        _SCRIPT.index("select_vendor_python()") : _SCRIPT.index("VENDOR_INFO=")
    ]
    env = {**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin"}
    env.pop("TORCH_FL_VENDOR_PYTHON", None)
    if override:
        env["TORCH_FL_VENDOR_PYTHON"] = str(explicit_python)
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            "set -euo pipefail\n"
            + capture
            + "\nPATH=/unavailable-sdk-bin\n"
            + select
            + '\nprintf "%s\\n" "$VENDOR_PYTHON"',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(explicit_python if override else image_python)


@pytest.mark.parametrize(
    ("target", "vendor", "sdk", "accepted"),
    [
        ("2.10.0", "2.10.0", "2604", True),
        ("2.10.0+cpu", "2.10.0+das.opt1.dtk2604", "26.04.4", True),
        ("2.10.0", "2.13.0+dtk2610.test", "26.10", False),
        ("2.9.1", "2.10.0", "2604", False),
        ("2.10.0", "2.10.0", "", True),
        ("2.10.0", "2.10.0", "unverified", True),
    ],
)
def test_matching_cohort_restores_checks_and_records_only_verified_sdk(
    target, vendor, sdk, accepted
):
    start = _SCRIPT.index('CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:')
    end = _SCRIPT.index("# FlagGems C++ package is optional", start)
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + _SCRIPT[start:end]
            + '\nprintf "CHECKS=%s,%s SDK=%s\\n" "$FLAGOS_DCU_SKIP_RUNTIME_CHECK" '
            '"$FLAGOS_DCU_SKIP_LEGACY_SMOKE" "${FLAGOS_SDK_VERSION:-unknown}"',
        ],
        env={
            **os.environ,
            "TORCH_FL_CPU_TORCH_VERSION": target,
            "VENDOR_TORCH_VERSION": vendor,
            "VENDOR_TORCH_BASE_VERSION": vendor.split("+", 1)[0],
            "DTK_VERSION": sdk,
            "FLAGOS_SDK_VERSION": "stale",
            "FLAGOS_DCU_SKIP_RUNTIME_CHECK": "1",
            "FLAGOS_DCU_SKIP_LEGACY_SMOKE": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    if accepted:
        assert result.returncode == 0, result.stderr
        expected_sdk = sdk if sdk in ("2604", "26.04.4") else "unknown"
        assert f"CHECKS=0,0 SDK={expected_sdk}" in result.stdout
    else:
        assert result.returncode == 1
        assert "must match vendor Torch" in result.stderr
