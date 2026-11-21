# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the CMake migration error without a vendor SDK."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("setting", ["cache", "environment"])
def test_cmake_rejects_the_retired_boxing_build(tmp_path, setting):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is unavailable")
    command = [cmake, "-S", str(ROOT), "-B", str(tmp_path)]
    env = {**os.environ, "FLAGOS_BUILD_BOXING": "0"}
    if setting == "cache":
        command.append("-DFLAGOS_BUILD_BOXING=ON")
    else:
        env["FLAGOS_BUILD_BOXING"] = "1"
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0
    assert "FLAGOS_BUILD_BOXING has been removed" in result.stderr
