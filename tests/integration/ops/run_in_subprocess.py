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

"""Run a backend case in a fresh interpreter.

Some graph compilers terminate the process on an unsupported graph instead of
returning an error. A child process keeps that failure attributable to the case
that built the graph and lets a caller fall back to the checked reference.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_python_file(
    path: str | Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``path`` and raise on a nonzero exit or a signal."""
    child_env = os.environ.copy()
    child_env.update(env or {})
    result = subprocess.run(
        [sys.executable, str(path)],
        env=child_env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stderr.count("Traceback") > 0:
        raise RuntimeError(
            f"{Path(path).name} exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_python_source(
    source: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``source`` in a fresh interpreter."""
    child_env = os.environ.copy()
    child_env.update(env or {})
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=child_env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stderr.count("Traceback") > 0:
        raise RuntimeError(
            f"program exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result
