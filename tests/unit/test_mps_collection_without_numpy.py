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

"""Keep shared operator collection independent of the MPS-only tests' imports.

The GCU and Kunlunxin lanes intentionally install no NumPy, but they select
``tests/integration/ops/`` and therefore import every MPS-only module there.
An MPS module with a module-level ``import numpy`` turned those sweeps into
collection errors before any operator ran. This test models that environment
in a subprocess, so the failure is caught in a lightweight unit run as well as
in the MPS lane itself.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPS_DIR = _REPO_ROOT / "tests/integration/ops"
_SHARED_MPS_MODULES = ("test_mps_erf.py", "test_mps_sqrt.py", "test_mps_tanh.py")


def test_shared_mps_modules_collect_without_numpy(tmp_path):
    """The no-NumPy shared sweep must reach collection, not report import errors."""
    package = tmp_path / "numpy"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'numpy'\")\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *_SHARED_MPS_MODULES,
            "--collect-only",
            "-q",
        ],
        cwd=_OPS_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    unittest_import_error = re.search(
        r"_FailedTest.*?ModuleNotFoundError: No module named 'numpy'",
        output,
        re.DOTALL,
    )
    assert result.returncode in (0, 5), output
    assert not unittest_import_error, output
