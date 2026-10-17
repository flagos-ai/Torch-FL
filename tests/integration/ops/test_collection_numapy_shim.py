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

"""Keep MPS-only tests from breaking a shared suite's collection."""

import os
import re
import subprocess
import sys


def test_mps_test_modules_collect_without_numpy(tmp_path):
    package = tmp_path / "numpy"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'numpy'\")\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    directory = os.path.dirname(__file__)
    modules = ["test_mps_erf.py", "test_mps_sqrt.py", "test_mps_tanh.py"]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *modules, "--collect-only", "-q"],
        cwd=directory,
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
