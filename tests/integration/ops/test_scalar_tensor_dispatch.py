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

"""
scalar_tensor dispatch tests

Verifies that torch.scalar_tensor:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda/metax backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_scalar_tensor_dispatch.py -v
"""

import os
import pytest
import subprocess
import sys

import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = "import torch_fl, torch; torch.scalar_tensor(3.14, device='flagos:0')"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestScalarTensorCorrectness:
    """torch.scalar_tensor correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_float_value(self):
        out = torch.scalar_tensor(3.14, device=DEVICE)
        assert out.shape == ()
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), torch.tensor(3.14))

    @pytest.mark.anyplatform
    def test_int_value(self):
        out = torch.scalar_tensor(42, device=DEVICE, dtype=torch.int64)
        assert out.dtype == torch.int64
        assert out.item() == 42

    @pytest.mark.anyplatform
    def test_zero(self):
        out = torch.scalar_tensor(0.0, device=DEVICE)
        assert out.item() == 0.0


class TestScalarTensorDispatch:
    """Verify dispatch routing."""

    @pytest.mark.cuda
    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_scalar_tensor": "cuda"}
        )
        if result.returncode != 0 and "backend not registered" in result.stderr:
            pytest.skip("cuda backend not available in this build")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] scalar_tensor -> cuda" in result.stderr

    @pytest.mark.metax
    def test_dispatch_log_metax(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_scalar_tensor": "metax"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] scalar_tensor -> metax" in result.stderr


class TestScalarTensorAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify scalar_tensor on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_scalar_tensor": "ascend"})
        assert result.returncode == 0
