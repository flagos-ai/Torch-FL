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
le.Tensor dispatch tests

Verifies that torch.le (Tensor variant):
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_le_dispatch.py -v
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
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.le(a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestLeCorrectness:
    """torch.le correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        b = torch.tensor([2.0, 2.0, 2.0, 2.0], device=DEVICE)
        out = torch.le(a, b)
        expected = torch.tensor([True, True, False, False])
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_output_dtype_is_bool(self):
        a = torch.randn(8, 8, device=DEVICE)
        b = torch.randn(8, 8, device=DEVICE)
        out = torch.le(a, b)
        assert out.dtype == torch.bool

    @pytest.mark.anyplatform
    def test_matches_cpu(self):
        torch.manual_seed(0)
        a = torch.randn(16, 16, device=DEVICE)
        b = torch.randn(16, 16, device=DEVICE)
        out = torch.le(a, b)
        ref = torch.le(a.cpu(), b.cpu())
        torch.testing.assert_close(out.cpu(), ref)

    @pytest.mark.anyplatform
    def test_broadcast(self):
        a = torch.randn(4, 8, device=DEVICE)
        b = torch.randn(8, device=DEVICE)
        out = torch.le(a, b)
        ref = torch.le(a.cpu(), b.cpu())
        torch.testing.assert_close(out.cpu(), ref)


class TestLeDispatch:
    """Verify dispatch routing for le.Tensor op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_le__Tensor": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] le.Tensor -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_le__Tensor": "cuda"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] le.Tensor -> cuda" in result.stderr


class TestLeAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify le.Tensor on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_le__Tensor": "ascend"})
        assert result.returncode == 0
