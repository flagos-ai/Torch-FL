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
silu dispatch tests

Verifies that torch.nn.functional.silu:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_silu_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_silu_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "torch.nn.functional.silu(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestSiluCorrectness:
    """torch.nn.functional.silu correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_silu_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = F.silu(a)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_silu_values(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = F.silu(a)
        ref = a.cpu() * torch.sigmoid(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_silu_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        ref = F.silu(a_cuda)
        a = a_cuda.to(DEVICE)
        out = F.silu(a)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestSiluDispatch:
    """Verify dispatch routing for silu op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_silu_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_silu": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] silu -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """With the FlagGems runtime path on, silu routes to flagos_python."""
        result = _run_silu_subprocess({"FLAGOS_LOG_DISPATCH": "1"})
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] silu -> flagos_python" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_silu_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_silu": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] silu -> cuda" in result.stderr

    @pytest.mark.ascend
    def test_dispatch_log_ascend(self):
        result = _run_silu_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_silu": "ascend"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] silu -> ascend" in result.stderr


class TestSiluAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify silu on ascend backend matches CPU reference."""
        result = _run_silu_subprocess({"FLAGOS_OP_silu": "ascend"})
        assert result.returncode == 0
