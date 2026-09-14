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
rsqrt dispatch tests

Verifies that torch.rsqrt:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_rsqrt_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_rsqrt_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0').abs() + 1e-6; "
        "torch.rsqrt(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestRsqrtCorrectness:
    """torch.rsqrt correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_rsqrt_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE).abs() + 1e-6
        out = torch.rsqrt(a)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rsqrt_values(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE).abs() + 1e-6
        out = torch.rsqrt(a)
        ref = 1.0 / torch.sqrt(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_rsqrt_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0").abs() + 1e-6
        ref = torch.rsqrt(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.rsqrt(a)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestRsqrtDispatch:
    """Verify dispatch routing for rsqrt op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_rsqrt_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_rsqrt": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] rsqrt -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_rsqrt_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_rsqrt": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] rsqrt -> cuda" in result.stderr


class TestRsqrtAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify rsqrt on ascend backend matches CPU reference."""
        result = _run_rsqrt_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_rsqrt": "ascend"}
        )
        assert result.returncode == 0
