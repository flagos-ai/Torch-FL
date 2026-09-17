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
mean.dim dispatch tests

Verifies that torch.mean(dim=...):
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_mean_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend


DEVICE = "flagos:0"


def _run_mean_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,8,device='flagos:0'); "
        "torch.mean(a, dim=-1)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestMeanDimCorrectness:
    """torch.mean(dim=...) correctness on flagos device."""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((128, 256), -1),
            ((64, 64, 64), 1),
            ((32, 16), 0),
        ],
    )
    @pytest.mark.anyplatform
    def test_mean_shape(self, shape, dim):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.mean(a, dim=dim)
        expected_shape = list(shape)
        del expected_shape[dim if dim >= 0 else len(shape) + dim]
        assert list(out.shape) == expected_shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_mean_keepdim(self):
        torch.manual_seed(1)
        a = torch.randn(32, 64, device=DEVICE)
        out = torch.mean(a, dim=-1, keepdim=True)
        assert out.shape == (32, 1)

    @pytest.mark.anyplatform
    def test_mean_values(self):
        torch.manual_seed(2)
        a = torch.randn(16, 32, device=DEVICE)
        out = torch.mean(a, dim=-1)
        ref = a.cpu().mean(dim=-1)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_mean_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(3)
        a_cuda = torch.randn(64, 128, device="cuda:0")
        ref = torch.mean(a_cuda, dim=-1)
        a = a_cuda.to(DEVICE)
        out = torch.mean(a, dim=-1)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestMeanDimDispatch:
    """Verify dispatch routing for mean.dim op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_mean_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_mean__dim": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] mean.dim -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """mean.dim dispatches to whatever backend this platform's conf lists.

        FlagGems-first is the generated default, but it is not unconditional:
        MetaX keeps mean.dim on the cuda boxing kernel because gems' non-inner-dim
        path uses a CUDA context triton-metax rejects. Asserting the conf's own
        value keeps this test meaningful on every platform rather than pinning it
        to the one it was written on.
        """
        result = _run_mean_subprocess({"FLAGOS_LOG": "dispatch"})
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        expected = routed_backend("mean.dim")
        assert f"[flagos dispatch] mean.dim -> {expected}" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_mean_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_mean__dim": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] mean.dim -> cuda" in result.stderr


class TestMeanDimAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify mean.dim on ascend backend matches CPU reference."""
        result = _run_mean_subprocess({"FLAGOS_OP_mean__dim": "ascend"})
        assert result.returncode == 0
