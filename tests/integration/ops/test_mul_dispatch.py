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
mul.Tensor dispatch tests

Verifies that torch.mul (Tensor variant):
  - produces correct results on flagos device
  - C++ wrapper routes to the backend the platform conf lists for the .Tensor
    overload (the vendor kernel on every current platform)
  - explicit overrides can still route to flaggems_python
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_mul_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend


DEVICE = "flagos:0"


def _run_mul_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.mul(a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestMulTensorCorrectness:
    """torch.mul correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_mul_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        b = torch.randn(*shape, device=DEVICE)
        out = torch.mul(a, b)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_mul_broadcast(self):
        torch.manual_seed(1)
        a = torch.randn(4, 8, device=DEVICE)
        b = torch.randn(8, device=DEVICE)
        out = torch.mul(a, b)
        ref = a.cpu() * b.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_mul_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        b_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.mul(a_cuda, b_cuda)
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.mul(a, b)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestMulTensorDispatch:
    """Verify dispatch routing for mul.Tensor op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_mul_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_mul__Tensor": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] mul.Tensor -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """mul.Tensor dispatches to whatever backend this platform's conf lists.

        ``flagos_python`` on CUDA/DCU/PPU and ``musa`` on MUSA: FlagGems has no
        Triton kernel for the .Tensor overload, so each platform's generated conf
        keeps it on its own vendor kernel. Reading the conf keeps the assertion
        true on all of them.
        """
        result = _run_mul_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_USE_FLAGGEMS": "1"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        expected = routed_backend("mul.Tensor")
        assert f"[flagos dispatch] mul.Tensor -> {expected}" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_mul_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mul__Tensor": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] mul.Tensor -> cuda" in result.stderr


class TestMulTensorAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify mul.Tensor on ascend backend matches CPU reference."""
        result = _run_mul_subprocess({"FLAGOS_OP_mul__Tensor": "ascend"})
        assert result.returncode == 0
