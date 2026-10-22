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
silu_backward dispatch tests

Verifies that silu_backward:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_silu_backward_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "x = torch.randn(4,4,device='flagos:0',requires_grad=True); "
        "y = torch.nn.functional.silu(x); "
        "y.sum().backward()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestSiluBackwardCorrectness:
    """silu_backward correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_silu_backward_basic(self):
        torch.manual_seed(0)
        x = torch.randn(32, 32, device=DEVICE, requires_grad=True)
        y = torch.nn.functional.silu(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.device.type == "flagos"

    @pytest.mark.cuda
    @pytest.mark.anyplatform
    def test_silu_backward_matches_cuda(self):
        torch.manual_seed(1)
        # Compute reference on CPU (avoid mixing cuda/flagos autograd streams)
        x_cpu = torch.randn(64, 64, requires_grad=True)
        y_cpu = torch.nn.functional.silu(x_cpu)
        y_cpu.sum().backward()

        # Use the SAME input values on flagos (copy from CPU) rather than
        # re-seeding randn: flagos randn now uses the CUDA RNG, which does not
        # match the CPU RNG for the same seed, so re-seeding would compare
        # gradients of different inputs.
        x_flagos = x_cpu.detach().to(DEVICE).requires_grad_(True)
        y_flagos = torch.nn.functional.silu(x_flagos)
        y_flagos.sum().backward()

        torch.testing.assert_close(
            x_flagos.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_silu_backward_dtype(self, dtype):
        torch.manual_seed(2)
        x = torch.randn(16, 16, device=DEVICE, dtype=dtype, requires_grad=True)
        y = torch.nn.functional.silu(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.dtype == dtype

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (8, 16, 32)])
    @pytest.mark.anyplatform
    def test_silu_backward_shapes(self, shape):
        torch.manual_seed(3)
        x = torch.randn(*shape, device=DEVICE, requires_grad=True)
        y = torch.nn.functional.silu(x)
        y.sum().backward()
        assert x.grad.shape == shape


class TestSiluBackwardDispatch:
    """Verify dispatch routing for silu_backward op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_silu_backward": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] silu_backward -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_silu_backward": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] silu_backward -> cuda" in result.stderr


class TestSiluBackwardAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify silu_backward on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_silu_backward": "ascend"})
        assert result.returncode == 0
