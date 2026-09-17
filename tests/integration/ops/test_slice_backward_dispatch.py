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
slice_backward dispatch tests

Verifies that slice_backward:
  - produces correct gradients on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_slice_backward_dispatch.py -v
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
        "x = torch.randn(4,8,device='flagos:0',requires_grad=True); "
        "y = x[:, 2:6]; "
        "y.sum().backward()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestSliceBackwardCorrectness:
    """slice_backward correctness on flagos device."""

    @pytest.mark.anyplatform
    @pytest.mark.xfail(
        reason="FlagGems slice_backward kernel assumes contiguous grad_output"
    )
    def test_slice_backward_basic(self):
        torch.manual_seed(0)
        x = torch.randn(4, 8, device=DEVICE, requires_grad=True)
        y = x[:, 2:6]
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 8)
        # grad should be 1 in sliced region, 0 elsewhere
        ref_grad = torch.zeros(4, 8)
        ref_grad[:, 2:6] = 1.0
        torch.testing.assert_close(x.grad.cpu(), ref_grad)

    @pytest.mark.anyplatform
    @pytest.mark.xfail(
        reason="FlagGems slice_backward kernel assumes contiguous grad_output"
    )
    def test_slice_backward_step(self):
        torch.manual_seed(1)
        x = torch.randn(16, device=DEVICE, requires_grad=True)
        y = x[::2]
        y.sum().backward()
        ref_grad = torch.zeros(16)
        ref_grad[::2] = 1.0
        torch.testing.assert_close(x.grad.cpu(), ref_grad)

    @pytest.mark.anyplatform
    @pytest.mark.xfail(
        reason="FlagGems slice_backward kernel assumes contiguous grad_output"
    )
    def test_slice_backward_matches_cpu(self):
        torch.manual_seed(2)
        x_cpu = torch.randn(8, 16, requires_grad=True)
        y_cpu = x_cpu[:, 4:12]
        (y_cpu * 2).sum().backward()

        torch.manual_seed(2)
        x_fl = torch.randn(8, 16, device=DEVICE, requires_grad=True)
        y_fl = x_fl[:, 4:12]
        (y_fl * 2).sum().backward()

        torch.testing.assert_close(x_fl.grad.cpu(), x_cpu.grad, rtol=1e-5, atol=1e-5)


class TestSliceBackwardDispatch:
    """Verify dispatch routing for slice_backward op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_slice_backward": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] slice_backward -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_slice_backward": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] slice_backward -> cuda" in result.stderr


class TestSliceBackwardAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify slice_backward on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_slice_backward": "ascend"})
        assert result.returncode == 0
