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
_softmax dispatch tests

Verifies that torch._softmax:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems (default) or cuda (via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_softmax_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend


DEVICE = "flagos:0"


def _run_softmax_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "x = torch.randn(4,8,device='flagos:0'); "
        "torch.softmax(x, dim=-1)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestSoftmaxCorrectness:
    """torch.softmax correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1, 10), (2, 8, 64)])
    @pytest.mark.anyplatform
    def test_softmax_shape(self, shape):
        torch.manual_seed(0)
        x = torch.randn(*shape, device=DEVICE)
        out = torch.softmax(x, dim=-1)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_softmax_sums_to_one(self):
        torch.manual_seed(1)
        x = torch.randn(32, 64, device=DEVICE)
        out = torch.softmax(x, dim=-1)
        sums = out.sum(dim=-1).cpu()
        torch.testing.assert_close(sums, torch.ones(32), rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_softmax_matches_cpu(self):
        torch.manual_seed(2)
        x_cpu = torch.randn(16, 32)
        ref = torch.softmax(x_cpu, dim=-1)
        x = x_cpu.to(DEVICE)
        out = torch.softmax(x, dim=-1)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_softmax_dim0(self):
        torch.manual_seed(3)
        x = torch.randn(8, 16, device=DEVICE)
        out = torch.softmax(x, dim=0)
        sums = out.sum(dim=0).cpu()
        torch.testing.assert_close(sums, torch.ones(16), rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_softmax_half(self):
        torch.manual_seed(4)
        x = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        out = torch.softmax(x, dim=-1)
        assert out.dtype == torch.float16
        sums = out.float().sum(dim=-1).cpu()
        torch.testing.assert_close(sums, torch.ones(8), rtol=1e-2, atol=1e-2)


class TestSoftmaxDispatch:
    """Verify C++ wrapper routes to the correct backend."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_softmax_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP__softmax": "flaggems_python"},
            check=False,
        )
        assert "[flagos dispatch] _softmax -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """_softmax dispatches to whatever backend this platform's conf lists.

        The conf name and the log name differ: the FlagGems C++ path is
        ``flaggems_cpp`` in the conf and ``-> flagos`` in the log, the Python path
        ``flaggems``/``-> flagos_python`` (csrc/aten/dispatcher.h, LogDispatch).
        routed_backend() owns that mapping, so this asserts the platform's real
        route instead of the one the test was written on.
        """
        result = _run_softmax_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_USE_FLAGGEMS": "1"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        expected = routed_backend("_softmax")
        assert f"[flagos dispatch] _softmax -> {expected}" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda(self):
        result = _run_softmax_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP__softmax": "cuda"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] _softmax -> cuda" in result.stderr


class TestSoftmaxAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify softmax on ascend backend matches CPU reference."""
        result = _run_softmax_subprocess({"FLAGOS_OP__softmax": "ascend"})
        assert result.returncode == 0
