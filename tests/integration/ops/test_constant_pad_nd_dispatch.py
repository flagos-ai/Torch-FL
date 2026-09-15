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
constant_pad_nd dispatch tests

Verifies that torch.nn.functional.pad (constant mode):
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_constant_pad_nd_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401

from backend_conf import routed_backend_or_none


DEVICE = "flagos:0"


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch, torch.nn.functional as F; "
        "a = torch.randn(2,3,4,device='flagos:0'); "
        "F.pad(a, (1,1), mode='constant', value=0.0)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestConstantPadNdCorrectness:
    """constant_pad_nd correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_pad_1d(self):
        torch.manual_seed(0)
        a = torch.randn(4, 8, device=DEVICE)
        out = F.pad(a, (2, 3), mode="constant", value=0.0)
        assert out.shape == (4, 13)
        assert out.device.type == "flagos"
        # padded regions should be zero
        assert torch.all(out.cpu()[:, :2] == 0.0)
        assert torch.all(out.cpu()[:, -3:] == 0.0)

    @pytest.mark.anyplatform
    def test_pad_2d(self):
        torch.manual_seed(1)
        a = torch.randn(2, 3, 4, device=DEVICE)
        out = F.pad(a, (1, 1, 2, 2), mode="constant", value=-1.0)
        assert out.shape == (2, 7, 6)
        # corners should be fill value
        assert out.cpu()[0, 0, 0].item() == -1.0

    @pytest.mark.anyplatform
    def test_pad_matches_cpu(self):
        torch.manual_seed(2)
        a_cpu = torch.randn(3, 5, 7)
        ref = F.pad(a_cpu, (1, 2, 0, 1), mode="constant", value=0.5)

        a_fl = a_cpu.to(DEVICE)
        out = F.pad(a_fl, (1, 2, 0, 1), mode="constant", value=0.5)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_pad_dtype(self, dtype):
        a = torch.randn(4, 4, device=DEVICE, dtype=dtype)
        out = F.pad(a, (1, 1), mode="constant", value=0.0)
        assert out.dtype == dtype
        assert out.shape == (4, 6)


class TestConstantPadNdDispatch:
    """Verify dispatch routing for constant_pad_nd op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        """The per-op override selects the FlagGems Python path for constant_pad_nd.

        Skipped where the conf routes constant_pad_nd to ``none``: the op is
        then not claimed on PrivateUse1 at all, so the call reaches cpu_fallback
        before the dispatcher and no override can show up in the log.
        """
        if routed_backend_or_none("constant_pad_nd") is None:
            pytest.skip("constant_pad_nd is routed to 'none' on this platform")
        result = _run_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_constant_pad_nd": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] constant_pad_nd -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_constant_pad_nd": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] constant_pad_nd -> cuda" in result.stderr


class TestConstantPadNdAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify constant_pad_nd on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_constant_pad_nd": "ascend"})
        assert result.returncode == 0
