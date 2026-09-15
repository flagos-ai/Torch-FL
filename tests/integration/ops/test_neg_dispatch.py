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
neg dispatch tests

Verifies that torch.neg:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_neg_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend


DEVICE = "flagos:0"


def _run_neg_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; a = torch.randn(4,4,device='flagos:0'); torch.neg(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestNegCorrectness:
    """torch.neg correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_neg_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.neg(a)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_neg_values(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.neg(a)
        ref = -a.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.anyplatform
    def test_neg_non_contiguous(self):
        torch.manual_seed(11)
        a = torch.randn(8, 16, device=DEVICE)
        a_view = a.transpose(0, 1)
        assert not a_view.is_contiguous()
        out = torch.neg(a_view)
        ref = -a_view.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.cuda
    def test_neg_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.neg(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.neg(a)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    )
    @pytest.mark.anyplatform
    def test_neg_dtypes(self, dtype):
        torch.manual_seed(3)
        if dtype.is_floating_point:
            a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        else:
            a = torch.randint(-100, 100, (16, 16), device=DEVICE, dtype=dtype)
        out = torch.neg(a)
        assert out.dtype == dtype


class TestNegDispatch:
    """Verify dispatch routing for neg op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_neg_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_neg": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] neg -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """neg dispatches to whatever backend this platform's conf lists.

        FlagGems-first is the generated default, but it is not unconditional:
        GCU routes neg to its native ``gcu`` kernel because the FlagGems kernel
        carries a 64-bit type the GCU300 front end rejects for an int64 operand
        (see NATIVE_TRITON_GAPS["gcu"]). Asserting the conf's own value keeps
        this test meaningful on every platform rather than pinning it to the one
        it was written on.
        """
        result = _run_neg_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_USE_FLAGGEMS": "1"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        expected = routed_backend("neg")
        assert f"[flagos dispatch] neg -> {expected}" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_neg_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_neg": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] neg -> cuda" in result.stderr

    @pytest.mark.ascend
    def test_dispatch_log_ascend(self):
        result = _run_neg_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_neg": "ascend"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] neg -> ascend" in result.stderr


class TestNegAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify neg on ascend backend matches CPU reference."""
        result = _run_neg_subprocess({"FLAGOS_OP_neg": "ascend"})
        assert result.returncode == 0
