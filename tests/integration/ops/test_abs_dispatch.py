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
abs dispatch tests

Verifies that torch.abs:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_abs_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend


DEVICE = "flagos:0"


def _run_abs_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; a = torch.randn(4,4,device='flagos:0'); torch.abs(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestAbsCorrectness:
    """torch.abs correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_abs_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.abs(a)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_abs_positive_values(self):
        a = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        out = torch.abs(a)
        ref = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_abs_negative_values(self):
        a = torch.tensor([-1.0, -2.0, -3.0], device=DEVICE)
        out = torch.abs(a)
        ref = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_abs_mixed(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.abs(a)
        ref = torch.abs(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_abs_zero(self):
        a = torch.zeros(16, device=DEVICE)
        out = torch.abs(a)
        assert (out.cpu() == 0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
    @pytest.mark.anyplatform
    def test_abs_dtypes(self, dtype):
        if dtype == torch.int32:
            a = torch.tensor([-1, -2, 3], device=DEVICE, dtype=dtype)
        else:
            a = torch.randn(16, device=DEVICE, dtype=dtype)
        out = torch.abs(a)
        ref = torch.abs(a.cpu())
        torch.testing.assert_close(out.cpu(), ref)

    @pytest.mark.cuda
    def test_abs_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.abs(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.abs(a)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-5, atol=1e-5)


class TestAbsDispatch:
    """Verify dispatch routing for abs op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_abs_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_abs": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] abs -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """abs dispatches to whatever backend this platform's conf lists.

        FlagGems-first is the generated default, but it is not unconditional:
        GCU routes abs to its native ``gcu`` kernel because the FlagGems kernel
        carries a 64-bit type the GCU300 front end rejects for an int64 operand
        (see NATIVE_TRITON_GAPS["gcu"]). Asserting the conf's own value keeps
        this test meaningful on every platform rather than pinning it to the one
        it was written on.
        """
        result = _run_abs_subprocess({"FLAGOS_LOG": "dispatch"})
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        expected = routed_backend("abs")
        assert f"[flagos dispatch] abs -> {expected}" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_abs_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_abs": "cuda",
            },
            check=False,
        )
        assert "[flagos dispatch] abs -> cuda" in result.stderr
