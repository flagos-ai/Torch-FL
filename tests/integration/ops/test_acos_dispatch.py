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
acos dispatch tests

Verifies that torch.acos:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_acos_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_acos_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.rand(4,4,device='flagos:0') * 2 - 1; "
        "torch.acos(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestAcosCorrectness:
    """torch.acos correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_acos_shape(self, shape):
        torch.manual_seed(0)
        a = torch.rand(*shape, device=DEVICE) * 2 - 1
        out = torch.acos(a)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_acos_known_values(self):
        a = torch.tensor([1.0, 0.0, -1.0], device=DEVICE)
        out = torch.acos(a)
        ref = torch.acos(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_acos_random(self):
        torch.manual_seed(1)
        a = torch.rand(32, 32, device=DEVICE) * 2 - 1
        out = torch.acos(a)
        ref = torch.acos(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_acos_boundary(self):
        a = torch.tensor([-1.0, 1.0], device=DEVICE)
        out = torch.acos(a)
        ref = torch.acos(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.anyplatform
    def test_acos_dtypes(self, dtype):
        torch.manual_seed(2)
        a = torch.rand(16, device=DEVICE, dtype=dtype) * 2 - 1
        out = torch.acos(a)
        ref = torch.acos(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_acos_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(3)
        a_cuda = torch.rand(64, 64, device="cuda:0") * 2 - 1
        ref = torch.acos(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.acos(a)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestAcosDispatch:
    """Verify dispatch routing for acos op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_acos_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_acos": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] acos -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_acos_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_acos": "cuda",
            },
            check=False,
        )
        assert "[flagos dispatch] acos -> cuda" in result.stderr
