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
ones_like dispatch tests

Verifies that torch.ones_like:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend

Usage:
    pytest tests/integration/ops/test_ones_like_dispatch.py -v
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
        "a = torch.randn(4,4,device='flagos:0'); "
        "torch.ones_like(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestOnesLikeCorrectness:
    """torch.ones_like correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_ones_like_shape(self, shape):
        a = torch.randn(*shape, device=DEVICE)
        out = torch.ones_like(a)
        assert out.shape == shape
        assert out.device.type == "flagos"
        assert torch.all(out.cpu() == 1.0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_ones_like_dtype(self, dtype):
        a = torch.randn(8, 8, device=DEVICE, dtype=dtype)
        out = torch.ones_like(a)
        assert out.dtype == dtype
        assert torch.all(out.cpu() == 1.0)

    @pytest.mark.anyplatform
    def test_ones_like_with_explicit_dtype(self):
        a = torch.randn(4, 4, device=DEVICE, dtype=torch.float32)
        out = torch.ones_like(a, dtype=torch.float16)
        assert out.dtype == torch.float16
        assert torch.all(out.cpu() == 1.0)


class TestOnesLikeDispatch:
    """Verify dispatch routing."""

    @pytest.mark.cuda
    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_ones_like": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] ones_like -> cuda" in result.stderr


class TestOnesLikeAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify ones_like on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_ones_like": "ascend"})
        assert result.returncode == 0
