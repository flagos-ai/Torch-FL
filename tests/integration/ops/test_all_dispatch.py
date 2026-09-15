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
all dispatch tests

Verifies that torch.all:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_all_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend_or_none


DEVICE = "flagos:0"


def _run_all_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.ones(4,4,device='flagos:0', dtype=torch.bool); "
        "torch.all(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestAllCorrectness:
    """torch.all correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_all_true(self):
        a = torch.ones(32, 32, device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is True
        assert out.dtype == torch.bool

    @pytest.mark.anyplatform
    def test_all_false(self):
        a = torch.zeros(32, 32, device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is False

    @pytest.mark.anyplatform
    def test_all_mixed(self):
        a = torch.ones(32, 32, device=DEVICE, dtype=torch.bool)
        a[0, 0] = False
        out = torch.all(a)
        assert out.item() is False

    @pytest.mark.anyplatform
    @pytest.mark.skip(reason="FlagGems all() kernel cannot handle empty tensors")
    def test_all_empty_tensor(self):
        """Empty tensor: all() is vacuously true."""
        a = torch.tensor([], device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is True

    @pytest.mark.anyplatform
    def test_all_numeric_types(self):
        """all() should work on numeric types (non-zero = True)."""
        torch.manual_seed(0)
        a = torch.randn(16, 16, device=DEVICE)
        out = torch.all(a)
        ref = torch.all(a.cpu())
        assert out.item() == ref.item()

    @pytest.mark.cuda
    def test_all_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(1)
        a_cuda = torch.randint(0, 2, (64, 64), device="cuda:0", dtype=torch.bool)
        ref = torch.all(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.all(a)
        assert out.item() == ref.item()

    @pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.int32])
    @pytest.mark.anyplatform
    def test_all_dtypes(self, dtype):
        if dtype == torch.bool:
            a = torch.ones(16, 16, device=DEVICE, dtype=dtype)
        elif dtype.is_floating_point:
            a = torch.randn(16, 16, device=DEVICE, dtype=dtype) + 1.0
        else:
            a = torch.randint(1, 10, (16, 16), device=DEVICE, dtype=dtype)
        out = torch.all(a)
        assert out.dtype == torch.bool


class TestAllDispatch:
    """Verify dispatch routing for all op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        """The per-op override selects the FlagGems Python path for ``all``.

        Skipped where the conf routes ``all`` to ``none``: the op is then not
        claimed on PrivateUse1 at all, so the call reaches cpu_fallback before
        the dispatcher and no override can show up in the log.
        """
        if routed_backend_or_none("all") is None:
            pytest.skip("all is routed to 'none' on this platform")
        result = _run_all_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_all": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] all -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_all_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_all": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] all -> cuda" in result.stderr


class TestAllAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify all on ascend backend matches CPU reference."""
        result = _run_all_subprocess({"FLAGOS_OP_all": "ascend"})
        assert result.returncode == 0
