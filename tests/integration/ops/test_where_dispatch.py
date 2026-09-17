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
where.self dispatch tests

Verifies that torch.where (condition, self, other):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda/metax backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_where_dispatch.py -v
"""

import os
import pytest
import subprocess
import sys

import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend_or_none


DEVICE = "flagos:0"


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "cond = torch.tensor([True, False, True], device='flagos:0'); "
        "a = torch.tensor([1.0, 2.0, 3.0], device='flagos:0'); "
        "b = torch.tensor([4.0, 5.0, 6.0], device='flagos:0'); "
        "torch.where(cond, a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestWhereCorrectness:
    """torch.where correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_basic(self):
        cond = torch.tensor([True, False, True, False], device=DEVICE)
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        b = torch.tensor([5.0, 6.0, 7.0, 8.0], device=DEVICE)
        out = torch.where(cond, a, b)
        expected = torch.tensor([1.0, 6.0, 3.0, 8.0])
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_matches_cpu(self):
        x = torch.randn(32, 32, device=DEVICE)
        y = torch.randn(32, 32, device=DEVICE)
        cond = x > 0
        out = torch.where(cond, x, y)
        ref = torch.where(cond.cpu(), x.cpu(), y.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_broadcast(self):
        torch.manual_seed(1)
        cond = torch.tensor([[True], [False], [True]], device=DEVICE)
        a = torch.randn(3, 4, device=DEVICE)
        b = torch.randn(3, 4, device=DEVICE)
        out = torch.where(cond, a, b)
        ref = torch.where(cond.cpu(), a.cpu(), b.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_output_device(self):
        cond = torch.tensor([True, False], device=DEVICE)
        a = torch.tensor([1.0, 2.0], device=DEVICE)
        b = torch.tensor([3.0, 4.0], device=DEVICE)
        out = torch.where(cond, a, b)
        assert out.device.type == "flagos"


class TestWhereDispatch:
    """Verify dispatch routing for where.self op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        """The per-op override selects the FlagGems Python path for where.self.

        Skipped where the conf routes where.self to ``none``: the op is then not
        claimed on PrivateUse1 at all, so the call reaches cpu_fallback before
        the dispatcher and no override can show up in the log.
        """
        if routed_backend_or_none("where.self") is None:
            pytest.skip("where.self is routed to 'none' on this platform")
        result = _run_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_where__self": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] where.self -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """With the FlagGems runtime path available, where.self keeps its conf route.

        Skipped where the conf routes where.self away from FlagGems. GCU does:
        FlagGems' kernel compiles for float32/float16/int32, but an int64
        operand hits GCU300's "64-bit data type not supported" rejection, so the
        op stays unregistered and ATen's composite runs on the CPU instead of
        turning a working cpu_fallback into a hard error.
        """
        if routed_backend_or_none("where.self") != "flagos_python":
            pytest.skip("where.self does not route through FlagGems on this platform")
        result = _run_subprocess({"FLAGOS_LOG": "dispatch"})
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] where.self -> flagos_python" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_where__self": "cuda"}
        )
        if result.returncode != 0 and "backend not registered" in result.stderr:
            pytest.skip("cuda backend not available in this build")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] where.self -> cuda" in result.stderr

    @pytest.mark.metax
    def test_dispatch_log_metax(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_where__self": "metax"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] where.self -> metax" in result.stderr


class TestWhereAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify where.self on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_where__self": "ascend"})
        assert result.returncode == 0
