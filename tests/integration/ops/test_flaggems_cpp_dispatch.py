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
FlagGems C++ dispatch (kFlagGemsCpp) integration tests.

Verifies that torch_fl built with FLAGGEMS_CPP=ON routes ops to the C++
FlagGems path (backend label "flagos") when FLAGOS_USE_FLAGGEMS_CPP=1, and
that results are numerically correct.

These tests require a torch_fl wheel built with FLAGGEMS_CPP=ON (i.e.
liboperators.so linked in).  They are gated by @pytest.mark.flaggems_cpp and
are not included in the default CI matrix (which uses FLAGGEMS_CPP=OFF
wheels); add them once FlagGems C++ runtime is in the CI build image.

Usage (on a C++ wheel):
    FLAGOS_USE_FLAGGEMS_CPP=1 pytest tests/integration/ops/test_flaggems_cpp_dispatch.py -v -s
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401

DEVICE = "flagos:0"
_CPP_ENV = {"FLAGOS_USE_FLAGGEMS_CPP": "1", "FLAGOS_LOG_DISPATCH": "1"}


def _run_subprocess(
    code: str, extra_env: dict | None = None
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(_CPP_ENV)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


def _assert_routes_cpp(result: subprocess.CompletedProcess, op_name: str) -> None:
    assert result.returncode == 0, (
        f"Subprocess for {op_name} failed (exit {result.returncode}):\n{result.stderr}"
    )
    assert f"[flagos dispatch] {op_name} -> flagos" in result.stderr, (
        f"Expected '-> flagos' for {op_name}, got:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Dispatch-log tests: verify each op logs "-> flagos" under CPP path
# ---------------------------------------------------------------------------


class TestFlaggemsCppDispatchLog:
    """Each test verifies that FLAGOS_USE_FLAGGEMS_CPP=1 routes the op to
    the C++ FlagGems backend (kFlagGemsCpp, logged as '-> flagos')."""

    @pytest.mark.flaggems_cpp
    def test_mm_routes_cpp(self):
        code = (
            "import torch_fl, torch; "
            "a=torch.randn(64,64,device='flagos:0'); "
            "b=torch.randn(64,64,device='flagos:0'); "
            "torch.mm(a,b)"
        )
        _assert_routes_cpp(_run_subprocess(code), "mm")

    @pytest.mark.flaggems_cpp
    def test_bmm_routes_cpp(self):
        code = (
            "import torch_fl, torch; "
            "a=torch.randn(4,64,64,device='flagos:0'); "
            "b=torch.randn(4,64,64,device='flagos:0'); "
            "torch.bmm(a,b)"
        )
        _assert_routes_cpp(_run_subprocess(code), "bmm")

    @pytest.mark.flaggems_cpp
    def test_embedding_routes_cpp(self):
        code = (
            "import torch_fl, torch, torch.nn.functional as F; "
            "w=torch.randn(100,32,device='flagos:0'); "
            "idx=torch.randint(0,100,(8,),device='flagos:0'); "
            "F.embedding(idx,w)"
        )
        _assert_routes_cpp(_run_subprocess(code), "embedding")

    @pytest.mark.flaggems_cpp
    def test_softmax_routes_cpp(self):
        code = (
            "import torch_fl, torch, torch.nn.functional as F; "
            "x=torch.randn(32,128,device='flagos:0'); "
            "F.softmax(x,dim=-1)"
        )
        _assert_routes_cpp(_run_subprocess(code), "_softmax")

    @pytest.mark.flaggems_cpp
    def test_sum_routes_cpp(self):
        code = (
            "import torch_fl, torch; "
            "x=torch.randn(64,128,device='flagos:0'); "
            "torch.sum(x)"
        )
        _assert_routes_cpp(_run_subprocess(code), "sum")

    @pytest.mark.flaggems_cpp
    def test_sum_dim_routes_cpp(self):
        code = (
            "import torch_fl, torch; "
            "x=torch.randn(64,128,device='flagos:0'); "
            "torch.sum(x,dim=1)"
        )
        _assert_routes_cpp(_run_subprocess(code), "sum.dim_IntList")

    @pytest.mark.flaggems_cpp
    def test_zeros_routes_cpp(self):
        code = "import torch_fl, torch; torch.zeros(64,64,device='flagos:0')"
        _assert_routes_cpp(_run_subprocess(code), "zeros")


# ---------------------------------------------------------------------------
# Numerical correctness: C++ result matches Python / CPU reference
# ---------------------------------------------------------------------------


class TestFlaggemsCppCorrectness:
    """Verify that the C++ FlagGems path produces numerically correct results.

    Correctness is verified relative to CPU reference tensors.
    All tests are marked flaggems_cpp and require a C++ FlagGems wheel.
    """

    @pytest.mark.flaggems_cpp
    def test_mm_correctness(self):
        torch.manual_seed(42)
        a_cpu = torch.randn(64, 64)
        b_cpu = torch.randn(64, 64)
        ref = torch.mm(a_cpu, b_cpu)

        a = a_cpu.to(DEVICE)
        b = b_cpu.to(DEVICE)
        out = torch.mm(a, b)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.flaggems_cpp
    def test_bmm_correctness(self):
        torch.manual_seed(0)
        a_cpu = torch.randn(4, 32, 32)
        b_cpu = torch.randn(4, 32, 32)
        ref = torch.bmm(a_cpu, b_cpu)

        a = a_cpu.to(DEVICE)
        b = b_cpu.to(DEVICE)
        out = torch.bmm(a, b)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.flaggems_cpp
    def test_embedding_correctness(self):
        torch.manual_seed(1)
        weight_cpu = torch.randn(100, 64)
        indices_cpu = torch.randint(0, 100, (16,))
        ref = F.embedding(indices_cpu, weight_cpu)

        weight = weight_cpu.to(DEVICE)
        indices = indices_cpu.to(DEVICE)
        out = F.embedding(indices, weight)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.flaggems_cpp
    def test_softmax_correctness(self):
        torch.manual_seed(2)
        x_cpu = torch.randn(32, 128)
        ref = F.softmax(x_cpu, dim=-1)

        x = x_cpu.to(DEVICE)
        out = F.softmax(x, dim=-1)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.flaggems_cpp
    def test_sum_correctness(self):
        torch.manual_seed(3)
        x_cpu = torch.randn(128, 256)
        ref = torch.sum(x_cpu)

        x = x_cpu.to(DEVICE)
        out = torch.sum(x)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.flaggems_cpp
    def test_zeros_shape(self):
        out = torch.zeros(32, 64, device=DEVICE)
        assert out.shape == (32, 64)
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), torch.zeros(32, 64))
