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
bmm dispatch tests

Verifies that torch.bmm and torch.bmm.out:
  - produce correct results on flagos device
  - C++ wrapper routes to flaggems (default) or cuda (via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_bmm_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def bmm4d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    4D BMM wrapper:
      a: [B, H, M, K]
      b: [B, H, K, N]
      out: [B, H, M, N]
    Implemented via flatten -> torch.bmm -> reshape.
    """
    assert a.dim() == 4 and b.dim() == 4, "bmm4d expects 4D tensors"
    assert a.shape[:2] == b.shape[:2], "batch/head dims must match"
    assert a.shape[-1] == b.shape[-2], "K dimension mismatch"
    bsz, heads, m, k = a.shape
    n = b.shape[-1]
    a3 = a.reshape(bsz * heads, m, k).contiguous()
    b3 = b.reshape(bsz * heads, k, n).contiguous()
    out3 = torch.bmm(a3, b3)
    return out3.reshape(bsz, heads, m, n)


@pytest.fixture(scope="session")
def cuda_ref():
    """Reference bmm results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    a = torch.randn(8, 128, 256, device="cuda:0", dtype=torch.float32)
    b = torch.randn(8, 256, 64, device="cuda:0", dtype=torch.float32)
    return a, b, torch.bmm(a, b)


def _run_bmm_subprocess(
    extra_env: dict,
    use_out: bool = False,
    check: bool = True,
    stub_python_op: bool = False,
) -> subprocess.CompletedProcess:
    """Run a minimal bmm call in a subprocess and return the result."""
    env = os.environ.copy()
    env.update(extra_env)
    setup = "import torch_fl, torch; "
    if stub_python_op:
        # This test only verifies routing; avoid triggering FlagGems BMM autotuning.
        setup += (
            "import flag_gems.ops; "
            "flag_gems.ops.bmm = lambda *args: "
            "(_ for _ in ()).throw(RuntimeError('bmm dispatch sentinel')); "
        )
    if use_out:
        code = setup + (
            "a = torch.randn(2,4,4,device='flagos:0'); "
            "b = torch.randn(2,4,4,device='flagos:0'); "
            "out = torch.empty(2,4,4,device='flagos:0'); "
            "torch.bmm(a, b, out=out)"
        )
    else:
        code = setup + (
            "a = torch.randn(2,4,4,device='flagos:0'); "
            "b = torch.randn(2,4,4,device='flagos:0'); "
            "torch.bmm(a, b)"
        )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if check:
        assert result.returncode == 0, (
            f"Subprocess failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result


class TestBmmDispatch:
    """torch.bmm correctness and cross-device consistency."""

    @pytest.mark.parametrize(
        "B,M,K,N",
        [(4, 128, 256, 64), (1, 1, 1, 1), (8, 512, 512, 512)],
    )
    @pytest.mark.anyplatform
    def test_bmm_shape(self, B, M, K, N):
        torch.manual_seed(0)
        a = torch.randn(B, M, K, device=DEVICE, dtype=torch.float32)
        b = torch.randn(B, K, N, device=DEVICE, dtype=torch.float32)
        out = torch.bmm(a, b)
        assert out.shape == (B, M, N)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_bmm_out(self):
        torch.manual_seed(1)
        a = torch.randn(4, 64, 128, device=DEVICE, dtype=torch.float32)
        b = torch.randn(4, 128, 32, device=DEVICE, dtype=torch.float32)
        out = torch.empty(4, 64, 32, device=DEVICE, dtype=torch.float32)
        ret = torch.bmm(a, b, out=out)
        assert ret.data_ptr() == out.data_ptr()
        assert out.shape == (4, 64, 32)

    @pytest.mark.cuda
    def test_bmm_matches_cuda_ref(self, cuda_ref):
        """flagos bmm result must match CUDA reference."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        a_cuda, b_cuda, ref = cuda_ref
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.bmm(a, b)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=1e-3,
            atol=1e-3,
            msg="bmm result on flagos differs from CUDA reference",
        )

    @pytest.mark.anyplatform
    def test_bmm_half(self):
        torch.manual_seed(3)
        a = torch.randn(4, 64, 128, device=DEVICE, dtype=torch.float16)
        b = torch.randn(4, 128, 32, device=DEVICE, dtype=torch.float16)
        out = torch.bmm(a, b)
        assert out.dtype == torch.float16
        assert out.shape == (4, 64, 32)

    @pytest.mark.anyplatform
    def test_bmm_half_attention_like_matches_cpu(self):
        """
        Simulate attention q@k^T BMM shape and compare flagos vs CPU.
        Shape mapping:
          [B, H, Q, D] x [B, H, D, K] -> flatten to [B*H, Q, D] x [B*H, D, K]
        """
        torch.manual_seed(7)
        batch = 1
        heads = 16
        q_len = 64
        k_len = 64
        head_dim = 64

        q = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float16)
        k = torch.randn(batch, heads, k_len, head_dim, dtype=torch.float16)
        k_t = k.transpose(-1, -2).contiguous()

        a_cpu = q.reshape(batch * heads, q_len, head_dim).contiguous()
        b_cpu = k_t.reshape(batch * heads, head_dim, k_len).contiguous()
        ref = torch.bmm(a_cpu, b_cpu)

        a = a_cpu.to(DEVICE)
        b = b_cpu.to(DEVICE)
        out = torch.bmm(a, b).float().cpu()
        ref = ref.float().cpu()

        torch.testing.assert_close(
            out,
            ref,
            rtol=8e-2,
            atol=8e-2,
            msg="attention-like fp16 bmm on flagos differs from CPU reference",
        )

    @pytest.mark.anyplatform
    def test_bmm4d_fp16_matches_cpu_reference(self):
        """
        Validate 4D bmm wrapper (flatten+bmm+reshape) with attention-like shapes.
        """
        torch.manual_seed(11)
        bsz, heads, q_len, k_len, head_dim = 1, 16, 22, 22, 128
        q_cpu = torch.randn(bsz, heads, q_len, head_dim, dtype=torch.float16)
        k_cpu = torch.randn(bsz, heads, k_len, head_dim, dtype=torch.float16)
        k_t_cpu = k_cpu.transpose(-1, -2).contiguous()

        ref = torch.matmul(q_cpu.float(), k_t_cpu.float())
        out = (
            bmm4d(
                q_cpu.to(DEVICE),
                k_t_cpu.to(DEVICE),
            )
            .float()
            .cpu()
        )
        torch.testing.assert_close(
            out,
            ref,
            rtol=8e-2,
            atol=8e-2,
            msg="bmm4d fp16 on flagos differs from CPU reference",
        )

    @pytest.mark.anyplatform
    def test_flagos_4d_matmul_vs_bmm4d(self):
        """
        Compare current flagos 4D matmul path against bmm4d path.
        If this fails while bmm4d matches CPU, current 4D matmul path is incorrect.
        """
        torch.manual_seed(12)
        bsz, heads, q_len, k_len, head_dim = 1, 16, 22, 22, 128
        q = torch.randn(bsz, heads, q_len, head_dim, dtype=torch.float16, device=DEVICE)
        k = torch.randn(bsz, heads, k_len, head_dim, dtype=torch.float16, device=DEVICE)
        k_t = k.transpose(-1, -2).contiguous()

        out_matmul = torch.matmul(q, k_t).float().cpu()
        out_bmm4d = bmm4d(q, k_t).float().cpu()
        torch.testing.assert_close(
            out_matmul,
            out_bmm4d,
            rtol=8e-2,
            atol=8e-2,
            msg="flagos 4D matmul and bmm4d are inconsistent",
        )


class TestBmmDispatchLog:
    """Verify C++ wrapper routes to the correct backend."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        """FLAGOS_OP_bmm=flaggems_python routes bmm to flagos_python backend."""
        result = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bmm": "flaggems_python"},
            check=False,
            stub_python_op=True,
        )
        assert "[flagos dispatch] bmm -> flagos_python" in result.stderr, (
            f"Expected flagos_python dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """With the FlagGems runtime path on, bmm routes to flagos_python."""
        result = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_USE_FLAGGEMS": "1"}
        )
        assert "[flagos dispatch] bmm -> flagos_python" in result.stderr, (
            f"Expected flagos_python dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        """FLAGOS_OP_bmm=cuda overrides to cuda backend."""
        result = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bmm": "cuda"}
        )
        assert "[flagos dispatch] bmm -> cuda" in result.stderr, (
            f"Expected cuda dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.flaggems
    def test_dispatch_log_bmm_out_flaggems_runtime(self):
        """With the FlagGems runtime path on, bmm.out routes to flagos_python."""
        result = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_USE_FLAGGEMS": "1"},
            use_out=True,
        )
        assert "[flagos dispatch] bmm.out -> flagos_python" in result.stderr, (
            f"Expected flagos_python dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.cuda
    def test_dispatch_log_bmm_out_cuda_override(self):
        """FLAGOS_OP_bmm__out=cuda overrides bmm.out to cuda."""
        result = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bmm__out": "cuda"},
            use_out=True,
        )
        assert "[flagos dispatch] bmm.out -> cuda" in result.stderr, (
            f"Expected cuda dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.ascend
    def test_dispatch_log_ascend_override(self):
        """FLAGOS_OP_bmm=ascend overrides to ascend backend."""
        result = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bmm": "ascend"}
        )
        assert "[flagos dispatch] bmm -> ascend" in result.stderr, (
            f"Expected ascend dispatch log, got:\n{result.stderr}"
        )


class TestBmmAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify bmm on ascend backend matches CPU reference."""
        result = _run_bmm_subprocess({"FLAGOS_OP_bmm": "ascend"})
        assert result.returncode == 0

    @pytest.mark.ascend
    def test_ascend_out_correctness(self):
        """Verify bmm.out on ascend backend matches CPU reference."""
        result = _run_bmm_subprocess(
            {"FLAGOS_OP_bmm__out": "ascend"},
            use_out=True,
        )
        assert result.returncode == 0
