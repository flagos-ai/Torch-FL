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
embedding dispatch tests

Verifies that torch.nn.functional.embedding (aten.embedding):
  - produces correct results on flagos device
  - C++ wrapper routes per the platform conf (or cuda via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_embedding_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401

from backend_conf import routed_backend, routed_backend_or_none


DEVICE = "flagos:0"


@pytest.fixture(scope="session")
def cuda_ref():
    """Reference embedding results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    weight = torch.randn(1000, 128, device="cuda:0", dtype=torch.float32)
    indices = torch.randint(0, 1000, (8, 16), device="cuda:0")
    return weight, indices, F.embedding(indices, weight)


def _run_embedding_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a minimal embedding call in a subprocess and return the result."""
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch, torch.nn.functional as F; "
        "w = torch.randn(100,32,device='flagos:0'); "
        "idx = torch.randint(0,100,(4,),device='flagos:0'); "
        "F.embedding(idx, w)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if check:
        assert result.returncode == 0, (
            f"Subprocess failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result


class TestEmbeddingDispatch:
    """torch.nn.functional.embedding correctness on flagos."""

    @pytest.mark.anyplatform
    def test_embedding_basic(self):
        torch.manual_seed(0)
        weight = torch.randn(100, 64, device=DEVICE, dtype=torch.float32)
        indices = torch.tensor([0, 5, 10, 99], device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.shape == (4, 64)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_embedding_2d_indices(self):
        torch.manual_seed(1)
        weight = torch.randn(500, 128, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 500, (8, 16), device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.shape == (8, 16, 128)

    @pytest.mark.anyplatform
    def test_embedding_correctness(self):
        """Verify embedding == index-select on weight."""
        torch.manual_seed(2)
        weight = torch.randn(50, 32, device=DEVICE, dtype=torch.float32)
        indices = torch.tensor([0, 3, 7, 49], device=DEVICE)
        out = F.embedding(indices, weight)
        expected = weight.index_select(0, indices)
        torch.testing.assert_close(
            out.cpu(),
            expected.cpu(),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.cuda
    def test_embedding_matches_cuda_ref(self, cuda_ref):
        """flagos embedding must match CUDA reference."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        w_cuda, idx_cuda, ref = cuda_ref
        w = w_cuda.to(DEVICE)
        idx = idx_cuda.to(DEVICE)
        out = F.embedding(idx, w)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=1e-5,
            atol=1e-5,
            msg="embedding on flagos differs from CUDA",
        )

    @pytest.mark.anyplatform
    def test_embedding_half(self):
        torch.manual_seed(3)
        weight = torch.randn(100, 64, device=DEVICE, dtype=torch.float16)
        indices = torch.tensor([0, 10, 50], device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.dtype == torch.float16
        assert out.shape == (3, 64)

    @pytest.mark.anyplatform
    def test_embedding_large_vocab(self):
        torch.manual_seed(4)
        weight = torch.randn(10000, 256, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 10000, (64,), device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.shape == (64, 256)

    @pytest.mark.anyplatform
    def test_embedding_via_module(self):
        """Test via nn.Embedding module."""
        torch.manual_seed(5)
        emb = torch.nn.Embedding(100, 64).to(DEVICE)
        indices = torch.tensor([0, 1, 2], device=DEVICE)
        out = emb(indices)
        assert out.shape == (3, 64)
        assert out.device.type == "flagos"


class TestEmbeddingDispatchLog:
    """Verify C++ wrapper routes to correct backend."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        """The per-op override selects the FlagGems Python path for embedding.

        Skipped where the conf routes embedding to ``none``: the op is then not
        claimed on PrivateUse1 at all, so the call reaches cpu_fallback before
        the dispatcher and no override can show up in the log.
        """
        if routed_backend_or_none("embedding") is None:
            pytest.skip("embedding is routed to 'none' on this platform")
        result = _run_embedding_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_embedding": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] embedding -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """Embedding dispatches to the backend this platform's conf routes it to.

        ``FLAGOS_USE_FLAGGEMS`` only asks for the runtime path; the conf decides
        the route, and FlagGems-first is not unconditional. The CUDA conf
        returns ``embedding`` to CUDA boxing because the FlagGems route raises
        ``RuntimeError: Triton Error [CUDA]: context is destroyed`` (see
        measured_flaggems_rollback in scripts/codegen/codegen_ops.py and
        docs/reference/operator-support.md), and GCU leaves it to cpu_fallback
        because the FlagGems embedding kernel cannot compile the int64 index
        operand GCU300 rejects. Platforms whose conf keeps the FlagGems route
        still log flagos_python, so asserting the conf's own value keeps this
        test meaningful on every platform rather than pinning it to the one it
        was written on.
        """
        if routed_backend_or_none("embedding") != "flagos_python":
            pytest.skip("embedding does not route through FlagGems on this platform")
        result = _run_embedding_subprocess({"FLAGOS_LOG": "dispatch"})
        expected = routed_backend("embedding")
        assert f"[flagos dispatch] embedding -> {expected}" in result.stderr, (
            f"Expected embedding -> {expected}, got:\n{result.stderr}"
        )

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_embedding_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_embedding": "cuda",
            }
        )
        assert "[flagos dispatch] embedding -> cuda" in result.stderr, (
            f"Expected cuda log, got:\n{result.stderr}"
        )

    @pytest.mark.ascend
    def test_dispatch_log_ascend_override(self):
        """FLAGOS_OP_embedding=ascend overrides to ascend backend."""
        result = _run_embedding_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_embedding": "ascend"}
        )
        assert "[flagos dispatch] embedding -> ascend" in result.stderr, (
            f"Expected ascend dispatch log, got:\n{result.stderr}"
        )


class TestEmbeddingAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify embedding on ascend backend matches CPU reference."""
        result = _run_embedding_subprocess({"FLAGOS_OP_embedding": "ascend"})
        assert result.returncode == 0
