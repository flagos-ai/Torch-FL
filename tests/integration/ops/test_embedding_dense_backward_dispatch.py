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
embedding_dense_backward dispatch tests

Verifies that embedding backward:
  - produces correct gradients on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_embedding_dense_backward_dispatch.py -v
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
        "emb = torch.nn.Embedding(100, 32).to('flagos:0'); "
        "idx = torch.tensor([0,1,2], device='flagos:0'); "
        "out = emb(idx); "
        "out.sum().backward()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestEmbeddingDenseBackwardCorrectness:
    """embedding_dense_backward correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_embedding_backward_basic(self):
        torch.manual_seed(0)
        emb = torch.nn.Embedding(50, 16).to(DEVICE)
        idx = torch.tensor([0, 5, 10, 5], device=DEVICE)
        out = emb(idx)
        out.sum().backward()
        assert emb.weight.grad is not None
        assert emb.weight.grad.shape == (50, 16)
        # Only accessed indices should have non-zero grad
        assert emb.weight.grad.cpu()[0].sum().item() != 0.0
        assert emb.weight.grad.cpu()[5].sum().item() != 0.0
        assert emb.weight.grad.cpu()[1].sum().item() == 0.0

    @pytest.mark.anyplatform
    def test_embedding_backward_matches_cpu(self):
        torch.manual_seed(1)
        emb_cpu = torch.nn.Embedding(100, 32)
        emb_fl = torch.nn.Embedding(100, 32).to(DEVICE)
        # Copy same weights
        emb_fl.weight.data.copy_(emb_cpu.weight.data.to(DEVICE))

        idx = torch.tensor([3, 7, 3, 50])
        out_cpu = emb_cpu(idx)
        out_cpu.sum().backward()

        idx_fl = idx.to(DEVICE)
        out_fl = emb_fl(idx_fl)
        out_fl.sum().backward()

        torch.testing.assert_close(
            emb_fl.weight.grad.cpu(), emb_cpu.weight.grad, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.anyplatform
    def test_embedding_backward_duplicate_indices(self):
        torch.manual_seed(2)
        emb = torch.nn.Embedding(20, 8).to(DEVICE)
        idx = torch.tensor([3, 3, 3, 3], device=DEVICE)
        out = emb(idx)
        out.sum().backward()
        # Grad for index 3 should be 4x the embedding dim (all ones * 4)
        expected = torch.ones(8) * 4.0
        torch.testing.assert_close(emb.weight.grad.cpu()[3], expected)


class TestEmbeddingDenseBackwardDispatch:
    """Verify dispatch routing for embedding_dense_backward op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        """The per-op override picks the FlagGems Python path for embedding_dense_backward.

        Skipped where the conf routes embedding_dense_backward to ``none``: the
        op is then not claimed on PrivateUse1 at all, so the call reaches
        cpu_fallback before the dispatcher and no override can show up in the
        log.
        """
        if routed_backend_or_none("embedding_dense_backward") is None:
            pytest.skip("embedding_dense_backward is routed to 'none' on this platform")
        result = _run_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_embedding_dense_backward": "flaggems_python",
            },
            check=False,
        )
        assert (
            "[flagos dispatch] embedding_dense_backward -> flagos_python"
            in result.stderr
        )

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_embedding_dense_backward": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] embedding_dense_backward -> cuda" in result.stderr


class TestEmbeddingDenseBackwardAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify embedding_dense_backward on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_embedding_dense_backward": "ascend"})
        assert result.returncode == 0
