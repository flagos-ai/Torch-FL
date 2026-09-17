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
cat dispatch tests

Verifies that torch.cat:
  - produces correct results on flagos device
  - C++ wrapper routes to the backend the platform conf lists for cat (the
    vendor kernel, since FlagGems has no cat kernel) or cuda via env override
  - dispatch log confirms the actual backend used
  - KV cache / normal inference common patterns (append, mask extension, non-contiguous input, etc.)

Usage:
    pytest tests/integration/ops/test_cat_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend


DEVICE = "flagos:0"


def _cat_cpu_reference(tensors, dim: int) -> torch.Tensor:
    return torch.cat([t.detach().cpu() for t in tensors], dim=dim)


def _assert_exact_match(dev: torch.Tensor, ref: torch.Tensor, msg: str = "") -> None:
    torch.testing.assert_close(dev.cpu(), ref.cpu(), rtol=0, atol=0, msg=msg)


def _assert_kv_prefix_unchanged(
    out: torch.Tensor, prefix_src: torch.Tensor, seq_len: int
) -> None:
    """After KV cache cat(dim=-2), the first seq_len positions should match prefix_src exactly."""
    _assert_exact_match(
        out[:, :, :seq_len, :],
        prefix_src,
        msg=f"prefix len={seq_len} corrupted after cat(dim=-2)",
    )


@pytest.fixture(scope="session")
def cuda_ref():
    """Reference cat results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    a = torch.randn(32, 64, device="cuda:0", dtype=torch.float32)
    b = torch.randn(16, 64, device="cuda:0", dtype=torch.float32)
    return a, b, torch.cat([a, b], dim=0)


def _run_cat_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a minimal cat call in a subprocess and return the result."""
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.cat([a, b], dim=0)"
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


class TestCatDispatch:
    """torch.cat correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_cat_dim0(self):
        torch.manual_seed(0)
        a = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        b = torch.randn(16, 64, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=0)
        assert out.shape == (48, 64)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_cat_dim1(self):
        torch.manual_seed(1)
        a = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        b = torch.randn(32, 128, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=1)
        ref = torch.cat([a.cpu(), b.cpu()], dim=1)
        assert out.shape == (32, 192)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_cat_single_tensor(self):
        torch.manual_seed(2)
        a = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a], dim=0)
        assert out.shape == a.shape
        torch.testing.assert_close(out.cpu(), a.cpu())

    @pytest.mark.anyplatform
    def test_cat_multiple_tensors(self):
        torch.manual_seed(3)
        tensors = [
            torch.randn(4, 8, device=DEVICE, dtype=torch.float32) for _ in range(5)
        ]
        out = torch.cat(tensors, dim=0)
        assert out.shape == (20, 8)

    @pytest.mark.anyplatform
    def test_cat_negative_dim(self):
        torch.manual_seed(4)
        a = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        b = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=-1)
        ref = torch.cat([a.cpu(), b.cpu()], dim=-1)
        assert out.shape == (4, 24)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_cat_cache_init_empty_1d_with_dim_minus2(self):
        """Match transformers DynamicLayer.update first cat call."""
        k_cache = torch.tensor([], device=DEVICE, dtype=torch.float16)
        k_states = torch.randn(1, 2, 3, 4, device=DEVICE, dtype=torch.float16)
        out = torch.cat([k_cache, k_states], dim=-2)
        assert out.shape == k_states.shape
        torch.testing.assert_close(
            out.cpu().float(), k_states.cpu().float(), rtol=0, atol=0
        )

    @pytest.mark.cuda
    def test_cat_matches_cuda_ref(self, cuda_ref):
        """flagos cat result must match CUDA reference."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        a_cuda, b_cuda, ref = cuda_ref
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.cat([a, b], dim=0)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=1e-5,
            atol=1e-5,
            msg="cat result on flagos differs from CUDA reference",
        )

    @pytest.mark.anyplatform
    def test_cat_half(self):
        torch.manual_seed(5)
        a = torch.randn(16, 32, device=DEVICE, dtype=torch.float16)
        b = torch.randn(8, 32, device=DEVICE, dtype=torch.float16)
        out = torch.cat([a, b], dim=0)
        assert out.dtype == torch.float16
        assert out.shape == (24, 32)

    @pytest.mark.anyplatform
    def test_cat_3d(self):
        torch.manual_seed(6)
        a = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 4, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=2)
        ref = torch.cat([a.cpu(), b.cpu()], dim=2)
        assert out.shape == (2, 4, 24)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_cat_rotate_half_pattern(self):
        """RoPE rotate_half uses cat((-x2, x1), dim=-1) on 4D tensors."""
        torch.manual_seed(7)
        x = torch.randn(1, 16, 22, 128, device=DEVICE, dtype=torch.float16)
        x1 = x[..., :64]
        x2 = x[..., 64:]
        out = torch.cat((-x2, x1), dim=-1)
        ref = torch.cat((-x2.float().cpu(), x1.float().cpu()), dim=-1)
        torch.testing.assert_close(out.float().cpu(), ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.anyplatform
    def test_cat_4d_dim0(self):
        """4D concatenation along dim=0 (outer/inner dimensions differ from dim=-2, covering different stride patterns)."""
        torch.manual_seed(8)
        a = torch.randn(2, 4, 8, 16, device=DEVICE, dtype=torch.float32)
        b = torch.randn(3, 4, 8, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=0)
        _assert_exact_match(out, _cat_cpu_reference([a, b], dim=0))

    @pytest.mark.anyplatform
    def test_cat_4d_dim1(self):
        """4D concatenation along dim=1."""
        torch.manual_seed(9)
        a = torch.randn(2, 3, 8, 16, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 5, 8, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=1)
        _assert_exact_match(out, _cat_cpu_reference([a, b], dim=1))

    @pytest.mark.anyplatform
    def test_cat_noncontiguous_inputs(self):
        """metax cat will first make inputs contiguous; non-contiguous inputs should also match CPU behavior."""
        torch.manual_seed(10)
        a = torch.randn(1, 8, 16, 64, device=DEVICE, dtype=torch.float16).transpose(
            2, 3
        )
        b = torch.randn(1, 8, 16, 64, device=DEVICE, dtype=torch.float16).transpose(
            2, 3
        )
        assert not a.is_contiguous()
        assert not b.is_contiguous()
        out = torch.cat([a, b], dim=-2)
        _assert_exact_match(out, _cat_cpu_reference([a, b], dim=-2))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cat_kv_shape_dtypes(self, dtype):
        """Common inference dtype (fp16/bf16) in KV shape should match CPU behavior."""
        torch.manual_seed(11)
        k_old = torch.randn(1, 8, 10, 128, device=DEVICE, dtype=dtype)
        k_new = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=dtype)
        out = torch.cat([k_old, k_new], dim=-2)
        _assert_exact_match(out, _cat_cpu_reference([k_old, k_new], dim=-2))


class TestCatInferencePatterns:
    """Common cat usage in inference loop (excluding KV operations)."""

    @pytest.mark.anyplatform
    def test_cat_attention_mask_extend_decode(self):
        """Each decode step extends attention_mask: cat([mask, ones], dim=-1)."""
        torch.manual_seed(20)
        mask = torch.ones(1, 22, device=DEVICE, dtype=torch.long)
        extra = torch.ones(1, 1, device=DEVICE, dtype=torch.long)
        out = torch.cat([mask, extra], dim=-1)
        _assert_exact_match(out, _cat_cpu_reference([mask, extra], dim=-1))

    @pytest.mark.anyplatform
    def test_cat_attention_mask_repeated_extend(self):
        """Simulate multiple decode steps extending mask continuously, maintaining the same prefix for each step."""
        torch.manual_seed(21)
        mask = torch.ones(1, 8, device=DEVICE, dtype=torch.long)
        for _ in range(6):
            prev = mask.clone()
            extra = torch.ones(mask.shape[0], 1, device=DEVICE, dtype=mask.dtype)
            mask = torch.cat([mask, extra], dim=-1)
            _assert_exact_match(mask[:, :-1], prev, msg="mask prefix corrupted")

    @pytest.mark.anyplatform
    def test_cat_stack_position_ids(self):
        """Some paths concatenate position_ids along dim."""
        torch.manual_seed(22)
        pos_old = torch.arange(22, device=DEVICE, dtype=torch.long).unsqueeze(0)
        pos_new = torch.tensor([[22]], device=DEVICE, dtype=torch.long)
        out = torch.cat([pos_old, pos_new], dim=-1)
        _assert_exact_match(out, _cat_cpu_reference([pos_old, pos_new], dim=-1))


class TestCatKvCacheAppend:
    """
    KV cache update scenario: DynamicLayer.update uses
    torch.cat([keys_old, key_new], dim=-2)。

    Regression: The prefill segment (first seq_len positions) after concatenation must match keys_old exactly.
    See tests/integration/test_first_layer_debug.py::test_kv_cache_update_isolated
    """

    @pytest.mark.parametrize("seq_len", [4, 22])
    def test_cat_kv_cache_append_preserves_prefill_prefix(self, seq_len):
        """After cat(dim=-2), the first half should match prefill cache element-wise."""
        torch.manual_seed(42)
        k_prefill = torch.randn(1, 8, seq_len, 128, device=DEVICE, dtype=torch.float16)
        k_new = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)

        out = torch.cat([k_prefill, k_new], dim=-2)
        assert out.shape == (1, 8, seq_len + 1, 128)

        torch.testing.assert_close(
            out[:, :, :seq_len, :].cpu(),
            k_prefill.cpu(),
            rtol=0,
            atol=0,
            msg="prefill K prefix corrupted after cat(dim=-2)",
        )
        torch.testing.assert_close(
            out[:, :, seq_len:, :].cpu(),
            k_new.cpu(),
            rtol=0,
            atol=0,
            msg="appended K suffix wrong after cat(dim=-2)",
        )

    @pytest.mark.anyplatform
    def test_cat_kv_cache_append_matches_cpu_reference(self):
        """flagos cat result should match CPU reference (typical Qwen3-0.6B decode shape)."""
        torch.manual_seed(0)
        seq_len = 22
        k_prefill = torch.randn(1, 8, seq_len, 128, device=DEVICE, dtype=torch.float16)
        k_new = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)

        out_dev = torch.cat([k_prefill, k_new], dim=-2)
        ref = torch.cat([k_prefill.cpu(), k_new.cpu()], dim=-2)

        torch.testing.assert_close(
            out_dev.cpu(),
            ref,
            rtol=0,
            atol=0,
            msg="flagos cat(dim=-2) differs from CPU reference",
        )

    @pytest.mark.anyplatform
    def test_cat_kv_cache_append_large_magnitude_fp16(self):
        """
        Use large dynamic range fp16 values (|x| can reach hundreds) similar to inference.
        """
        torch.manual_seed(123)
        seq_len = 22
        k_prefill = (
            torch.randn(1, 8, seq_len, 128, device=DEVICE, dtype=torch.float16) * 80.0
        )
        k_new = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16) * 80.0

        out = torch.cat([k_prefill, k_new], dim=-2)
        prefix = out[:, :, :seq_len, :]

        diff = (prefix.cpu().float() - k_prefill.cpu().float()).abs()
        max_diff = diff.max().item()
        assert max_diff == 0.0, (
            f"prefill prefix max abs diff {max_diff} after cat, expected 0"
        )

    @pytest.mark.anyplatform
    def test_cat_kv_cache_append_v_cache(self):
        """V cache append with same shape, ensuring both K/V paths are covered."""
        torch.manual_seed(7)
        seq_len = 22
        v_prefill = torch.randn(1, 8, seq_len, 128, device=DEVICE, dtype=torch.float16)
        v_new = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)

        out = torch.cat([v_prefill, v_new], dim=-2)
        torch.testing.assert_close(
            out[:, :, :seq_len, :].cpu(), v_prefill.cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            out[:, :, seq_len:, :].cpu(), v_new.cpu(), rtol=0, atol=0
        )

    @pytest.mark.anyplatform
    def test_cat_kv_cache_repeated_single_token_appends(self):
        """Simulate generate loop: each step appends 1 token, history prefix should always remain unchanged."""
        torch.manual_seed(30)
        cache = torch.randn(1, 8, 12, 128, device=DEVICE, dtype=torch.float16)
        for step in range(8):
            prev_len = cache.size(2)
            snapshot = cache.clone()
            new_tok = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)
            cache = torch.cat([cache, new_tok], dim=-2)
            assert cache.size(2) == prev_len + 1
            _assert_kv_prefix_unchanged(cache, snapshot, prev_len)
            _assert_exact_match(
                cache[:, :, prev_len:, :], new_tok, msg=f"step {step} new token slice"
            )

    @pytest.mark.anyplatform
    def test_cat_kv_cache_multi_token_chunk(self):
        """Append multiple tokens at once (chunked prefill / mini-batch decode)."""
        torch.manual_seed(31)
        cache = torch.randn(1, 8, 20, 128, device=DEVICE, dtype=torch.float16)
        chunk = torch.randn(1, 8, 4, 128, device=DEVICE, dtype=torch.float16)
        out = torch.cat([cache, chunk], dim=-2)
        _assert_kv_prefix_unchanged(out, cache, 20)
        _assert_exact_match(out, _cat_cpu_reference([cache, chunk], dim=-2))

    @pytest.mark.anyplatform
    def test_cat_kv_cache_three_tensors(self):
        """Concatenate three tensors at once (segmented cache / manual concatenation)."""
        torch.manual_seed(32)
        parts = [
            torch.randn(1, 8, n, 128, device=DEVICE, dtype=torch.float16)
            for n in (10, 3, 2)
        ]
        out = torch.cat(parts, dim=-2)
        _assert_exact_match(out, _cat_cpu_reference(parts, dim=-2))
        offset = 0
        for part in parts:
            n = part.size(2)
            _assert_exact_match(
                out[:, :, offset : offset + n, :],
                part,
                msg=f"segment at offset={offset}",
            )
            offset += n

    @pytest.mark.anyplatform
    def test_cat_kv_cache_batch_dim_gt_one(self):
        """When batch_size > 1, each batch row is concatenated independently."""
        torch.manual_seed(33)
        cache = torch.randn(2, 8, 15, 128, device=DEVICE, dtype=torch.float16)
        new_tok = torch.randn(2, 8, 1, 128, device=DEVICE, dtype=torch.float16)
        out = torch.cat([cache, new_tok], dim=-2)
        _assert_kv_prefix_unchanged(out, cache, 15)
        _assert_exact_match(out, _cat_cpu_reference([cache, new_tok], dim=-2))

    @pytest.mark.anyplatform
    def test_cat_kv_cache_gqa_query_head_shape(self):
        """When Q is projected to num_attention_heads=16 (not kv_heads=8), same concatenation along seq dimension."""
        torch.manual_seed(34)
        cache = torch.randn(1, 16, 18, 128, device=DEVICE, dtype=torch.float16)
        new_tok = torch.randn(1, 16, 1, 128, device=DEVICE, dtype=torch.float16)
        out = torch.cat([cache, new_tok], dim=-2)
        _assert_kv_prefix_unchanged(out, cache, 18)
        _assert_exact_match(out, _cat_cpu_reference([cache, new_tok], dim=-2))

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_cat_kv_cache_various_head_dims(self, head_dim):
        """Different head_dim results in different inner_size products, regression stride calculation."""
        torch.manual_seed(35)
        cache = torch.randn(1, 8, 11, head_dim, device=DEVICE, dtype=torch.float16)
        new_tok = torch.randn(1, 8, 2, head_dim, device=DEVICE, dtype=torch.float16)
        out = torch.cat([cache, new_tok], dim=-2)
        _assert_kv_prefix_unchanged(out, cache, 11)
        _assert_exact_match(out, _cat_cpu_reference([cache, new_tok], dim=-2))

    @pytest.mark.anyplatform
    def test_cat_kv_cache_noncontiguous_inputs(self):
        """Non-contiguous KV cache: cat views with head_dim step=2."""
        torch.manual_seed(36)
        base = torch.randn(1, 8, 14, 128, device=DEVICE, dtype=torch.float16)
        cache_nc = base[..., ::2]
        assert not cache_nc.is_contiguous()
        new_tok = torch.randn(1, 8, 1, 64, device=DEVICE, dtype=torch.float16)
        out = torch.cat([cache_nc, new_tok], dim=-2)
        _assert_exact_match(out, _cat_cpu_reference([cache_nc, new_tok], dim=-2))

    @pytest.mark.anyplatform
    def test_cat_kv_cache_empty_1d_then_states(self):
        """Same as DynamicLayer first update: empty 1D cache + first segment states."""
        empty = torch.tensor([], device=DEVICE, dtype=torch.float16)
        states = torch.randn(1, 8, 6, 128, device=DEVICE, dtype=torch.float16)
        out = torch.cat([empty, states], dim=-2)
        _assert_exact_match(out, _cat_cpu_reference([empty, states], dim=-2))


class TestCatDispatchLog:
    """Verify C++ wrapper routes to the correct backend."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        """FLAGOS_OP_cat=flaggems_python routes cat to flagos_python backend."""
        result = _run_cat_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_cat": "flaggems_python"},
            check=False,
        )
        assert "[flagos dispatch] cat -> flagos_python" in result.stderr, (
            f"Expected flagos_python dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """cat dispatches to whatever backend this platform's conf lists.

        FlagGems has no Triton kernel for cat, so every generated conf keeps it
        on that platform's vendor kernel -- ``cuda`` on CUDA/DCU/PPU, ``musa`` on
        MUSA. Reading the conf keeps the assertion true on all of them and still
        checks the real property: the runtime honours the table it was given.
        """
        result = _run_cat_subprocess({"FLAGOS_LOG": "dispatch"})
        expected = routed_backend("cat")
        assert f"[flagos dispatch] cat -> {expected}" in result.stderr, (
            f"Expected {expected} dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        """FLAGOS_OP_cat=cuda overrides to cuda backend."""
        result = _run_cat_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_cat": "cuda"},
        )
        assert "[flagos dispatch] cat -> cuda" in result.stderr, (
            f"Expected cuda dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.ascend
    def test_dispatch_log_ascend_override(self):
        """FLAGOS_OP_cat=ascend overrides to ascend backend."""
        result = _run_cat_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_cat": "ascend"}
        )
        assert "[flagos dispatch] cat -> ascend" in result.stderr, (
            f"Expected ascend dispatch log, got:\n{result.stderr}"
        )


class TestCatAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify cat on ascend backend matches CPU reference."""
        result = _run_cat_subprocess({"FLAGOS_OP_cat": "ascend"})
        assert result.returncode == 0
