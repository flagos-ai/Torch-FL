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
Basic Operations and Tensor Tests

All tests run on the flagos device.

Usage:
    pytest tests/integration/test_ops.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import torch_fl


@pytest.fixture(scope="session")
def device():
    print(
        f"\nflagos device count={torch_fl.flagos.device_count()}  "
        f"FlagGems enabled={torch_fl.is_flaggems_enabled()}  "
        f"registered ops={len(torch_fl.get_registered_ops())}"
    )
    return "flagos:0"


# ---------------------------------------------------------------------------
# 1. Tensor creation
# ---------------------------------------------------------------------------


class TestTensorCreation:
    def test_randn(self, device):
        x = torch.randn(1024, 1024, device=device)
        assert x.device.type == "flagos"

    def test_zeros(self, device):
        assert torch.zeros(64, 64, device=device).sum().item() == 0.0

    def test_ones(self, device):
        assert torch.ones(64, 64, device=device).sum().item() == 64 * 64

    def test_arange(self, device):
        assert torch.arange(10, device=device)[-1].item() == 9

    def test_full(self, device):
        assert torch.full((4, 4), 3.14, device=device)[0, 0].item() == pytest.approx(
            3.14
        )


# ---------------------------------------------------------------------------
# 2. Arithmetic operations
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_add(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        assert torch.allclose(a + b, torch.tensor([5.0, 7.0, 9.0], device=device))

    def test_sub(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        assert torch.allclose(a - b, torch.tensor([-3.0, -3.0, -3.0], device=device))

    def test_mul(self, device):
        a = torch.tensor([2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0], device=device)
        assert torch.allclose(a * b, torch.tensor([8.0, 15.0], device=device))

    def test_div(self, device):
        a = torch.tensor([10.0, 20.0], device=device)
        b = torch.tensor([2.0, 5.0], device=device)
        assert torch.allclose(a / b, torch.tensor([5.0, 4.0], device=device))

    def test_neg(self, device):
        a = torch.tensor([1.0, -2.0, 3.0], device=device)
        assert torch.allclose(
            torch.neg(a), torch.tensor([-1.0, 2.0, -3.0], device=device)
        )


# ---------------------------------------------------------------------------
# 3. Matrix operations
# ---------------------------------------------------------------------------


class TestMatrixOps:
    # Issue #409: the two mm comparisons below are the only ones in this file
    # that are checked at 1e-3 rather than 1e-4. A vendor cube unit may run the
    # fp32 GEMM with a down-precision math mode -- Ascend's aclnnMm is called
    # with cube_math_type=1 (ALLOW_FP32_DOWN_PRECISION, hf32), which the repo
    # records as ~2.5e-3 against CPU -- so 1e-4 flagged the platform's chosen
    # accumulation as a torch_fl defect. 1e-3 is the tolerance the
    # marker-selected ops/test_mm_dispatch.py::test_mm_matches_cuda_ref already
    # uses for this exact shape and comparison.
    def test_mm(self, device):
        torch.manual_seed(0)
        a = torch.randn(64, 128, device=device)
        b = torch.randn(128, 32, device=device)
        out = torch.mm(a, b)
        assert out.shape == (64, 32)
        assert torch.allclose(
            out.cpu(), torch.mm(a.cpu(), b.cpu()), rtol=1e-3, atol=1e-3
        )

    def test_mm_out(self, device):
        torch.manual_seed(1)
        a = torch.randn(32, 16, device=device)
        b = torch.randn(16, 8, device=device)
        out = torch.empty(32, 8, device=device)
        torch.mm(a, b, out=out)
        assert torch.allclose(
            out.cpu(), torch.mm(a.cpu(), b.cpu()), rtol=1e-3, atol=1e-3
        )

    def test_matmul(self, device):
        a = torch.randn(64, 128, device=device)
        b = torch.randn(128, 32, device=device)
        assert torch.matmul(a, b).shape == (64, 32)

    def test_bmm(self, device):
        a = torch.randn(4, 64, 128, device=device)
        b = torch.randn(4, 128, 32, device=device)
        assert torch.bmm(a, b).shape == (4, 64, 32)


# ---------------------------------------------------------------------------
# 4. Reduction operations
# ---------------------------------------------------------------------------


class TestReductions:
    def test_sum(self, device):
        x = torch.ones(100, device=device)
        assert x.sum().item() == 100.0

    def test_mean(self, device):
        x = torch.empty(100, device=device)
        x.fill_(5.0)
        assert x.mean().item() == pytest.approx(5.0)

    def test_mean_dim(self, device):
        torch.manual_seed(0)
        x = torch.randn(16, 32, device=device)
        out = torch.mean(x, dim=-1)
        assert out.shape == (16,)
        assert torch.allclose(out.cpu(), x.cpu().mean(dim=-1), rtol=1e-4, atol=1e-4)

    def test_mean_dim_keepdim(self, device):
        torch.manual_seed(1)
        x = torch.randn(8, 12, device=device)
        out = torch.mean(x, dim=0, keepdim=True)
        assert out.shape == (1, 12)
        assert torch.allclose(
            out.cpu(), x.cpu().mean(dim=0, keepdim=True), rtol=1e-4, atol=1e-4
        )

    def test_max(self, device):
        x = torch.tensor([1.0, 5.0, 3.0], device=device)
        assert x.max().item() == 5.0

    def test_min(self, device):
        x = torch.tensor([1.0, 5.0, 3.0], device=device)
        assert x.min().item() == 1.0


# ---------------------------------------------------------------------------
# 5. Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_embedding_basic(self, device):
        torch.manual_seed(0)
        weight = torch.randn(100, 64, device=device)
        indices = torch.tensor([0, 5, 10, 99], device=device)
        out = F.embedding(indices, weight)
        assert out.shape == (4, 64)
        assert out.device.type == "flagos"
        expected = weight.index_select(0, indices)
        assert torch.allclose(out.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)

    def test_embedding_2d_indices(self, device):
        torch.manual_seed(1)
        weight = torch.randn(50, 32, device=device)
        indices = torch.randint(0, 50, (4, 8), device=device)
        out = F.embedding(indices, weight)
        assert out.shape == (4, 8, 32)
        assert torch.allclose(
            out.cpu(), F.embedding(indices.cpu(), weight.cpu()), rtol=1e-5, atol=1e-5
        )


# ---------------------------------------------------------------------------
# 6. Shape operations
# ---------------------------------------------------------------------------


class TestShapeOps:
    def test_reshape(self, device):
        x = torch.randn(4, 8, device=device)
        assert x.reshape(2, 16).shape == (2, 16)

    def test_view(self, device):
        x = torch.randn(4, 8, device=device)
        assert x.view(32).shape == (32,)

    def test_transpose(self, device):
        x = torch.randn(4, 8, device=device)
        assert x.t().shape == (8, 4)

    def test_unsqueeze_squeeze(self, device):
        x = torch.randn(4, device=device)
        assert x.unsqueeze(0).shape == (1, 4)
        assert x.unsqueeze(0).squeeze(0).shape == (4,)

    def test_expand(self, device):
        x = torch.randn(1, 4, device=device)
        assert x.expand(3, 4).shape == (3, 4)

    def test_cat(self, device):
        a = torch.randn(2, 4, device=device)
        b = torch.randn(3, 4, device=device)
        assert torch.cat([a, b], dim=0).shape == (5, 4)

    def test_stack(self, device):
        a = torch.randn(4, device=device)
        b = torch.randn(4, device=device)
        assert torch.stack([a, b]).shape == (2, 4)


# ---------------------------------------------------------------------------
# 7. Copy and transfer
# ---------------------------------------------------------------------------


class TestCopyTransfer:
    def test_to_cpu(self, device):
        x = torch.randn(4, 4, device=device)
        assert x.cpu().device.type == "cpu"

    def test_to_device(self, device):
        x = torch.randn(4, 4, device="cpu")
        assert x.to(device).device.type == "flagos"

    def test_clone(self, device):
        x = torch.randn(4, 4, device=device)
        y = x.clone()
        assert y.data_ptr() != x.data_ptr()
        assert torch.allclose(x, y)

    def test_to_copy_cpu_to_device_with_dtype(self, device):
        cpu_t = torch.randn(16, 16)
        dev_t = cpu_t.to(device=device, dtype=torch.float16)
        assert dev_t.device.type == device.split(":")[0]
        assert dev_t.dtype == torch.float16
        assert torch.allclose(dev_t.cpu().float(), cpu_t, atol=1e-2, rtol=1e-2)

    def test_to_copy_device_to_cpu_with_dtype(self, device):
        dev_t = torch.randn(16, 16, device=device)
        cpu_t = dev_t.to(device="cpu", dtype=torch.float64)
        assert cpu_t.device.type == "cpu"
        assert cpu_t.dtype == torch.float64
        assert torch.allclose(cpu_t.float(), dev_t.cpu(), atol=1e-5, rtol=1e-5)

    def test_to_copy_device_to_device_copy(self, device):
        dev_t = torch.randn(16, 16, device=device)
        copied = dev_t.to(device=device, copy=True)
        assert copied.device.type == device.split(":")[0]
        assert copied.data_ptr() != dev_t.data_ptr()
        assert torch.allclose(copied.cpu(), dev_t.cpu(), atol=1e-5, rtol=1e-5)

    def test_to_copy_non_contiguous_roundtrip(self, device):
        cpu_t = torch.randn(8, 16).t()
        assert not cpu_t.is_contiguous()
        roundtrip = cpu_t.to(device).cpu()
        assert torch.allclose(roundtrip, cpu_t, atol=1e-5, rtol=1e-5)

    def test_to_copy_empty_tensor(self, device):
        cpu_t = torch.empty(0, 3)
        dev_t = cpu_t.to(device)
        assert dev_t.device.type == device.split(":")[0]
        assert dev_t.shape == cpu_t.shape
        assert dev_t.cpu().shape == cpu_t.shape


# ---------------------------------------------------------------------------
# 8. Indexing and slicing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_basic_index(self, device):
        x = torch.arange(10, device=device, dtype=torch.float32)
        assert x[5].item() == 5.0

    def test_slice(self, device):
        x = torch.arange(10, device=device, dtype=torch.float32)
        assert x[2:5].shape == (3,)

    def test_masked_select(self, device):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        mask = x > 2
        assert torch.masked_select(x, mask).shape[0] == 2


# ---------------------------------------------------------------------------
# 9. Activation functions
# ---------------------------------------------------------------------------


class TestActivations:
    def test_relu(self, device):
        x = torch.tensor([-1.0, 0.0, 1.0], device=device)
        assert torch.allclose(
            torch.relu(x), torch.tensor([0.0, 0.0, 1.0], device=device)
        )

    def test_sigmoid(self, device):
        x = torch.zeros(4, device=device)
        assert torch.allclose(
            torch.sigmoid(x), torch.full((4,), 0.5, device=device), atol=1e-5
        )

    def test_softmax(self, device):
        x = torch.randn(4, device=device)
        s = torch.softmax(x, dim=0)
        assert s.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_silu(self, device):
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)
        assert torch.allclose(F.silu(x).cpu(), F.silu(x.cpu()), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 10. Unary / elementwise ops (Qwen3 inference path)
# ---------------------------------------------------------------------------


class TestInferencePathOps:
    """Ops commonly hit in Qwen3 forward (RoPE, mask, activations)."""

    def test_cos(self, device):
        x = torch.tensor([0.0, 1.5708, 3.14159], device=device)
        assert torch.allclose(
            torch.cos(x).cpu(), torch.cos(x.cpu()), rtol=1e-4, atol=1e-4
        )

    def test_sin(self, device):
        x = torch.tensor([0.0, 1.5708, 3.14159], device=device)
        assert torch.allclose(
            torch.sin(x).cpu(), torch.sin(x.cpu()), rtol=1e-4, atol=1e-4
        )

    def test_rsqrt(self, device):
        x = torch.tensor([1.0, 4.0, 16.0], device=device)
        assert torch.allclose(
            torch.rsqrt(x).cpu(), torch.rsqrt(x.cpu()), rtol=1e-4, atol=1e-4
        )

    def test_pow_scalar(self, device):
        x = torch.tensor([2.0, 3.0, 4.0], device=device)
        assert torch.allclose(
            torch.pow(x, 2).cpu(), torch.pow(x.cpu(), 2), rtol=1e-4, atol=1e-4
        )

    def test_mul_scalar(self, device):
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        assert torch.allclose((x * 2.0).cpu(), x.cpu() * 2.0, rtol=1e-4, atol=1e-4)

    def test_bitwise_and(self, device):
        a = torch.tensor([0b1010, 0b1100, 0b1111], dtype=torch.int32, device=device)
        b = torch.tensor([0b1001, 0b0100, 0b1010], dtype=torch.int32, device=device)
        assert torch.equal(
            a & b,
            torch.tensor([0b1000, 0b0100, 0b1010], dtype=torch.int32, device=device),
        )

    def test_where(self, device):
        cond = torch.tensor([True, False, True, False], device=device)
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        b = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device)
        out = torch.where(cond, a, b)
        expected = torch.where(cond.cpu(), a.cpu(), b.cpu())
        assert torch.allclose(out.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_index_tensor(self, device):
        x = torch.arange(10, device=device, dtype=torch.float32)
        indices = torch.tensor([1, 3, 7], device=device)
        assert torch.allclose(x[indices].cpu(), x.cpu()[indices.cpu()])


# ---------------------------------------------------------------------------
# 11. Device properties
# ---------------------------------------------------------------------------


class TestDeviceProperties:
    def test_device_type(self, device):
        x = torch.randn(4, device=device)
        assert x.device.type == "flagos"

    def test_is_not_cpu(self, device):
        assert not torch.randn(4, device=device).is_cpu


# ---------------------------------------------------------------------------
# 12. Autograd
# ---------------------------------------------------------------------------


class TestAutograd:
    def test_grad_computed(self, device):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None

    def test_grad_shape(self, device):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        (x * x).sum().backward()
        assert x.grad.shape == (4, 4)

    def test_grad_values(self, device):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        (x * x).sum().backward()
        assert torch.allclose(x.grad, 2 * x.detach(), atol=1e-5)


# ---------------------------------------------------------------------------
# 13. Synchronization
# ---------------------------------------------------------------------------


class TestSync:
    def test_synchronize(self, device):
        torch_fl.flagos.synchronize()
