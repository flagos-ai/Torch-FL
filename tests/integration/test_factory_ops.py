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
    pytest tests/integration/test_factory_ops.py -v
"""

import pytest
import torch
import torch_fl


def _masked_select_is_served() -> bool:
    """Whether this wheel has a route for `masked_select` on this case's shape.

    Asked by calling the operator, on the 4-element shape the case uses: the
    conf's route is not enough to answer it. The PPU lane (run 40171498946)
    routes `masked_select` to FlagGems in `backends_thead.conf` and still raised
    `DispatchStub: missing kernel for cuda`, because that wheel imported without
    the FlagGems Python runtime -- the call reaches ATen's CUDA dispatch stub,
    which a CPU-only torch has no kernel for. A probe on a 1-element input takes
    the other arm and answers "served" for a shape the case does not use
    (`test_dtype_coverage.py` measured exactly that), so the probe below asks
    the same call the case asks. Same probe as test_ops.py's.
    """
    try:
        tensor = torch.ones(4, device="flagos:0")
        mask = torch.tensor([False, False, True, True], device="flagos:0")
        out = torch.ops.aten.masked_select(tensor, mask)
    except (NotImplementedError, RuntimeError):
        return False
    return out.device.type == "flagos" and out.numel() == 2


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


# ---------------------------------------------------------------------------
# 3. Matrix operations
# ---------------------------------------------------------------------------


class TestMatrixOps:
    def test_mm(self, device):
        a = torch.randn(64, 128, device=device)
        b = torch.randn(128, 32, device=device)
        assert torch.mm(a, b).shape == (64, 32)

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
        x = torch.ones(100, device=device) * 5.0
        assert x.mean().item() == pytest.approx(5.0)

    def test_max(self, device):
        x = torch.tensor([1.0, 5.0, 3.0], device=device)
        assert x.max().item() == 5.0

    def test_min(self, device):
        x = torch.tensor([1.0, 5.0, 3.0], device=device)
        assert x.min().item() == 1.0


# ---------------------------------------------------------------------------
# 5. Shape operations
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
# 6. Copy and transfer
# ---------------------------------------------------------------------------


class TestCopyTransfer:
    def test_cpu_and_device_have_compatible_shallow_copy_types(self, device):
        cpu_tensor = torch.ones(1)
        device_tensor = cpu_tensor.to(device)

        assert torch._has_compatible_shallow_copy_type(cpu_tensor, device_tensor)
        assert torch._has_compatible_shallow_copy_type(device_tensor, cpu_tensor)

    def test_module_to_device_preserves_tied_parameters(self, device):
        previous = torch.__future__.get_swap_module_params_on_conversion()
        torch.__future__.set_swap_module_params_on_conversion(False)

        class TiedParameters(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(4, 4, bias=False)
                self.second = torch.nn.Linear(4, 4, bias=False)
                self.second.weight = self.first.weight

        try:
            model = TiedParameters().to(device)
            assert model.first.weight is model.second.weight
            assert not torch.__future__.get_swap_module_params_on_conversion()
        finally:
            torch.__future__.set_swap_module_params_on_conversion(previous)

    def test_module_to_device_migrates_parameters_in_place(self, device):
        previous = torch.__future__.get_swap_module_params_on_conversion()
        torch.__future__.set_swap_module_params_on_conversion(True)

        try:
            model = torch.nn.Linear(4, 2)
            weight = model.weight
            bias = model.bias
            parameters = list(model.parameters())
            expected = {
                name: value.detach().clone() for name, value in model.named_parameters()
            }

            model.to(device)

            assert model.weight is weight
            assert model.bias is bias
            assert list(model.parameters()) == parameters
            assert all(
                parameter.device.type == "flagos" for parameter in model.parameters()
            )
            for name, value in model.named_parameters():
                actual_value = value.cpu()
                assert torch.equal(actual_value, expected[name])
                del actual_value

            inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            # Nothing is freed before the comparison: `reference` is owned by
            # its own call here, but the earlier shape of this block released
            # the input tensors first and could only mismatch on a backend that
            # reads them after the fact. The names are unbound after the assert.
            device_weight = weight.detach().clone()
            device_bias = bias.detach().clone()
            reference = torch.ops.aten.linear(
                inputs, expected["weight"], expected["bias"]
            )
            with torch.no_grad():
                actual = torch.ops.aten.linear(
                    inputs.to(device),
                    device_weight,
                    device_bias,
                ).cpu()
            # Same weights, same inputs, two summation orders. Printing the
            # difference as well as the values matters: a one-ulp disagreement
            # renders as two identical tensors, and the reader needs to know
            # it is looking at the last bit rather than at a wrong result.
            assert torch.allclose(actual, reference), (
                f"linear on {device} differs from CPU: {actual} vs {reference} "
                f"(max abs diff {(actual - reference).abs().max().item():.3g})"
            )
            del actual, reference, inputs, device_weight, device_bias

            model.cpu()
            assert model.weight is weight
            assert model.bias is bias
        finally:
            torch.__future__.set_swap_module_params_on_conversion(previous)

    def test_module_cpu_after_forward(self, device):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
        ).to(device)
        output = model(torch.randn(4, 8, device=device)).sum()

        model.cpu()

        assert output.grad_fn is not None
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert not torch.__future__.get_swap_module_params_on_conversion()

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
# 7. Indexing and slicing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_basic_index(self, device):
        x = torch.arange(10, device=device, dtype=torch.float32)
        assert x[5].item() == 5.0

    def test_slice(self, device):
        x = torch.arange(10, device=device, dtype=torch.float32)
        assert x[2:5].shape == (3,)

    @pytest.mark.skipif(
        not _masked_select_is_served(),
        reason="no route serves masked_select on this wheel's build and conf, "
        "so the call reaches ATen's CUDA dispatch stub, which a CPU-only torch "
        "has no kernel for",
    )
    def test_masked_select(self, device):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        mask = x > 2
        assert torch.masked_select(x, mask).shape[0] == 2


# ---------------------------------------------------------------------------
# 8. Activation functions
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


# ---------------------------------------------------------------------------
# 9. Device properties
# ---------------------------------------------------------------------------


class TestDeviceProperties:
    def test_device_type(self, device):
        x = torch.randn(4, device=device)
        assert x.device.type == "flagos"

    def test_is_not_cpu(self, device):
        assert not torch.randn(4, device=device).is_cpu


# ---------------------------------------------------------------------------
# 10. Autograd
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
# 11. Synchronization
# ---------------------------------------------------------------------------


class TestSync:
    def test_synchronize(self, device):
        torch_fl.flagos.synchronize()
