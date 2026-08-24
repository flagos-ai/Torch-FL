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

import os

import pytest
import torch
import torch.nn.functional as F
import torch_fl


DEVICE = "flagos:0"
AMP_DTYPES = (torch.float16, torch.bfloat16)

# PPU shares the CUDA build selector, so its SDK marker distinguishes it from
# NVIDIA. Native AMP backends have explicit selectors. Device availability is
# enforced by integration conftest.
BUILD_ACCELERATOR = torch_fl._build_accelerator()
PPU_RUNTIME = bool(os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME"))
AMP_RUNTIME = (
    BUILD_ACCELERATOR
    in {
        "ascend",
        "dcu",
        "gcu",
        "metax",
        "musa",
    }
    or PPU_RUNTIME
)
pytestmark = pytest.mark.skipif(
    not AMP_RUNTIME,
    reason="AMP tests require an Ascend, DCU, GCU, MetaX, MUSA, or PPU runtime",
)


def test_amp_device_contract():
    assert torch.amp.is_autocast_available("flagos")
    assert torch.flagos.get_amp_supported_dtype() == list(AMP_DTYPES)
    assert torch.get_autocast_dtype("flagos") == torch.float16


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_autocast_disables_unsupported_dtype(dtype):
    with pytest.warns(UserWarning, match="target dtype is not supported"):
        with torch.autocast("flagos", dtype=dtype):
            assert not torch.is_autocast_enabled("flagos")


def test_float64_copy_preserves_dtype():
    cpu = torch.tensor([1.25, -2.5], dtype=torch.float64)
    device = cpu.to(DEVICE)

    assert device.dtype == torch.float64
    torch.testing.assert_close(device.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.bool,
    ],
)
def test_eager_dtype_preservation(dtype):
    values = torch.ones(4, device=DEVICE, dtype=dtype)
    result = values + values
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_lower_precision(dtype):
    a = torch.randn(32, 32, device=DEVICE)
    b = torch.randn(32, 32, device=DEVICE)
    image = torch.randn(2, 3, 8, 8, device=DEVICE)
    weight = torch.randn(4, 3, 3, 3, device=DEVICE)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.mm(a, b).dtype == dtype
        assert F.linear(a, b).dtype == dtype
        assert F.conv2d(image, weight).dtype == dtype


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_fp32(dtype):
    x = torch.rand(64, device=DEVICE, dtype=dtype) + 0.5
    normalized = torch.randn(4, 16, device=DEVICE, dtype=dtype)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.log(x).dtype == torch.float32
        assert F.layer_norm(normalized, (16,)).dtype == torch.float32
        assert F.mse_loss(x, x).dtype == torch.float32


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_fp32_set_optional_dtype(dtype):
    x = torch.randn(8, 16, device=DEVICE, dtype=dtype)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.softmax(x, dim=-1).dtype == torch.float32
        assert torch.softmax(x, dim=-1, dtype=dtype).dtype == dtype


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_promote(dtype):
    x = torch.randn(32, device=DEVICE, dtype=dtype)
    y = torch.randn(32, device=DEVICE, dtype=torch.float32)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.atan2(x, y).dtype == torch.float32


def test_autocast_nested_state():
    assert not torch.is_autocast_enabled("flagos")
    with torch.autocast("flagos", dtype=torch.float16):
        assert torch.is_autocast_enabled("flagos")
        assert torch.get_autocast_dtype("flagos") == torch.float16
        with torch.autocast("flagos", enabled=False):
            assert not torch.is_autocast_enabled("flagos")
        assert torch.is_autocast_enabled("flagos")
    assert not torch.is_autocast_enabled("flagos")


def test_binary_cross_entropy_uses_backend_fallthrough():
    pred = torch.rand(32, device=DEVICE, dtype=torch.float16)
    target = torch.rand(32, device=DEVICE, dtype=torch.float16)

    with torch.autocast("flagos", dtype=torch.float16):
        loss = F.binary_cross_entropy(pred, target)

    # PrivateUse1 follows the standard policy lists. BCE is not in those lists,
    # so it remains usable through the backend fallthrough and preserves the
    # backend's native result dtype.
    assert loss.dtype == torch.float16


def test_amp_unscale_detects_non_finite_values():
    values = torch.tensor([65536.0, float("inf")], device=DEVICE)
    found_inf = torch.zeros((), device=DEVICE)
    inv_scale = torch.tensor(1.0 / 65536.0, device=DEVICE)

    torch._amp_foreach_non_finite_check_and_unscale_([values], found_inf, inv_scale)

    torch.testing.assert_close(values[0].cpu(), torch.tensor(1.0))
    assert torch.isinf(values[1]).item()
    assert found_inf.cpu().item() == 1.0


def test_amp_unscale_out_variant():
    values = torch.tensor([4.0, float("inf")], device=DEVICE)
    found_inf = torch.zeros((), device=DEVICE)
    inv_scale = torch.tensor(0.25, device=DEVICE)
    output = torch.empty_like(values)

    torch.ops.aten._amp_foreach_non_finite_check_and_unscale.out(
        [values], found_inf, inv_scale, out=[output]
    )

    torch.testing.assert_close(output[0].cpu(), torch.tensor(1.0))
    assert torch.isinf(output[1]).item()
    # The out overload follows the CPU reference and leaves found_inf unchanged;
    # GradScaler uses the in-place overload above for overflow detection.
    assert found_inf.cpu().item() == 0.0


def test_autocast_grad_scaler_training_step():
    model = torch.nn.Linear(8, 4).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("flagos", init_scale=8.0)
    x = torch.randn(2, 8, device=DEVICE)
    target = torch.randn(2, 4, device=DEVICE)

    with torch.autocast("flagos", dtype=torch.float16):
        output = model(x)
        loss = F.mse_loss(output, target)

    assert output.dtype == torch.float16
    assert loss.dtype == torch.float32
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert all(torch.isfinite(p).all().item() for p in model.parameters())


def test_grad_scaler_finite_step_and_growth():
    param = torch.nn.Parameter(torch.tensor([2.0], device=DEVICE))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scaler = torch.amp.GradScaler("flagos", init_scale=8.0, growth_interval=1)

    scaler.scale((param * param).sum()).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.testing.assert_close(param.detach().cpu(), torch.tensor([1.6]))
    assert scaler.get_scale() == 16.0


def test_grad_scaler_overflow_skips_step_and_backs_off():
    param = torch.nn.Parameter(torch.tensor([2.0], device=DEVICE))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scaler = torch.amp.GradScaler("flagos", init_scale=8.0, backoff_factor=0.5)
    inf = torch.tensor(float("inf"), device=DEVICE)

    scaler.scale(param.sum() * inf).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.testing.assert_close(param.detach().cpu(), torch.tensor([2.0]))
    assert scaler.get_scale() == 4.0
