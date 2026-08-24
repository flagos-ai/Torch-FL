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

"""One public AMP contract shared by every FlagOS hardware backend.

Every backend registers the same `AutocastPrivateUse1` policy lists, so the
observable `torch.amp` behavior is asserted by the same test functions on all
hardware. Only genuine route differences select or skip individual cases,
through the capabilities in `amp_support`.
"""

import pytest
import torch
import torch.nn.functional as F

from amp_support import require


AMP_DTYPES = (torch.float16, torch.bfloat16)

pytestmark = pytest.mark.amp


@pytest.mark.anyplatform
def test_autocast_device_registration():
    """The flagos backend advertises the standard AMP entry points."""
    assert torch.amp.is_autocast_available("flagos")
    assert torch.flagos.get_amp_supported_dtype() == list(AMP_DTYPES)
    assert torch.get_autocast_dtype("flagos") == torch.float16


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_autocast_rejects_unsupported_target_dtype(dtype):
    """Only float16 and bfloat16 are valid autocast targets."""
    with pytest.warns(UserWarning, match="target dtype is not supported"):
        with torch.autocast("flagos", dtype=dtype):
            assert not torch.is_autocast_enabled("flagos")


@pytest.mark.anyplatform
def test_autocast_nested_state():
    """Nested autocast regions restore the enclosing state."""
    assert not torch.is_autocast_enabled("flagos")
    with torch.autocast("flagos", dtype=torch.float16):
        assert torch.is_autocast_enabled("flagos")
        assert torch.get_autocast_dtype("flagos") == torch.float16
        with torch.autocast("flagos", enabled=False):
            assert not torch.is_autocast_enabled("flagos")
        assert torch.is_autocast_enabled("flagos")
    assert not torch.is_autocast_enabled("flagos")


@pytest.mark.amp_device
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
def test_eager_dtype_preservation(dtype, amp_capabilities, amp_device):
    """Eager arithmetic outside autocast keeps the requested dtype."""
    require(amp_capabilities, "device")

    values = torch.ones(4, device=amp_device, dtype=dtype)
    assert (values + values).dtype == dtype


@pytest.mark.amp_device
def test_float64_copy_preserves_dtype(amp_capabilities, amp_device):
    """float64 survives the host round trip.

    GradScaler depends on this: it derives the inverse scale through
    `scale.double().reciprocal().float()`, so a backend that silently
    downcasts float64 cannot scale gradients correctly.
    """
    require(amp_capabilities, "device")

    cpu = torch.tensor([1.25, -2.5], dtype=torch.float64)
    device = cpu.to(amp_device)

    assert device.dtype == torch.float64
    torch.testing.assert_close(device.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.amp_device
@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_lower_precision_policy(dtype, amp_capabilities, amp_device):
    """Matmul-family ops run in the selected lower-precision dtype."""
    require(amp_capabilities, "device")

    a = torch.randn(32, 32, device=amp_device)
    b = torch.randn(32, 32, device=amp_device)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.mm(a, b).dtype == dtype
        assert F.linear(a, b).dtype == dtype


@pytest.mark.amp_device
@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_convolution_policy(dtype, amp_capabilities, amp_device):
    """Convolution follows the lower-precision policy where it is routed."""
    require(amp_capabilities, "convolution")

    image = torch.randn(2, 3, 8, 8, device=amp_device)
    weight = torch.randn(4, 3, 3, 3, device=amp_device)

    with torch.autocast("flagos", dtype=dtype):
        assert F.conv2d(image, weight).dtype == dtype


@pytest.mark.amp_device
@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_fp32_policy(dtype, amp_capabilities, amp_device):
    """Numerically sensitive ops are promoted to float32."""
    require(amp_capabilities, "device")

    x = torch.rand(64, device=amp_device, dtype=dtype) + 0.5
    normalized = torch.randn(4, 16, device=amp_device, dtype=dtype)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.log(x).dtype == torch.float32
        assert F.layer_norm(normalized, (16,)).dtype == torch.float32
        assert F.mse_loss(x, x).dtype == torch.float32


@pytest.mark.amp_device
@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_fp32_set_optional_dtype_policy(dtype, amp_capabilities, amp_device):
    """An explicit dtype argument overrides the float32 default."""
    require(amp_capabilities, "device")

    x = torch.randn(8, 16, device=amp_device, dtype=dtype)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.softmax(x, dim=-1).dtype == torch.float32
        assert torch.softmax(x, dim=-1, dtype=dtype).dtype == dtype


@pytest.mark.amp_device
@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_autocast_promote_policy(dtype, amp_capabilities, amp_device):
    """Mixed-precision inputs promote to the widest dtype."""
    require(amp_capabilities, "device")

    x = torch.randn(32, device=amp_device, dtype=dtype)
    y = torch.randn(32, device=amp_device, dtype=torch.float32)

    with torch.autocast("flagos", dtype=dtype):
        assert torch.atan2(x, y).dtype == torch.float32


@pytest.mark.amp_device
def test_unlisted_op_uses_backend_fallthrough(amp_capabilities, amp_device):
    """Ops outside the policy lists keep the backend's native result dtype."""
    require(amp_capabilities, "device")

    pred = torch.rand(32, device=amp_device, dtype=torch.float16)
    target = torch.rand(32, device=amp_device, dtype=torch.float16)

    with torch.autocast("flagos", dtype=torch.float16):
        loss = F.binary_cross_entropy(pred, target)

    # PrivateUse1 follows the standard policy lists. BCE is not in those lists,
    # so it remains usable through the backend fallthrough.
    assert loss.dtype == torch.float16


@pytest.mark.amp_device
@pytest.mark.amp_grad_scaler
def test_amp_unscale_detects_non_finite_values(amp_capabilities, amp_device):
    """The in-place unscale route reports overflow through found_inf."""
    require(amp_capabilities, "grad_scaler")

    values = torch.tensor([65536.0, float("inf")], device=amp_device)
    found_inf = torch.zeros((), device=amp_device)
    inv_scale = torch.tensor(1.0 / 65536.0, device=amp_device)

    torch._amp_foreach_non_finite_check_and_unscale_([values], found_inf, inv_scale)

    torch.testing.assert_close(values[0].cpu(), torch.tensor(1.0))
    assert torch.isinf(values[1]).item()
    assert found_inf.cpu().item() == 1.0


@pytest.mark.amp_device
@pytest.mark.amp_grad_scaler
def test_amp_unscale_out_variant(amp_capabilities, amp_device):
    """The `.out` overload unscales without mutating found_inf."""
    require(amp_capabilities, "grad_scaler")

    values = torch.tensor([4.0, float("inf")], device=amp_device)
    found_inf = torch.zeros((), device=amp_device)
    inv_scale = torch.tensor(0.25, device=amp_device)
    output = torch.empty_like(values)

    torch.ops.aten._amp_foreach_non_finite_check_and_unscale.out(
        [values], found_inf, inv_scale, out=[output]
    )

    torch.testing.assert_close(output[0].cpu(), torch.tensor(1.0))
    assert torch.isinf(output[1]).item()
    # The out overload follows the CPU reference and leaves found_inf unchanged;
    # GradScaler uses the in-place overload above for overflow detection.
    assert found_inf.cpu().item() == 0.0


@pytest.mark.amp_device
@pytest.mark.amp_grad_scaler
def test_autocast_grad_scaler_training_step(amp_capabilities, amp_device):
    """A full autocast plus GradScaler training step keeps parameters finite."""
    require(amp_capabilities, "grad_scaler")

    model = torch.nn.Linear(8, 4).to(amp_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("flagos", init_scale=8.0)
    x = torch.randn(2, 8, device=amp_device)
    target = torch.randn(2, 4, device=amp_device)

    with torch.autocast("flagos", dtype=torch.float16):
        output = model(x)
        loss = F.mse_loss(output, target)

    assert output.dtype == torch.float16
    assert loss.dtype == torch.float32
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert all(torch.isfinite(p).all().item() for p in model.parameters())


@pytest.mark.amp_device
@pytest.mark.amp_grad_scaler
def test_grad_scaler_finite_step_and_growth(amp_capabilities, amp_device):
    """A finite step applies the update and grows the scale."""
    require(amp_capabilities, "grad_scaler")

    param = torch.nn.Parameter(torch.tensor([2.0], device=amp_device))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scaler = torch.amp.GradScaler("flagos", init_scale=8.0, growth_interval=1)

    scaler.scale((param * param).sum()).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.testing.assert_close(param.detach().cpu(), torch.tensor([1.6]))
    assert scaler.get_scale() == 16.0


@pytest.mark.amp_device
@pytest.mark.amp_grad_scaler
def test_grad_scaler_overflow_skips_step_and_backs_off(amp_capabilities, amp_device):
    """An overflowing step is skipped and the scale backs off."""
    require(amp_capabilities, "grad_scaler")

    param = torch.nn.Parameter(torch.tensor([2.0], device=amp_device))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scaler = torch.amp.GradScaler("flagos", init_scale=8.0, backoff_factor=0.5)
    inf = torch.tensor(float("inf"), device=amp_device)

    scaler.scale(param.sum() * inf).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.testing.assert_close(param.detach().cpu(), torch.tensor([2.0]))
    assert scaler.get_scale() == 4.0
