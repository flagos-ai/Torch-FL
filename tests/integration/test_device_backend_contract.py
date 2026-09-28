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

"""Small cross-platform contract for the public ``flagos`` tensor surface.

The core cases gate correctness, the smoke case checks usable public device
methods, and the final diagnostic records a capability without gating CI. This
does not replace the larger operator, dtype, or multi-device suites.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl

pytestmark = pytest.mark.anyplatform
DEVICE = "flagos:0"


@pytest.mark.backend_contract
def test_registration_allocation_and_empty_tensor():
    assert torch._C._get_privateuse1_backend_name() == "flagos"
    assert torch_fl.flagos.is_available()
    assert torch_fl.flagos.device_count() > 0

    values = torch.zeros((2, 3), dtype=torch.float32, device=DEVICE)
    assert values.device == torch.device(DEVICE)
    assert values.dtype == torch.float32
    torch.testing.assert_close(values.cpu(), torch.zeros((2, 3)))

    empty = torch.empty((0, 3), dtype=torch.int64, device=DEVICE)
    assert empty.shape == (0, 3)
    assert empty.dtype == torch.int64
    assert empty.numel() == 0
    assert empty.cpu().shape == empty.shape


@pytest.mark.backend_contract
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int64, torch.bool]
)
def test_host_device_round_trip_preserves_dtype_and_values(dtype):
    cpu = torch.tensor([0, 1, 2, 3], dtype=dtype)
    device = cpu.to(DEVICE)

    assert device.device == torch.device(DEVICE)
    assert device.dtype == dtype
    torch.testing.assert_close(device.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.backend_contract
def test_factory_uses_explicit_and_current_device_index():
    previous = torch_fl.flagos.current_device()
    try:
        torch_fl.flagos.set_device(0)
        explicit = torch.ones(4, dtype=torch.float32, device=DEVICE)
        implicit = torch.ones(4, dtype=torch.float32, device="flagos")
        assert explicit.device.index == implicit.device.index == 0
        torch.testing.assert_close(explicit.cpu(), torch.ones(4))
        torch.testing.assert_close(implicit.cpu(), torch.ones(4))
    finally:
        torch_fl.flagos.set_device(previous)


@pytest.mark.backend_contract
def test_factory_honors_nonzero_device_index():
    if torch_fl.flagos.device_count() < 2:
        pytest.skip("explicit flagos:1 factory requires two flagos devices")

    previous = torch_fl.flagos.current_device()
    try:
        torch_fl.flagos.set_device(0)
        other = torch.ones(4, dtype=torch.float32, device="flagos:1")
        assert other.device.index == 1
        torch.testing.assert_close(other.cpu(), torch.ones(4))
        torch_fl.flagos.set_device(1)
        current = torch.ones(4, dtype=torch.float32, device="flagos")
        assert current.device.index == 1
        torch.testing.assert_close(current.cpu(), torch.ones(4))
    finally:
        torch_fl.flagos.set_device(previous)


@pytest.mark.backend_contract
def test_view_reshape_transpose_and_stride_match_cpu():
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    device = cpu.to(DEVICE)
    view = device.view(2, 6)
    transposed = device.transpose(0, 1)
    reshaped = transposed.reshape(2, 6)

    assert view.stride() == cpu.view(2, 6).stride()
    assert transposed.shape == (4, 3)
    assert transposed.stride() == cpu.transpose(0, 1).stride()
    assert not transposed.is_contiguous()
    torch.testing.assert_close(view.cpu(), cpu.view(2, 6), rtol=0, atol=0)
    torch.testing.assert_close(transposed.cpu(), cpu.transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(
        reshaped.cpu(), cpu.transpose(0, 1).reshape(2, 6), rtol=0, atol=0
    )


@pytest.mark.backend_contract
def test_basic_and_advanced_indexing_match_cpu():
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    device = cpu.to(DEVICE)
    indices = torch.tensor([2, 0], dtype=torch.int64, device=DEVICE)

    torch.testing.assert_close(device[1, 1:3].cpu(), cpu[1, 1:3], rtol=0, atol=0)
    torch.testing.assert_close(
        device[indices].cpu(), cpu[indices.cpu()], rtol=0, atol=0
    )
    assert device[2, 3].item() == cpu[2, 3].item()


@pytest.mark.backend_contract
def test_inplace_and_out_reuse_destination():
    cpu = torch.tensor([1.0, 2.0, 3.0])
    increment = torch.tensor([4.0, 5.0, 6.0])
    device = cpu.to(DEVICE)
    device_increment = increment.to(DEVICE)

    device.add_(device_increment)
    expected = cpu + increment
    torch.testing.assert_close(device.cpu(), expected, rtol=0, atol=0)

    destination = torch.empty_like(device)
    returned = torch.add(device, device_increment, out=destination)
    assert returned is destination
    assert destination.device == torch.device(DEVICE)
    torch.testing.assert_close(destination.cpu(), expected + increment, rtol=0, atol=0)


@pytest.mark.backend_contract
def test_cpu_scalar_tensor_can_join_device_add():
    device = torch.tensor([1.0, 2.0], device=DEVICE)
    scalar = torch.tensor(3.0)
    torch.testing.assert_close(
        torch.add(device, scalar).cpu(), torch.tensor([4.0, 5.0]), rtol=0, atol=0
    )


@pytest.mark.backend_contract
def test_minimal_autograd_forward_and_backward():
    cpu = torch.tensor([1.0, 2.0, 3.0])
    device = cpu.to(DEVICE).requires_grad_()

    loss = (device * device).sum()
    assert loss.item() == 14.0
    loss.backward()

    assert device.grad is not None
    assert device.grad.device == device.device
    torch.testing.assert_close(device.grad.cpu(), 2 * cpu, rtol=0, atol=0)


@pytest.mark.backend_contract
@pytest.mark.parametrize(
    "operation",
    ["functional", "inplace", "out_operand", "out_destination", "scalar_out"],
)
def test_mixed_device_arithmetic_rejects_mismatch_in_subprocess(operation):
    if torch_fl.flagos.device_count() < 2:
        pytest.skip("mixed-device arithmetic requires two flagos devices")

    script = """
import torch_fl
import torch
import sys

first = torch.ones(2, device='flagos:0')
second = torch.ones(2, device='flagos:1')
operations = {
    'functional': lambda: torch.add(first, second),
    'inplace': lambda: first.add_(second),
    'out_operand': lambda: torch.add(first, second, out=torch.empty_like(first)),
    'out_destination': lambda: torch.add(first, first, out=torch.empty_like(second)),
    'scalar_out': lambda: torch.add(first, 1.0, out=torch.empty_like(second)),
}
try:
    operations[sys.argv[1]]()
except RuntimeError as error:
    assert 'device' in str(error).lower(), str(error)
else:
    raise AssertionError(f'mixed-device {sys.argv[1]} add silently succeeded')
"""
    env = os.environ.copy()
    env["FLAGOS_STARTUP_PROFILE"] = "minimal"
    result = subprocess.run(
        [sys.executable, "-c", script, operation],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.backend_smoke
def test_public_device_methods_do_not_raise():
    assert isinstance(torch_fl.flagos.current_device(), int)
    assert torch_fl.flagos.get_device_properties(0) is not None
    torch_fl.flagos.synchronize()


@pytest.mark.backend_capability
def test_complex_elementwise_capability_is_reported(record_property):
    """Complex compute is optional; report it after every gated case has run."""
    cpu = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex64)
    try:
        device = cpu.to(DEVICE)
        observed = (device + device).cpu()
    except Exception as error:  # noqa: BLE001 - this is a diagnostic case
        status = f"unavailable: {type(error).__name__}: {error}"
    else:
        try:
            torch.testing.assert_close(observed, cpu + cpu, rtol=0, atol=0)
        except AssertionError as error:
            status = f"incorrect: {error}"
        else:
            status = "supported"
    record_property("complex_elementwise", status)
    print(f"complex_elementwise: {status}")
