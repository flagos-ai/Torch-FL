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

"""Copy flagos tensors and compare them with the same CPU inputs."""

from functools import partial

import pytest
import torch

import torch_fl  # noqa: F401 -- importing claims the flagos device

DEVICE = "flagos:0"


def _cpu_tensor(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(8, 8)


@pytest.mark.anyplatform
def test_host_to_device_round_trip():
    cpu = _cpu_tensor(0)
    device = cpu.to(DEVICE)
    assert device.device.type == "flagos"
    torch.testing.assert_close(device.cpu(), cpu)


@pytest.mark.anyplatform
def test_device_to_device_copy():
    cpu = _cpu_tensor(1)
    device = cpu.to(DEVICE)
    copied = device.clone()
    assert copied.device.type == "flagos"
    torch.testing.assert_close(copied.cpu(), cpu)


@pytest.mark.anyplatform
def test_non_contiguous_source_round_trip():
    cpu = _cpu_tensor(2)
    transposed_cpu = cpu.t()
    assert not transposed_cpu.is_contiguous()
    device = transposed_cpu.to(DEVICE)
    torch.testing.assert_close(device.cpu(), transposed_cpu)


@pytest.fixture(params=("ordinary", "native"))
def run_on_wheel(request, pytestconfig):
    if request.param == "native":
        from ._native_host import run_native_host_case

        return partial(run_native_host_case, pytestconfig=pytestconfig)
    return lambda case: case()


def _check_strided_transfer():
    views = {
        "transpose": lambda x: x.transpose(0, 2),
        "offset_slice": lambda x: x[..., 1::2].transpose(0, 2),
        "expand": lambda x: x[:, :1, :].expand_as(x),
        "empty": lambda x: x[:, :0, :],
    }
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            cpu = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).to(dtype)
            device = cpu.to(DEVICE)
            for layout, view in views.items():
                source, reference = view(device), view(cpu)
                assert source.stride() == reference.stride(), layout
                assert source.storage_offset() == reference.storage_offset(), layout
                for memory_format in (torch.preserve_format, torch.contiguous_format):
                    cloned = source.clone(memory_format=memory_format)
                    expected = reference.clone(memory_format=memory_format)
                    assert cloned.device == source.device, layout
                    assert cloned.stride() == expected.stride(), layout
                    if source.numel():
                        assert cloned.data_ptr() != source.data_ptr(), layout
                    torch.testing.assert_close(cloned.cpu(), expected, atol=0, rtol=0)
                    torch.testing.assert_close(
                        source.cpu(memory_format=memory_format),
                        reference.to("cpu", copy=True, memory_format=memory_format),
                        atol=0,
                        rtol=0,
                        check_stride=True,
                    )

                target = torch.full_like(cpu, -1)
                target_device = target.to(DEVICE)
                # An expanded source is valid, while the destination must be
                # writable without overlapping elements.
                target_view = (
                    target_device if layout == "expand" else view(target_device)
                )
                reference_view = target if layout == "expand" else view(target)
                assert target_view.copy_(source) is target_view, layout
                reference_view.copy_(reference)
                assert target_view.stride() == reference_view.stride(), layout
                torch.testing.assert_close(target_device.cpu(), target, atol=0, rtol=0)


def _check_copy_stream_order():
    # Use the measured large-copy shape to exercise the asynchronous transfer.
    host = torch.linspace(-1, 1, 200 * 40999 * 3).reshape(200, 40999, 3)
    expected = host.cos().cos()
    source = host.to(DEVICE)
    side = torch.Stream(device=source.device)
    side.wait_stream(torch.accelerator.current_stream(source.device))
    with side, torch.no_grad():
        target = torch.empty_like(source)
        for name, copy in (("clone", torch.clone), ("copy_", target.copy_)):
            for _ in range(4):
                result = torch.cos(copy(torch.cos(source)))
                torch.testing.assert_close(
                    result.cpu(),
                    expected,
                    atol=1e-6,
                    rtol=1e-6,
                    msg=lambda message: f"{name}: {message}",
                )


@pytest.mark.anyplatform
def test_strided_transfer(run_on_wheel):
    run_on_wheel(_check_strided_transfer)


@pytest.mark.anyplatform
def test_copy_stream_order(run_on_wheel):
    run_on_wheel(_check_copy_stream_order)
