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
        "dense_offset": lambda x: x[1:],
        "offset_slice": lambda x: x[..., 1::2].transpose(0, 2),
        "strided_axes": lambda x: x[::2, ::2, 1::2],
        "expand": lambda x: x[:, :1, :].expand_as(x),
        "scalar": lambda x: x[1, 1, 1],
        "strided_vector": lambda x: x[1, 1, 1::2],
        "empty": lambda x: x[:, :0, :].transpose(0, 2),
    }
    dtypes = (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.uint8,
        torch.int64,
        torch.bool,
    )
    values = torch.arange(-30, 30).reshape(3, 4, 5)
    with torch.no_grad():
        for dtype in dtypes:
            cpu = (values % 2 == 0) if dtype == torch.bool else values.to(dtype)
            device = cpu.to(DEVICE)
            for layout, view in views.items():
                source, reference = view(device), view(cpu)
                case = f"{layout}, {dtype}"
                assert source.stride() == reference.stride(), case
                assert source.storage_offset() == reference.storage_offset(), case
                compare = partial(
                    torch.testing.assert_close,
                    atol=0,
                    rtol=0,
                    msg=lambda message: f"{case}: {message}",
                )

                packed, expected_packed = source.contiguous(), reference.contiguous()
                assert packed.device == source.device, case
                assert packed.stride() == expected_packed.stride(), case
                compare(packed.cpu(), expected_packed)
                for memory_format in (torch.preserve_format, torch.contiguous_format):
                    cloned = (
                        source.clone()
                        if memory_format == torch.preserve_format
                        else source.clone(memory_format=memory_format)
                    )
                    expected = reference.clone(memory_format=memory_format)
                    assert cloned.device == source.device, case
                    assert cloned.stride() == expected.stride(), case
                    if source.numel():
                        assert cloned.data_ptr() != source.data_ptr(), case
                    compare(cloned.cpu(), expected)
                    compare(
                        source.cpu(memory_format=memory_format),
                        reference.to("cpu", copy=True, memory_format=memory_format),
                        check_stride=True,
                    )

                dense_storage = torch.zeros(source.numel() + 2, dtype=dtype)
                dense_device_storage = dense_storage.to(DEVICE)
                dense = dense_device_storage[1:-1].view(source.shape)
                assert dense.copy_(source) is dense, case
                dense_storage[1:-1].copy_(reference.reshape(-1))
                compare(dense_device_storage.cpu(), dense_storage)

                # Use separate storage for each copy so missed writes and
                # writes outside a non-overlapping destination view are visible.
                for copy_source, expected_source in (
                    (source, reference),
                    (packed, expected_packed),
                ):
                    target = torch.zeros_like(cpu)
                    target_device = target.to(DEVICE)
                    target_view = (
                        target_device if layout == "expand" else view(target_device)
                    )
                    reference_view = target if layout == "expand" else view(target)
                    assert target_view.copy_(copy_source) is target_view, case
                    reference_view.copy_(expected_source)
                    assert target_view.stride() == reference_view.stride(), case
                    compare(target_device.cpu(), target)


def _check_copy_stream_order():
    # Use the measured large-copy shape to exercise the asynchronous transfer.
    host = torch.linspace(-1, 1, 200 * 40999 * 3).reshape(200, 40999, 3)
    values = (host, -host)
    inputs = [(value.to(DEVICE), value) for value in values]
    device = inputs[0][0].device
    side = torch.Stream(device=device)
    side.wait_stream(torch.accelerator.current_stream(device))
    with side, torch.no_grad():
        target = torch.empty_like(inputs[0][0])
        for name, copy in (("clone", torch.clone), ("copy_", target.copy_)):
            for source, expected in inputs * 2:
                # Two sign changes give an exact reference. Alternating inputs
                # exposes a copy that reads an intermediate from a prior call.
                result = torch.neg(copy(torch.neg(source)))
                torch.testing.assert_close(
                    result.cpu(),
                    expected,
                    atol=0,
                    rtol=0,
                    msg=lambda message: f"{name}: {message}",
                )


@pytest.mark.anyplatform
def test_strided_transfer(run_on_wheel):
    run_on_wheel(_check_strided_transfer)


@pytest.mark.anyplatform
def test_copy_stream_order(run_on_wheel):
    run_on_wheel(_check_copy_stream_order)
