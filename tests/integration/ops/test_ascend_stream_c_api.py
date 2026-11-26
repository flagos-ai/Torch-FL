"""FlagCX and the Python stream context share the same Ascend registry."""

import ctypes
from pathlib import Path

import pytest
import torch
import torch_fl


pytestmark = pytest.mark.ascend


@pytest.mark.parametrize("device", [0, 1])
def test_c_api_stream_selection_matches_python_context(device):
    if device >= torch.flagos.device_count():
        pytest.skip("second device unavailable")
    library = ctypes.CDLL(str(Path(torch_fl.__file__).parent / "lib/libflagos.so"))
    for name in ["GetDefaultStreamForDevice", "GetCurrentStreamForDevice"]:
        function = getattr(library, name)
        function.argtypes = [ctypes.c_int]
        function.restype = ctypes.c_void_p
    library.SetCurrentStreamForDevice.argtypes = [ctypes.c_int, ctypes.c_void_p]
    library.SetCurrentStreamForDevice.restype = ctypes.c_int

    previous_device = torch.flagos.current_device()
    previous_stream = library.GetCurrentStreamForDevice(device)
    try:
        auxiliary = torch.flagos.Stream(device=device)
        with torch.flagos.stream(auxiliary):
            current = library.GetCurrentStreamForDevice(device)
            assert current != library.GetDefaultStreamForDevice(device)
            torch.flagos.set_device(previous_device)
            assert library.GetCurrentStreamForDevice(device) == current
        assert library.GetCurrentStreamForDevice(device) == previous_stream
        assert library.SetCurrentStreamForDevice(device, current) == 0
        assert torch.flagos.current_stream(device).cuda_stream == current
    finally:
        library.SetCurrentStreamForDevice(device, previous_stream)
        torch.flagos.set_device(previous_device)


@pytest.mark.native_source_host
def test_acl_event_orders_eager_ops_across_streams():
    device = torch.flagos.current_device()
    target = f"flagos:{device}"
    default = torch.flagos.current_stream(device)
    consumer = torch.flagos.Stream(device=device)
    reference = torch.arange(32, dtype=torch.bfloat16)
    value = reference.to(target)
    factor = torch.full_like(reference, -1).to(target)

    # Exercise slot reuse in the host queue before recording the event.
    for _ in range(513):
        value = torch.mul(value, factor)
    complete = default.record_event()

    with torch.flagos.stream(consumer):
        consumer.wait_event(complete)
        result = torch.mul(value, factor)
    consumer.synchronize()
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)
