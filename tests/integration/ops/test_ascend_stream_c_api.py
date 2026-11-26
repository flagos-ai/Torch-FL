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
