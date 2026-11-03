"""Physical memory queries are distinct from cached allocator statistics."""

import pytest
import torch_fl  # noqa: F401
import torch


pytestmark = pytest.mark.ascend


@pytest.mark.parametrize("device", [0, 1])
def test_physical_memory_query_and_device_restoration(device):
    if device >= torch.flagos.device_count():
        pytest.skip("second device unavailable")
    previous = torch.flagos.current_device()
    free, total = torch.flagos.mem_get_info(torch.device(f"flagos:{device}"))
    assert torch.flagos.current_device() == previous
    assert 0 < free <= total
    assert total >= 32 * 1024**3
    assert total > torch.flagos.memory_reserved(device)
    assert torch.flagos.get_device_properties(device).total_memory == total
