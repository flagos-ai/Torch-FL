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

import pytest
import torch
import torch_fl


def _cuda_driver_error(msg: str) -> bool:
    return (
        "NVIDIA driver on your system is too old" in msg
        or "Found no NVIDIA driver on your system" in msg
    )


def _flagos_ready() -> bool:
    return torch_fl.flagos.device_count() > 0


@pytest.mark.skipif(not _flagos_ready(), reason="flagos device not available")
class TestCloneDispatchCase:
    def test_clone_preserve_succeeds(self):
        x = torch.arange(16, dtype=torch.float16).reshape(4, 4).to("flagos:0")
        y = x.clone()
        assert y.device.type == "flagos"
        assert torch.equal(y.cpu(), x.cpu())

    def test_clone_contiguous_format_works(self):
        x = torch.arange(16, dtype=torch.float16).reshape(4, 4).to("flagos:0")
        y = x.clone(memory_format=torch.contiguous_format)
        assert y.device.type == "flagos"
        assert torch.equal(y.cpu(), x.cpu())

    @pytest.mark.parametrize("side_stream", [False, True])
    def test_hygon_large_clone_orders_next_kernel(self, side_stream):
        if not torch_fl.backend_config_path().endswith("backends_hygon.conf"):
            pytest.skip("Hygon contiguous clone stream regression")

        # A host read of the clone itself waits for device work and can hide
        # the ordering bug. Queue a consumer first and only read its result.
        shape = (200, 40999, 3)
        host = torch.full(shape, 7.0, dtype=torch.float32)
        device = torch.device("flagos:0")
        x = host.to(device)
        current = torch.flagos.current_stream(device)
        stream = torch.flagos.Stream(device=device) if side_stream else current
        stream.wait_stream(current)
        with torch.flagos.stream(stream):
            for _ in range(8):
                result = x.clone() + 4
                assert torch.equal(result.cpu(), host + 4)

    def test_copy_kernel_works(self):
        src = torch.arange(16, dtype=torch.float16).reshape(4, 4).to("flagos:0")
        dst = torch.empty_like(src)
        dst.copy_(src)
        assert dst.device.type == "flagos"
        assert torch.equal(dst.cpu(), src.cpu())
