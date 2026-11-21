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

"""Hygon stream ownership when the wheel has the native stream guard."""

import pytest
import torch

import torch_fl
from torch_fl import _C
from torch_fl._platform import detect_platform

DEVICE = "flagos:0"


@pytest.mark.anyplatform
def test_hygon_owned_stream_is_distinct_and_tensor_bindable():
    if detect_platform() != "dcu" or not hasattr(_C, "_get_native_current_raw_stream"):
        pytest.skip("requires the Hygon native stream guard")
    current = torch_fl.flagos.current_stream(DEVICE)
    high_priority = torch_fl.flagos.Stream(DEVICE, priority=-1)
    ordinary = torch_fl.flagos.Stream(DEVICE)
    assert isinstance(high_priority, torch.Stream)
    assert len({current.stream_id, high_priority.stream_id, ordinary.stream_id}) == 3
    tensor = torch.arange(32).to(DEVICE)
    tensor.record_stream(high_priority)
    tensor.record_stream(ordinary)
    with torch_fl.flagos.stream(high_priority):
        assert (
            torch_fl.flagos.current_stream(DEVICE).stream_id == high_priority.stream_id
        )
    assert torch_fl.flagos.current_stream(DEVICE).stream_id == current.stream_id
