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

"""Device properties: assert the types and invariants, never the magnitudes.

The contract is that ``get_device_properties`` returns a usable, torch-shaped
properties object on every chip. Memory sizes and core counts differ per vendor
by orders of magnitude, so this file asserts only what must hold everywhere --
``name`` is a non-empty string, ``total_memory`` is a positive integer -- and
deliberately checks no value that would pin the test to one accelerator.
"""

import pytest
from device_init import torch, torch_fl


DEVICE = "flagos:0"


@pytest.mark.anyplatform
def test_properties_name_is_a_nonempty_string():
    props = torch_fl.flagos.get_device_properties(0)
    assert isinstance(props.name, str)
    assert props.name


@pytest.mark.anyplatform
def test_properties_total_memory_is_positive():
    props = torch_fl.flagos.get_device_properties(0)
    assert isinstance(props.total_memory, int)
    assert props.total_memory > 0


@pytest.mark.anyplatform
def test_properties_multi_processor_count_is_nonnegative():
    props = torch_fl.flagos.get_device_properties(0)
    assert isinstance(props.multi_processor_count, int)
    assert props.multi_processor_count >= 0


@pytest.mark.anyplatform
def test_properties_accept_device_object_and_index():
    by_index = torch_fl.flagos.get_device_properties(0)
    by_device = torch_fl.flagos.get_device_properties(torch.device(DEVICE))
    assert by_index.name == by_device.name
    assert by_index.total_memory == by_device.total_memory
