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

"""Contract tests for GuardImpl::synchronizeDevice."""

import pytest

# Import order is part of the runtime contract: torch_fl loads the selected
# accelerator assets before torch initialises its backend registry.
import torch_fl  # noqa: F401
import torch


def require_flagos_device():
    if not torch.flagos.is_available():
        pytest.skip("the flagos backend reports no available device")


def test_accelerator_synchronize_current_device():
    require_flagos_device()

    original = torch.flagos.current_device()
    torch.accelerator.synchronize()

    assert torch.flagos.current_device() == original


def test_accelerator_synchronize_named_device_restores_current_device():
    require_flagos_device()
    if torch.flagos.device_count() < 2:
        pytest.skip("device restoration needs two flagos devices")

    original = torch.flagos.current_device()
    target = (original + 1) % torch.flagos.device_count()

    torch.accelerator.synchronize(torch.device("flagos", target))

    assert torch.flagos.current_device() == original
