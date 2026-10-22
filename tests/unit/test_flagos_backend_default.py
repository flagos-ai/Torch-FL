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

"""The flagos backend must be reachable through ``get_default_backend_for_device``.

``register_flagos_backend`` used to key both the registration's ``devices=``
argument and ``Backend.default_device_backend_map`` with the literal
``"privateuseone"``.  At runtime the device type is ``"flagos"`` -- torch_fl
renames PrivateUse1, and ``torch.device("privateuseone:0")`` normalises to type
``"flagos"`` before any lookup -- so that entry was unreachable,
``get_default_backend_for_device`` raised, and
``init_process_group(device_id=torch.device("flagos:0"))`` could not
auto-select a backend because PyTorch never found one for the accelerator it
had just detected (#506).

These run in CI on vendor runners where ``import torch_fl`` works; they need no
accelerator, only the device *name* the map is keyed by.
"""

import torch_fl  # noqa: F401 - must precede torch: it preloads the device libraries

import torch
import torch.distributed as dist


def _device_type() -> str:
    return torch._C._get_privateuse1_backend_name()


def test_backend_is_registered():
    assert "flagos" in dist.Backend.backend_list


def test_default_backend_is_registered_under_the_runtime_device_type():
    device = torch.device(f"{_device_type()}:0")
    assert dist.get_default_backend_for_device(device) == "flagos"


def test_the_legacy_spelling_resolves_to_the_same_device_type():
    """``privateuseone:0`` normalises to the renamed type, so the same key serves it.

    This is the case that made the old ``"privateuseone"`` key look correct: the
    string appears in the spelling, but never as the type the lookup reads.
    """
    assert torch.device("privateuseone:0").type == _device_type()
    assert (
        dist.get_default_backend_for_device(torch.device("privateuseone:0")) == "flagos"
    )
