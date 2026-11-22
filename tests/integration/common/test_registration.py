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

"""The device torch_fl claims: name, availability, count, and selection.

Every assertion is a protocol invariant -- a name that must be ``flagos``, a
count that must be at least one, an index that must be addressable -- so the
same file is correct on every chip. No vendor magnitude and no platform name
appears.
"""

import pytest
import torch

import torch_fl

DEVICE = "flagos:0"


def _registered_backend_name() -> str | None:
    """The PrivateUse1 name torch is currently using (``None`` if unqueryable).

    The accessor is version-sensitive, so its absence downgrades the check to
    the device-string one below rather than failing an otherwise-correct wheel.
    """
    getter = getattr(torch._C, "_get_privateuse1_backend_name", None)
    if getter is None:
        return None
    return str(getter())


@pytest.mark.anyplatform
def test_backend_is_registered_as_flagos():
    assert torch_fl.flagos.is_available()
    assert torch.device(DEVICE).type == "flagos"
    name = _registered_backend_name()
    if name is not None:
        assert name == "flagos"


@pytest.mark.anyplatform
def test_device_count_at_least_one():
    assert torch_fl.flagos.device_count() >= 1


@pytest.mark.anyplatform
def test_current_device_is_addressable():
    assert 0 <= torch_fl.flagos.current_device() < torch_fl.flagos.device_count()


@pytest.mark.anyplatform
def test_set_device_round_trip():
    original = torch_fl.flagos.current_device()
    try:
        torch_fl.flagos.set_device(0)
        assert torch_fl.flagos.current_device() == 0
    finally:
        torch_fl.flagos.set_device(original)
    assert torch_fl.flagos.current_device() == original


@pytest.mark.anyplatform
def test_device_context_manager_restores_selection():
    original = torch_fl.flagos.current_device()
    with torch_fl.flagos.device(0):
        assert torch_fl.flagos.current_device() == 0
    assert torch_fl.flagos.current_device() == original
