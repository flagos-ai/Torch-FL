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

"""Enflame device names from TOPS and the optional EFML management library."""

import ctypes


def _efml_board_name(bus_id: str) -> str:
    try:
        efml = ctypes.CDLL("libefml.so.1")
        initialize = efml.efmlInit_v2
        release = efml.efmlShutdown
        get_handle = efml.efmlDeviceGetHandleByPciBusId_v2
        get_name = efml.efmlDeviceGetName
    except (OSError, AttributeError):
        return ""

    get_handle.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    get_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    if initialize() != 0:
        return ""
    try:
        handle = ctypes.c_void_p()
        if get_handle(bus_id.encode(), ctypes.byref(handle)) != 0 or not handle:
            return ""
        name = ctypes.create_string_buffer(256)
        if get_name(handle, name, len(name)) != 0:
            return ""
        return name.value.decode(errors="replace")
    finally:
        release()


def get_device_name(device_id: int) -> str:
    from torch_fl import _C

    name, bus_id = _C._get_enflame_device_identity(device_id)
    if name and name != "GCU":
        return name
    # TOPS reports "GCU" for some boards. Its PCI address identifies the
    # same physical device in EFML even when runtime ordinals are remapped.
    return (bus_id and _efml_board_name(bus_id)) or "Enflame GCU"
