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

"""Unit coverage for the MetaX device-properties current-device restore.

``_query_metax_device_properties`` has to move the maca runtime to the device it
is probing, because mcMemGetInfo reports the *current* device's memory. Nothing
but that function moves the runtime back, and the runtime's current device is
process state torch's device counter does not track, so a probe that leaves it
moved sends every later operation meant for device 0 to the last device probed.

No MetaX hardware is needed to pin that: the function takes its runtime handle as
an argument, so a fake one records the mcSetDevice calls and reports the state
they leave behind. What this file cannot check is the observable damage on a real
device -- only that the sequence of calls is the one that avoids it.
"""

import ctypes
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "torch_fl" / "accelerator" / "metax" / "_metax_compat.py"

_spec = importlib.util.spec_from_file_location("metax_compat_under_test", MODULE)
metax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metax)

# Attribute ids the function reads, and the values this fake reports for them.
_ATTRS = {16: 64, 10: 32, 39: 1024, 8: 49152, 81: 131072, 21: 8, 22: 0}


class _Fn:
    """A ctypes function stand-in: assignable argtypes/restype, callable."""

    def __init__(self, impl):
        self._impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._impl(*args)


def _read(ptr, ctype):
    return ctypes.cast(ptr, ctypes.POINTER(ctype)).contents.value


def _write(ptr, ctype, value):
    ctypes.cast(ptr, ctypes.POINTER(ctype)).contents.value = value


class _FakeRuntime:
    """Just enough mcruntime for ``_query_metax_device_properties``."""

    def __init__(
        self,
        current_device=0,
        reports_current_device=True,
        get_device_ret=0,
        meminfo_raises=False,
        total_mib=8192,
        probe_count=0,
    ):
        self.current_device = current_device
        self.set_device_calls = []
        self.meminfo_calls = 0
        self._meminfo_raises = meminfo_raises
        self._total_mib = total_mib
        self.mcDeviceGetName = _Fn(self._device_get_name)
        self.mcDeviceGetAttribute = _Fn(self._device_get_attribute)
        self.mcMemGetInfo = _Fn(self._mem_get_info)
        self.mcSetDevice = _Fn(self._set_device)
        if reports_current_device:
            self.mcGetDevice = _Fn(self._get_device)
            self._get_device_ret = get_device_ret

    # mcDeviceGetName(char *name, int len, int device)
    def _device_get_name(self, name_buf, length, device_index):
        name_buf.value = b"MetaX Fake"
        return 0

    # mcDeviceGetAttribute(int *value, int attr, int device)
    def _device_get_attribute(self, value_ptr, attr, device_index):
        _write(value_ptr, ctypes.c_int, _ATTRS.get(attr, 0))
        return 0

    # mcMemGetInfo(size_t *free, size_t *total) -- reports the current device
    def _mem_get_info(self, free_ptr, total_ptr):
        if self._meminfo_raises:
            raise RuntimeError("mcMemGetInfo failed")
        self.meminfo_calls += 1
        _write(free_ptr, ctypes.c_size_t, self._total_mib * 1024 * 1024 // 2)
        _write(total_ptr, ctypes.c_size_t, self._total_mib * 1024 * 1024)
        return 0

    def _set_device(self, device_index):
        self.set_device_calls.append(device_index)
        self.current_device = device_index
        return 0

    # mcGetDevice(int *device)
    def _get_device(self, device_ptr):
        _write(device_ptr, ctypes.c_int, self.current_device)
        return self._get_device_ret


def test_probe_restores_the_device_it_found():
    # Device 3 is current, the probe asks about device 1: the runtime must be
    # left on 3, not on the device that was probed.
    runtime = _FakeRuntime(current_device=3)
    props = metax._query_metax_device_properties(runtime, 1)

    assert runtime.set_device_calls == [1, 3]
    assert runtime.current_device == 3
    assert props.total_memory == 8192
    assert props.name == "MetaX Fake"
    assert runtime.meminfo_calls == 1


def test_probe_restores_the_same_device_when_it_probes_it():
    runtime = _FakeRuntime(current_device=2)
    metax._query_metax_device_properties(runtime, 2)
    # Probed and previous coincide, but the restore still has to run: it is what
    # makes the *end state* the probe's contract rather than the start state.
    assert runtime.set_device_calls == [2, 2]
    assert runtime.current_device == 2


def test_probe_restores_when_the_memory_read_raises():
    runtime = _FakeRuntime(current_device=1, meminfo_raises=True)
    with pytest.raises(RuntimeError, match="mcMemGetInfo failed"):
        metax._query_metax_device_properties(runtime, 0)
    # The failure must not leave the runtime on the probed device.
    assert runtime.set_device_calls == [0, 1]
    assert runtime.current_device == 1


def test_probe_leaves_the_device_alone_when_it_cannot_read_it_back():
    # A runtime with no mcGetDevice cannot say where it was, so there is nowhere
    # to restore to: the probe must fall back to its previous behaviour rather
    # than move the runtime to a placeholder index.
    runtime = _FakeRuntime(current_device=5, reports_current_device=False)
    metax._query_metax_device_properties(runtime, 0)
    assert runtime.set_device_calls == [0]
    assert runtime.current_device == 0


def test_probe_leaves_the_device_alone_when_the_read_back_fails():
    runtime = _FakeRuntime(current_device=5, get_device_ret=1)
    metax._query_metax_device_properties(runtime, 2)
    assert runtime.set_device_calls == [2]
    assert runtime.current_device == 2
