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

"""Unit coverage for the MUSA arm of ``torch.flagos``'s stream dispatch.

MUSA names itself neither "gcu" nor a CUDA runtime, and the dispatch in
``torch_fl/flagos/__init__.py`` used to end in a bare ``else`` that meant Ascend.
So ``torch.flagos.Stream()`` on MUSA imported
``torch_fl.accelerator.ascend.acl_stream`` -- whose first statement is
``ctypes.CDLL("libascendcl.so")`` -- and died on the missing library: FSDP2's
``FSDPCommContext.lazy_init`` asks for four streams and got that instead
(current issue #451). The Ascend module must not be reachable from MUSA at all,
which ``sys.modules`` is what these tests check; a vendor with no stream class of
its own must fail on the missing implementation instead of on that library.

The vendor branches are reached by calling ``Stream.__init__`` on an
``object.__new__(Stream)`` rather than ``Stream(...)``: ``Stream.__new__`` proxies
``torch.cuda.Stream`` and only falls back to ``object.__new__`` where there is no
CUDA runtime, so constructing the class directly would test the host rather than
the dispatch. Nothing here needs a MUSA device or the extension's real bindings.
"""

import sys
import types

import pytest
import torch

from torch_fl import flagos

_ASCEND_STREAM_MODULE = "torch_fl.accelerator.ascend.acl_stream"
_HANDLE = 0x7F0000001234


@pytest.fixture
def vendor(monkeypatch):
    """Select the platform and stub the C bindings the dispatch reads."""

    def select(name):
        monkeypatch.setattr(flagos, "_platform", lambda: name)
        monkeypatch.setattr(flagos, "_has_cuda_runtime", lambda: False)
        monkeypatch.delitem(sys.modules, _ASCEND_STREAM_MODULE, raising=False)

    monkeypatch.setattr(
        flagos,
        "_C",
        types.SimpleNamespace(
            _get_device=lambda: 0,
            _set_device=lambda index: None,
            _get_musa_current_raw_stream=lambda index: _HANDLE,
        ),
    )
    return select


def _ascend_was_imported() -> bool:
    return _ASCEND_STREAM_MODULE in sys.modules


def _build_stream(device=None, priority=0):
    """Run ``Stream.__init__`` without ``Stream.__new__``'s CUDA inheritance."""
    stream = object.__new__(flagos.Stream)
    flagos.Stream.__init__(stream, device, priority)
    return stream


def test_stream_builds_the_shared_musa_stream(vendor):
    """The stream MUSA hands out is the handle its ops run on."""
    vendor("musa")

    stream = _build_stream()

    assert stream.musa_stream == _HANDLE
    assert stream.cuda_stream == _HANDLE
    assert stream.stream_id == _HANDLE
    assert stream.device == torch.device("flagos", 0)
    assert "native_stream=0x7f0000001234" in repr(stream)
    assert not _ascend_was_imported()


def test_fsdp_streams_are_one_queue(vendor):
    """FSDP asks for four streams; all four must carry the same handle.

    That is the point of the MUSA arm rather than a side effect: mudnn is bound
    to ``GetDefaultMusaStream`` for the life of the process and Triton launches on
    the same handle, so a second stream would only make the copies FSDP wanted to
    overlap unordered against the ops they feed.
    """
    vendor("musa")

    streams = [_build_stream(priority=-1) for _ in range(3)]
    streams.append(_build_stream())

    assert {stream.musa_stream for stream in streams} == {_HANDLE}
    assert not _ascend_was_imported()


def test_musa_stream_answers_the_fsdp_startup_calls(vendor, monkeypatch):
    """``lazy_init`` and the collectives only need these to be safe."""
    vendor("musa")
    synced = []
    monkeypatch.setattr(
        flagos, "synchronize", lambda device=None: synced.append(device)
    )
    stream = _build_stream(priority=-1)

    stream.wait_stream(object())
    stream.wait_event(object())
    assert stream.query() is True

    # The waits are no-ops on a single queue; synchronize must reach the device.
    stream.synchronize()
    assert synced == [0]

    event = stream.record_event()
    assert type(event).__name__ == "_HostTimedEvent"
    assert event.query() is True


def test_event_is_host_timed_on_musa(vendor):
    """MUSA has no event class of its own, so the host clock is the event."""
    vendor("musa")

    event = flagos.Event(enable_timing=True)

    assert isinstance(event, flagos._HostTimedEvent)
    assert event.enable_timing is True
    assert not _ascend_was_imported()


def test_current_stream_is_a_musa_handle(vendor):
    """``_real_current_stream`` answers MUSA from C++ before any vendor arm."""
    vendor("musa")

    current = flagos.current_stream()

    assert isinstance(current, flagos._DefaultStreamHandle)
    assert current.musa_stream == _HANDLE
    assert not _ascend_was_imported()


@pytest.mark.skipif(
    hasattr(torch._C, "_cuda_setStream"),
    reason="stream() only takes the vendor branch without a CUDA stream registry",
)
def test_stream_context_manager_is_a_no_op_on_musa(vendor):
    """Every MUSA stream is the shared one, so ``with`` only has to run."""
    vendor("musa")
    stream = _build_stream()

    ran = False
    with flagos.stream(stream):
        ran = True

    assert ran
    assert not _ascend_was_imported()


def test_unlisted_vendor_refuses_instead_of_becoming_ascend(vendor):
    """The bare ``else`` is gone: no vendor is handed another vendor's runtime."""
    vendor("ppu")

    with pytest.raises(RuntimeError) as excinfo:
        _build_stream()

    message = str(excinfo.value)
    assert "ppu" in message
    assert "ascend" in message
    assert not _ascend_was_imported()


def test_unlisted_vendor_event_degrades_to_the_host_clock(vendor):
    """Events fail open where streams fail closed, deliberately.

    An event that cannot be the vendor's is still a usable event, because the host
    clock is one; a stream that cannot be the vendor's has no handle to stand in
    for it, and inventing one would point somewhere the device does not.
    """
    vendor("ppu")

    event = flagos.Event()

    assert isinstance(event, flagos._HostTimedEvent)
    assert not _ascend_was_imported()
