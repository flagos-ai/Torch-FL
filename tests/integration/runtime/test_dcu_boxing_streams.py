# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
"""Real-stream gates for the optional DCU boxing build, not HSDP certification."""

import concurrent.futures
import contextlib
import ctypes
import gc
import threading

import pytest
import torch
import torch_fl

pytestmark = pytest.mark.dcu


@pytest.fixture(autouse=True)
def boxed_dcu():
    if not hasattr(torch_fl._C, "_dcu_stream_info"):
        pytest.skip("requires DCU FLAGOS_BUILD_BOXING=1")
    torch.flagos.set_device(0)
    yield
    torch.flagos.synchronize()


@contextlib.contextmanager
def pending(stream):
    """Hold one stream without a GPU busy loop or a device-wide barrier."""
    lib = ctypes.CDLL("libgalaxyhip.so")
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    ready, release, finished = threading.Event(), threading.Event(), threading.Event()

    @callback_type
    def callback(_):
        ready.set()
        release.wait(30)
        finished.set()

    launch = lib.hipLaunchHostFunc
    launch.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    assert launch(stream.cuda_stream, callback, None) == 0
    assert ready.wait(10)
    try:
        yield finished
        assert not finished.is_set(), "blocker expired; timing evidence is invalid"
    finally:
        release.set()
        assert finished.wait(35)
        stream.synchronize()


def test_distinct_streams_events_and_nonblocking_query():
    a, b = torch.flagos.Stream(), torch.flagos.Stream()
    assert a.cuda_stream != b.cuda_stream
    assert a.cuda_stream != torch.flagos.default_stream().cuda_stream
    with pending(a):
        event = a.record_event()
        assert not a.query()
        assert not event.query()
        assert b.query(), "an unrelated idle stream must remain independent"
        b.wait_event(event)
        assert not b.query()
    b.synchronize()
    assert event.query()


def test_context_and_raw_handle_restore_on_exception():
    previous = torch.flagos.current_stream()
    stream = torch.flagos.Stream()
    with pytest.raises(RuntimeError, match="test error"):
        with torch.flagos.stream(stream):
            assert torch.flagos.current_stream() == stream
            assert torch._C._cuda_getCurrentRawStream(0) == stream.cuda_stream
            raise RuntimeError("test error")
    assert torch.flagos.current_stream() == previous


def test_current_stream_is_thread_local():
    streams = [torch.flagos.Stream(), torch.flagos.Stream()]
    barrier = threading.Barrier(2)

    def worker(stream):
        with torch.flagos.stream(stream):
            barrier.wait(timeout=10)
            return torch.flagos.current_stream().cuda_stream

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        assert list(pool.map(worker, streams)) == [s.cuda_stream for s in streams]


def test_external_stream_is_borrowed_not_owned():
    stream = torch.flagos.Stream()
    borrowed = torch.flagos.ExternalStream(stream.cuda_stream, device=0)
    assert borrowed.cuda_stream == stream.cuda_stream
    with torch.flagos.stream(borrowed):
        assert torch.flagos.current_stream().cuda_stream == stream.cuda_stream
    del borrowed
    gc.collect()
    assert stream.query()


def test_peer_device_context_restores_both_registries():
    if torch.flagos.device_count() < 2:
        pytest.skip("requires two DCUs")
    previous = [torch.flagos.current_stream(i) for i in (0, 1)]
    with torch.flagos.stream(torch.flagos.Stream(device=1)):
        assert torch.flagos.current_device() == 1
    assert torch.flagos.current_device() == 0
    assert [torch.flagos.current_stream(i) for i in (0, 1)] == previous


def test_record_stream_prevents_pending_consumer_reuse():
    producer, consumer = torch.flagos.Stream(), torch.flagos.Stream()
    with torch.flagos.stream(producer):
        x = torch.empty(65536, device="flagos")
    producer.synchronize()
    with pending(consumer):
        x.record_stream(consumer)
        ptr = x.data_ptr()
        del x
        gc.collect()
        with torch.flagos.stream(producer):
            allocations = [torch.empty(65536, device="flagos") for _ in range(32)]
        assert ptr not in [t.data_ptr() for t in allocations]


def test_clone_on_selected_stream_and_native_alias():
    source = torch.arange(64, dtype=torch.float32).to("flagos")
    a, b = torch.flagos.Stream(), torch.flagos.Stream()
    with torch.flagos.stream(a):
        x = source * 3
        produced = a.record_event()
    with torch.flagos.stream(b):
        b.wait_event(produced)
        result = x.clone()
    b.synchronize()
    torch.testing.assert_close(result.cpu(), torch.arange(64, dtype=torch.float32) * 3)
    # The existing bridge must retain native view/device semantics. This PR
    # deliberately does not replace Storage or add differentiable identities.
    native = torch_fl._C._flagos_to_cuda_view(result)
    for view in (native.view(8, 8), native[1:], native.view(8, 8).t()):
        assert view.device.type == "cuda"


def test_compiled_launch_uses_selected_stream():
    from torch_fl.compile.device_interface import FlagOSDeviceInterface

    fn = torch.compile(lambda x: x * 2 + 1, backend="flagos", fullgraph=True)
    host = torch.arange(256, dtype=torch.float32)
    x = host.to("flagos")
    outputs = []
    for _ in range(2):
        stream = torch.flagos.Stream()
        with torch.flagos.stream(stream):
            assert FlagOSDeviceInterface.get_raw_stream(0) == stream.cuda_stream
            outputs.append((fn(x), stream))
    for output, stream in outputs:
        stream.synchronize()
        torch.testing.assert_close(output.cpu(), host * 2 + 1)


def test_exact_self_copy_does_not_wait_for_pending_stream():
    x = torch.empty(32, device="flagos")
    stream = torch.flagos.Stream()
    before = x._version
    with pending(stream):
        with torch.flagos.stream(stream):
            x.copy_(x)
        assert x._version == before + 1
        assert not stream.query()


def test_raw_storage_resize_preserves_device_and_selected_stream_data():
    if torch.flagos.device_count() < 2:
        pytest.skip("requires two DCUs")
    host = torch.arange(16, dtype=torch.float32)
    source = host.to("flagos:1")
    stream = torch.flagos.Stream(device=1)
    with torch.flagos.stream(stream):
        x = source * 3
    # The allocation belongs to device 1 even though device 0 is ambient.
    torch.flagos.set_device(0)
    previous = torch.flagos.current_stream(1)
    torch_fl._C._dcu_set_stream(stream.stream_id, 1)
    storage = x.untyped_storage()
    try:
        storage.resize_(storage.nbytes() * 2)
    finally:
        torch_fl._C._dcu_set_stream(previous.stream_id, 1)
    assert storage.device == torch.device("flagos:1")
    assert torch.flagos.current_device() == 0
    stream.synchronize()
    torch.testing.assert_close(x.cpu(), host * 3)


@pytest.mark.parametrize("device", [256, 65536])
def test_device_index_does_not_wrap_to_another_card(device):
    with pytest.raises(RuntimeError, match="Invalid DCU stream device index"):
        torch.flagos.current_stream(device)
    with pytest.raises(RuntimeError, match="Invalid DCU stream device index"):
        torch.flagos.ExternalStream(0, device=device)


def test_compiled_mutation_cannot_escape_to_default_stream():
    fn = torch.compile(lambda x: x.add_(1), backend="flagos", fullgraph=True)
    stream = torch.flagos.Stream()
    # Compile and warm the exact specialization before holding the queue.
    with torch.flagos.stream(stream):
        warm = torch.zeros(256, device="flagos")
        fn(warm)
    stream.synchronize()
    x = torch.zeros(256, device="flagos")
    torch.flagos.synchronize()
    with pending(stream):
        with torch.flagos.stream(stream):
            fn(x)
        # A raw-getter equality check alone cannot prove launch placement. An
        # escaped launch on stream 0 would finish here and change these bytes;
        # a correctly queued launch must remain behind the held selected stream.
        with torch.flagos.stream(torch.flagos.default_stream()):
            torch.flagos.default_stream().synchronize()
            torch.testing.assert_close(x.cpu(), torch.zeros(256))
    torch.testing.assert_close(x.cpu(), torch.ones(256))
