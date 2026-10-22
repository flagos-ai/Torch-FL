# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
"""DCU stream Python contracts, using isolated mocks rather than HIP hardware."""

import ast
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(monkeypatch):
    state = threading.local()

    def current_device():
        return getattr(state, "device", 0)

    def set_device(index):
        state.device = index

    def streams():
        if not hasattr(state, "streams"):
            state.streams = {}
        return state.streams

    def stream_info(index, use_default=False):
        return (0 if use_default else streams().get(index, 7), index, 20)

    def exchange(sid, index):
        streams()[index] = sid

    class MockStream:
        def __new__(cls, device=None, priority=0, **kwargs):
            obj = object.__new__(cls)
            obj.device_index = kwargs.get(
                "device_index",
                int(str(device).split(":")[-1]) if device else current_device(),
            )
            obj.device_type = kwargs.get("device_type", 20)
            obj.stream_id = kwargs.get("stream_id", 100 + obj.device_index)
            obj.device = types.SimpleNamespace(type="flagos", index=obj.device_index)
            return obj

    torch = types.ModuleType("torch")
    torch.Stream = MockStream
    torch.Event = type("MockEvent", (), {})
    torch.flagos = types.SimpleNamespace(
        current_device=current_device, set_device=set_device
    )
    torch.device = lambda value: types.SimpleNamespace(
        type=str(value).split(":")[0],
        index=int(str(value).split(":")[1]) if ":" in str(value) else None,
    )
    bridge = types.SimpleNamespace(
        _dcu_stream_info=stream_info,
        _dcu_set_stream=exchange,
        _dcu_raw_stream=lambda sid, index: 1000 + sid,
        _dcu_external_stream=lambda raw, index: (raw, index),
    )
    package = types.ModuleType("torch_fl")
    package._C = bridge
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_fl", package)
    spec = importlib.util.spec_from_file_location(
        "_test_dcu_stream_contract", ROOT / "torch_fl/accelerator/dcu/dcu_stream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return types.SimpleNamespace(
        module=module,
        torch=torch,
        bridge=bridge,
        state=state,
        exchange=exchange,
        current_device=current_device,
        set_device=set_device,
        stream_info=stream_info,
    )


@pytest.mark.parametrize("existing", [False, True])
def test_context_tls_is_initialized_before_publication(runtime, existing):
    m = runtime.module
    stream = m.current_stream() if existing else m.DcuStream()
    assert isinstance(stream._contexts, threading.local)


def test_same_stream_first_enter_on_two_threads(runtime):
    stream = runtime.module.DcuStream(device=1)
    barrier = threading.Barrier(2)
    errors = []
    stacks = []

    def worker():
        try:
            barrier.wait(timeout=5)
            with stream:
                stacks.append(id(stream._contexts.stack))
                assert runtime.current_device() == 1
                barrier.wait(timeout=5)
            assert not stream._contexts.stack
            assert runtime.current_device() == 0
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert len(set(stacks)) == 2


def test_nested_context_restores_target_streams_and_ambient_device(runtime):
    m = runtime.module
    a, b = m.DcuStream(device=1), m.DcuStream(device=2)
    with a:
        assert runtime.current_device() == 1
        with b:
            assert runtime.current_device() == 2
        assert runtime.current_device() == 1
        assert runtime.stream_info(2)[0] == 7
        with a:
            assert len(a._contexts.stack) == 2
        assert runtime.stream_info(1)[0] == a.stream_id
    assert runtime.current_device() == 0
    assert runtime.stream_info(1)[0] == 7
    assert not a._contexts.stack


@pytest.mark.parametrize("operation", ["set_stream", "stream_context", "enter"])
def test_enter_failure_rolls_back_without_leaving_stack(runtime, operation):
    m = runtime.module
    a = m.DcuStream(device=1)
    primary = RuntimeError("injected partially completed exchange")

    def fail_new(sid, index):
        runtime.exchange(sid, index)
        if sid == a.stream_id:
            raise primary

    runtime.bridge._dcu_set_stream = fail_new
    with pytest.raises(RuntimeError) as captured:
        if operation == "set_stream":
            m.set_stream(a)
        elif operation == "stream_context":
            with m.stream_context(a):
                pytest.fail("failed context must not enter its body")
        else:
            a.__enter__()
    assert captured.value is primary
    assert runtime.current_device() == 0
    assert runtime.stream_info(1)[0] == 7
    assert not getattr(a._contexts, "stack", [])


def test_exit_stream_failure_still_restores_device(runtime):
    a = runtime.module.DcuStream(device=1)
    context = runtime.module.stream_context(a)
    context.__enter__()

    def fail_restore(sid, index):
        raise RuntimeError("restore stream failed")

    runtime.bridge._dcu_set_stream = fail_restore
    with pytest.raises(RuntimeError, match="restore stream failed"):
        context.__exit__(None, None, None)
    assert runtime.current_device() == 0


def test_cleanup_does_not_mask_model_error(runtime):
    primary = ValueError("model failed")
    a = runtime.module.DcuStream(device=1)
    with pytest.raises(ValueError) as captured:
        with a:

            def fail_restore(sid, index):
                raise RuntimeError("restore stream failed")

            runtime.bridge._dcu_set_stream = fail_restore
            raise primary
    assert captured.value is primary
    assert runtime.current_device() == 0
    assert not a._contexts.stack
    notes = getattr(primary, "__notes__", None) or getattr(
        primary, "_flagos_restore_errors", None
    )
    assert notes


def test_both_restore_failures_preserve_first_and_attempt_device(runtime):
    a = runtime.module.DcuStream(device=1)
    context = runtime.module.stream_context(a)
    context.__enter__()
    attempted = []

    def fail_stream(sid, index):
        raise RuntimeError("first: restore stream failed")

    def fail_device(index):
        attempted.append(index)
        raise ValueError("second: restore device failed")

    runtime.bridge._dcu_set_stream = fail_stream
    runtime.torch.flagos.set_device = fail_device
    with pytest.raises(RuntimeError, match="first"):
        context.__exit__(None, None, None)
    assert attempted == [0]


def test_external_stream_preserves_keyword_type_and_borrowing_contract(runtime):
    m = runtime.module
    borrowed = m.DcuExternalStream(stream_ptr=1234, device=1)
    assert isinstance(borrowed, m.DcuExternalStream)
    assert isinstance(borrowed, m.DcuStream)
    assert borrowed.stream_id == 1234
    assert borrowed.device_index == 1
    with borrowed:
        assert runtime.current_device() == 1
    assert runtime.current_device() == 0
    assert isinstance(m.external_stream(1234, 1), m.DcuExternalStream)


@pytest.mark.parametrize("provider", ["native", "facade"])
def test_cuda_interface_uses_one_actual_provider(runtime, monkeypatch, provider):
    torch = runtime.torch
    torch.flagos._C = runtime.bridge
    if provider == "native":
        stream_class = type("NativeStream", (), {})
        event_class = type("NativeEvent", (), {})
    else:
        stream_class, event_class = runtime.module.DcuStream, runtime.module.DcuEvent
    cuda = types.SimpleNamespace(
        Stream=stream_class,
        Event=event_class,
        current_stream=lambda: stream_class(),
        _exchange_device=lambda index: 0,
        _maybe_exchange_device=lambda index: 0,
    )
    for name in (
        "current_device",
        "set_device",
        "device_count",
        "stream",
        "set_stream",
        "_set_stream_by_id",
        "synchronize",
        "memory_allocated",
    ):
        setattr(cuda, name, lambda *args, **kwargs: None)
    torch.cuda = cuda
    torch._C = types.SimpleNamespace(_cuda_getCurrentRawStream=lambda index: 88)
    di = types.ModuleType("torch._dynamo.device_interface")
    di.CudaInterface = type("CudaInterface", (), {})
    dynamo = types.ModuleType("torch._dynamo")
    dynamo.device_interface = di
    torch._dynamo = dynamo
    monkeypatch.setitem(sys.modules, "torch._dynamo", dynamo)
    monkeypatch.setitem(sys.modules, "torch._dynamo.device_interface", di)
    stream_package = "torch_fl.accelerator.dcu.dcu_stream"
    monkeypatch.setitem(sys.modules, stream_package, runtime.module)
    tree = ast.parse(
        (ROOT / "torch_fl/compile/device_interface.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_repair_stock_cuda_interface"
    )
    namespace = {
        "torch": torch,
        "platform_profile": lambda: types.SimpleNamespace(vendor="dcu"),
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "device_interface.py", "exec"
        ),
        namespace,
    )
    for _ in range(2):
        namespace[function.name]()
        assert di.CudaInterface.Stream is cuda.Stream
        assert di.CudaInterface.Event is cuda.Event
        assert isinstance(di.CudaInterface.current_stream(), di.CudaInterface.Stream)
        assert di.CudaInterface.get_raw_stream(0) == 88


@pytest.mark.parametrize("operation", ["set_stream", "stream_context"])
def test_set_device_partial_failure_rolls_back(runtime, operation):
    a = runtime.module.DcuStream(device=1)
    primary = ValueError("injected device selection failure")

    def fail(index):
        runtime.set_device(index)
        if index == 1:
            raise primary

    runtime.torch.flagos.set_device = fail
    with pytest.raises(ValueError) as captured:
        if operation == "set_stream":
            runtime.module.set_stream(a)
        else:
            with runtime.module.stream_context(a):
                pytest.fail("body must not run")
    assert captured.value is primary
    assert runtime.current_device() == 0
    assert runtime.stream_info(1)[0] == 7


def test_many_threads_reenter_shared_stream(runtime):
    stream = runtime.module.DcuStream(device=1)
    barrier = threading.Barrier(8)
    errors = []

    def work():
        try:
            barrier.wait(timeout=10)
            for _ in range(100):
                with stream:
                    with stream:
                        assert len(stream._contexts.stack) == 2
                assert not stream._contexts.stack
                assert runtime.current_device() == 0
        except BaseException as error:
            errors.append(error)

    workers = [threading.Thread(target=work) for _ in range(8)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=15)
    assert not any(thread.is_alive() for thread in workers)
    assert not errors


@pytest.mark.parametrize("api", ["set_stream", "stream_context"])
def test_none_context_is_noop(runtime, api):
    m = runtime.module
    if api == "set_stream":
        m.set_stream(None)
    else:
        with m.stream_context(None):
            assert runtime.current_device() == 0
    assert runtime.current_device() == 0
    assert runtime.stream_info(0)[0] == 7


@pytest.mark.parametrize("device", ["cpu", "meta", "musa:0"])
def test_invalid_device_tag_rejected(runtime, device):
    with pytest.raises(ValueError, match="Expected a DCU device"):
        runtime.module.device_index(device)


@pytest.mark.parametrize("tag", [1, 20])
def test_set_stream_by_id_accepts_actual_registry_tags(runtime, tag):
    runtime.module.set_stream_by_id(44, 1, tag)
    assert runtime.current_device() == 1
    assert runtime.stream_info(1)[0] == 44


def test_set_stream_by_id_rejects_foreign_device_tag(runtime):
    with pytest.raises(ValueError, match="stream tag"):
        runtime.module.set_stream_by_id(44, 1, 0)
    assert runtime.current_device() == 0


def test_restore_diagnostics_accumulate_on_python310(runtime):
    # A duck-typed diagnostic carrier exercises Python 3.10's branch without
    # relying on the interpreter that runs this mock suite.
    carrier = types.SimpleNamespace()
    runtime.module._annotate_restore_errors(carrier, [ValueError("first")])
    runtime.module._annotate_restore_errors(carrier, [RuntimeError("second")])
    assert len(carrier._flagos_restore_errors) == 2


@pytest.mark.parametrize("api", ["context", "enter"])
def test_cross_device_context_restores_both_stream_registries(runtime, api):
    m = runtime.module
    runtime.exchange(21, 0)
    runtime.exchange(22, 1)
    destination = m.DcuStream(device=1)
    source_replacement = m.DcuStream(device=0)
    context = destination if api == "enter" else m.stream_context(destination)
    with context:
        m.set_stream(source_replacement)
        assert runtime.stream_info(0)[0] == source_replacement.stream_id
    assert runtime.current_device() == 0
    assert runtime.stream_info(0)[0] == 21
    assert runtime.stream_info(1)[0] == 22


def test_all_restoration_attempts_run_after_destination_failure(runtime):
    m = runtime.module
    context = m.stream_context(m.DcuStream(device=1))
    context.__enter__()
    calls = []

    def restore(sid, index):
        calls.append(index)
        if index == 1:
            raise RuntimeError("destination restore failed")
        runtime.exchange(sid, index)

    runtime.bridge._dcu_set_stream = restore
    with pytest.raises(RuntimeError, match="destination restore failed"):
        context.__exit__(None, None, None)
    assert calls == [1, 0]
    assert runtime.current_device() == 0


def test_enter_stack_append_failure_restores_without_masking(runtime):
    class BrokenStack(list):
        def append(self, value):
            raise MemoryError("append failed")

    stream = runtime.module.DcuStream(device=1)
    stream._contexts.stack = BrokenStack()
    with pytest.raises(MemoryError, match="append failed"):
        stream.__enter__()
    assert runtime.current_device() == 0
    assert runtime.stream_info(1)[0] == 7
    assert not stream._contexts.stack


@pytest.mark.parametrize(
    "pointer, error",
    [
        (-1, ValueError),
        (1 << 128, ValueError),
        (True, TypeError),
        (1.5, TypeError),
        ("123", TypeError),
        (None, TypeError),
    ],
)
def test_external_pointer_rejected_before_vendor_call(runtime, pointer, error):
    def forbidden(*args):
        pytest.fail("invalid pointer must not reach the vendor")

    runtime.bridge._dcu_external_stream = forbidden
    with pytest.raises(error):
        runtime.module.DcuExternalStream(stream_ptr=pointer)


def test_external_pointer_accepts_zero_and_index_protocol(runtime):
    class Pointer:
        def __index__(self):
            return 1234

    assert runtime.module.DcuExternalStream(stream_ptr=0).stream_id == 0
    assert runtime.module.DcuExternalStream(stream_ptr=Pointer()).stream_id == 1234


def test_no_callable_add_note_preserves_cleanup_evidence(runtime):
    class LegacyError(Exception):
        add_note = None

    error = LegacyError("primary")
    runtime.module._annotate_restore_errors(error, [RuntimeError("cleanup")])
    assert len(error._flagos_restore_errors) == 1
