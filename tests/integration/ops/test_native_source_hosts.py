# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts of the precompiled source-host wheel."""

import sys
import importlib
from types import SimpleNamespace

import pytest
import torch
import torch_fl

pytestmark = [pytest.mark.nvidia, pytest.mark.native_source_host, pytest.mark.flaggems]


@pytest.mark.parametrize("shape", [(), (0,), (137,), (2, 11, 37)])
def test_unseen_shapes_keep_host_artifacts(shape):
    before = torch_fl.native_host.audit()["host_artifacts"]
    x = torch.randn(shape).to("flagos")
    with torch.no_grad():
        actual = torch.cos(x)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), x.cpu().cos(), rtol=1e-5, atol=1e-6)
    assert torch_fl.native_host.audit()["host_artifacts"] == before


def test_broadcast_strides_and_retained_outputs():
    a = torch.arange(231, dtype=torch.float32).reshape(11, 21).t().to("flagos")
    b = torch.arange(11, dtype=torch.float32).to("flagos")
    with torch.no_grad():
        first = a + b
        second = a + b * 2
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(first.cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(second.cpu(), a.cpu() + b.cpu() * 2, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()


def test_alias_and_slice_preserve_storage():
    x = torch.arange(33, dtype=torch.float32).to("flagos")
    alias = torch.ops.aten.alias(x)
    view = x[1:25:3]
    assert alias._cdata != x._cdata
    assert alias.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    torch.testing.assert_close(view.cpu(), x.cpu()[1:25:3], rtol=0, atol=0)


def test_mutation_uses_current_stream():
    x = torch.zeros(257, device="flagos")
    source = torch.arange(257, dtype=torch.float32).to("flagos")
    torch_fl.flagos.synchronize()
    stream = torch_fl.flagos.Stream()
    with torch_fl.flagos.stream(stream), torch.no_grad():
        result = x.copy_(source)
        output = torch.cos(x)
    stream.synchronize()
    assert result is x
    torch.testing.assert_close(x.cpu(), source.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), source.cpu().cos(), rtol=1e-5, atol=1e-6)


def test_missing_route_never_reaches_cpu():
    library = torch.library.Library("native_source_host_test", "DEF")
    library.define("missing(Tensor value) -> Tensor")
    library.impl("missing", lambda value: value.clone(), "CPU")
    x = torch.empty(1, device="flagos")
    with pytest.raises(RuntimeError, match="do not permit CPU fallback"):
        torch.ops.native_source_host_test.missing(x)


def test_warm_kernel_does_not_execute_flaggems_python():
    x = torch.randn(263).to("flagos")
    with torch.no_grad():
        torch.cos(x)
        torch.cos(x)
        frames = []

        def record(frame, event, arg):
            if event == "call" and any(
                part in frame.f_code.co_filename for part in ("/flag_gems/", "/triton/")
            ):
                frames.append(frame.f_code.co_filename + ":" + frame.f_code.co_name)

        sys.setprofile(record)
        try:
            torch.cos(x)
        finally:
            sys.setprofile(None)
    assert not frames, frames
    assert torch_fl.native_host.audit()["launched_kernels"]


def test_direct_jit_warmup_does_not_launch_and_keeps_return_type():
    module = importlib.import_module("flag_gems.ops.scalar_tensor")
    entry = torch_fl.native_host._hosts.resolve(module.scalar_tensor_kernel)
    output = torch.zeros((), device="flagos")
    first = entry.warmup(output, 1.25, grid=(1,))
    second = entry.run(output, 1.25, grid=(1,), warmup=True)
    torch_fl.flagos.synchronize()
    assert first is second
    assert hasattr(first, "metadata")
    assert output.cpu().item() == 0
    launched = entry[(1,)](output, 1.25)
    assert launched is first
    assert entry[(1,)](output, 1.25) is first
    torch_fl.flagos.synchronize()
    assert output.cpu().item() == 1.25


def test_nested_abi_validation_without_a_device_launch():
    from torch_fl.native_host._generated._launcher import Launcher

    metadata = SimpleNamespace(
        num_warps=1,
        shared=0,
        global_scratch_size=0,
        global_scratch_align=1,
        profile_scratch_size=0,
        profile_scratch_align=1,
    )
    launcher = Launcher(
        SimpleNamespace(signature={"values": (("i32", "i32"),)}), metadata
    )
    # A zero grid exercises ABI packing without invoking any device kernel.
    launcher(0, 1, 1, 0, 0, (), None, None, None, ([3, 4],))
    with pytest.raises(IndexError):
        launcher(0, 1, 1, 0, 0, (), None, None, None, ((),))
    with pytest.raises(ValueError, match="tuple or list"):
        launcher(0, 1, 1, 0, 0, (), None, None, None, (3,))
