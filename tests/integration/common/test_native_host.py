# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared inference behavior on the lane's installed native host runtime."""

import pytest
import torch

import torch_fl

_DEVICE = torch.device("flagos:0")


def _capture_launch(runtime, op, *, nonempty=True, specialized=True):
    result, enqueues = runtime.capture_enqueues(op)
    handle = runtime.current_stream(_DEVICE.index)
    assert all(stream == handle for stream, _ in enqueues), enqueues
    if nonempty:
        assert enqueues, "the operation did not enqueue device work"
        if specialized:
            assert (handle, True) in enqueues, enqueues
    return result


def _check_layout_and_storage():
    runtime = torch_fl.native_host._model_runtime()
    views = {
        "dense": lambda x: x,
        "transpose": lambda x: x.transpose(0, 2),
        "slice": lambda x: x[:, :, ::2],
        "expand": lambda x: x[:, :1, :].expand_as(x),
        "scalar": lambda x: x[0, 0, 0],
        "empty": lambda x: x[:, :0, :],
    }
    side = torch.Stream(device=_DEVICE)
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 1e-2)):
        cpu = torch.linspace(-2, 2, 105).reshape(3, 5, 7).to(dtype)
        changed = cpu + 0.25
        with side, torch.no_grad():
            first_device, second_device = cpu.to(_DEVICE), changed.to(_DEVICE)
            for layout, view in views.items():
                first_value, second_value = view(first_device), view(second_device)
                reference = view(cpu)
                assert first_value.stride() == reference.stride(), layout
                nonempty = first_value.numel() > 0
                # The published launcher can submit the first call while the
                # native host records the specialization for the second one.
                first = _capture_launch(
                    runtime,
                    lambda: torch.cos(first_value),
                    nonempty=nonempty,
                    specialized=False,
                )
                second = _capture_launch(
                    runtime, lambda: torch.cos(second_value), nonempty=nonempty
                )
                side.synchronize()
                for result, expected in (
                    (first, torch.cos(reference)),
                    (second, torch.cos(view(changed))),
                ):
                    assert result.device == _DEVICE
                    torch.testing.assert_close(
                        result.cpu(), expected, rtol=tol, atol=tol
                    )
                if nonempty:
                    assert first.data_ptr() != second.data_ptr(), layout


def _check_dynamic_shapes_and_stream_scope():
    runtime = torch_fl.native_host._model_runtime()
    original = torch.accelerator.current_stream(_DEVICE)
    side = torch.Stream(device=_DEVICE)

    def inference(x, output):
        return output.copy_(torch.cos(x))

    model = torch_fl.native_host.compile(inference)
    for size in (35, 57, 35):
        reference = torch.linspace(-2, 2, size)
        with side, torch.no_grad():
            value = reference.to(_DEVICE)
            output = torch.empty_like(value)
            first = _capture_launch(
                runtime, lambda: model(value, output), specialized=False
            )
            actual = _capture_launch(runtime, lambda: model(value, output))
        assert torch.accelerator.current_stream(_DEVICE) == original
        assert actual.device == _DEVICE
        assert first.data_ptr() == actual.data_ptr() == output.data_ptr()
        side.synchronize()
        torch.testing.assert_close(actual.cpu(), torch.cos(reference))
    with pytest.raises(RuntimeError, match=r"require torch.no_grad\(\)"):
        with side, torch.enable_grad():
            model(value, output)
    assert torch.accelerator.current_stream(_DEVICE) == original


@pytest.mark.anyplatform
def test_layout_and_storage():
    from ._native_host import run_native_host_case

    run_native_host_case(_check_layout_and_storage)


@pytest.mark.anyplatform
def test_dynamic_shapes_and_stream_scope():
    from ._native_host import run_native_host_case

    run_native_host_case(_check_dynamic_shapes_and_stream_scope)
