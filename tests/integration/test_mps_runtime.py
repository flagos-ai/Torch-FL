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

"""The Apple Silicon (M-series) device-runtime floor, with zero operators.

This is the stage-0 gate from ``.claude/skills/runtime-bringup/SKILL.md``: the
28-function C ABI plus the allocator, exercised without a single aten op claimed
on PrivateUse1. It asserts the runtime works -- allocate, host/device copy,
stream and event synchronization, allocator reuse -- not that any operator runs.
An operator failure on top of a broken runtime is nearly impossible to
attribute, which is why this file exists before the MPSGraph kernel layer.

Every case self-skips off an MPS build: the marker keeps CI manifests explicit,
and the skipif keeps a broad ``pytest tests/integration`` collection from
attempting Metal on a machine that has none.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from platform_support import detect_platform
from torch_fl import flagos
from torch_fl.accelerator._vendor_libtorch import active_torch_lib

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

DEVICE = "flagos:0"


def test_device_is_visible():
    assert flagos.device_count() > 0
    assert flagos.is_available()


def test_empty_allocates_on_the_device():
    x = torch.empty(4, 4, device=DEVICE)
    assert x.device.type == "flagos"
    assert x.numel() == 16


def test_host_to_device_round_trip():
    host = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    device = host.to(DEVICE)
    assert device.device.type == "flagos"
    assert torch.equal(device.cpu(), host)


def test_stream_and_timing_event_synchronize():
    stream = flagos.Stream()
    event = flagos.Event(enable_timing=True)
    with flagos.stream(stream):
        y = torch.empty(1 << 20, device=DEVICE)
        y.fill_(1.0)
    event.record(stream)
    stream.synchronize()
    event.synchronize()
    assert event.query()


def test_event_elapsed_time_is_nonnegative():
    start = flagos.Event(enable_timing=True)
    end = flagos.Event(enable_timing=True)
    start.record()
    torch.empty(1 << 16, device=DEVICE).fill_(0.0)
    end.record()
    end.synchronize()
    # ``A.elapsed_time(B)`` measures from A to B (torch's own convention, which
    # the wrapper forwards as EventElapsedTime(ms, A, B) -> B - A), so the
    # earlier event is the receiver.
    assert start.elapsed_time(end) >= 0.0


def test_allocator_reuses_blocks_across_two_streams():
    streams = [flagos.Stream(), flagos.Stream()]
    flagos.empty_cache()
    for _ in range(4):
        for stream in streams:
            with flagos.stream(stream):
                block = torch.empty(1 << 16, device=DEVICE)
                block.fill_(2.0)
            stream.synchronize()
            del block
    flagos.synchronize()
    # Reuse means the reserved pool did not grow with every allocation; the
    # exact number is a measurement, so this asserts only that the pool is
    # bounded, which is what block reuse buys.
    assert flagos.memory_reserved() < (1 << 24)


def test_empty_cache_keeps_allocated_accounting():
    live = torch.empty(1 << 12, device=DEVICE)
    assert flagos.memory_allocated() > 0
    del live
    flagos.empty_cache()
    assert flagos.memory_allocated() == 0


def test_coexists_with_torch_backends_mps():
    """Both the flagos device and torch's own MPS backend stay usable.

    They own separate command queues and allocations, so forward tensors raised
    on one must round-trip without disturbing the other. MPS autograd still
    stops at a PyTorch engine assertion once check_mps_autograd is enabled;
    see that test and design doc section 10.
    """
    flagos_value = torch.ones(4, 4, dtype=torch.float32).to(DEVICE)
    assert torch.equal(flagos_value.cpu(), torch.ones(4, 4))

    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is unavailable in this torch build")
    mps_value = torch.ones(4, 4, dtype=torch.float32, device="mps")
    assert torch.equal(mps_value.cpu(), torch.ones(4, 4))
    # The flagos device is untouched by the MPS round-trip above.
    assert flagos_value.device.type == "flagos"


def _require_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is unavailable in this torch build")


def test_mps_to_flagos_and_back():
    _require_mps()
    host = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    source = host.to("mps")
    assert not source.is_contiguous()

    on_flagos = source.to(DEVICE)
    assert on_flagos.device.type == "flagos"
    assert on_flagos.shape == source.shape
    assert torch.equal(on_flagos.cpu(), host)

    back = on_flagos.to("mps")
    assert back.device.type == "mps"
    assert torch.equal(back.cpu(), host)


def test_copy_between_mps_and_flagos():
    _require_mps()
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source = host.to("mps")

    on_flagos = torch.empty(3, 4, device=DEVICE)
    on_flagos.copy_(source)
    assert torch.equal(on_flagos.cpu(), host)

    on_mps = on_flagos.to("mps")
    assert torch.equal(on_mps.cpu(), host)


def test_copy_mps_to_noncontiguous_flagos():
    _require_mps()
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    on_flagos_storage = torch.empty(4, 3, device=DEVICE)
    on_flagos = on_flagos_storage.t()
    assert not on_flagos.is_contiguous()
    on_flagos.copy_(host.to("mps"))
    assert torch.equal(on_flagos.cpu(), host)


def test_mps_to_flagos_converts_dtype_and_empty_tensor():
    _require_mps()
    host = torch.arange(6, dtype=torch.float32)
    on_flagos = host.to("mps").to(DEVICE, dtype=torch.float64)
    assert on_flagos.dtype == torch.float64
    assert torch.equal(on_flagos.cpu(), host.to(torch.float64))

    empty = torch.empty(0, 3, device="mps")
    copied = empty.to(DEVICE)
    assert copied.shape == empty.shape
    on_flagos_empty = torch.empty(0, 3, device=DEVICE)
    on_flagos_empty.copy_(empty)
    assert torch.equal(on_flagos_empty.cpu(), empty.cpu())


@pytest.mark.parametrize("swap", [False, True])
def test_module_mps_flagos_round_trip_preserves_parameters(swap):
    _require_mps()
    model = torch.nn.Linear(4, 3)
    expected = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    previous = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(swap)
    try:
        model.to("mps")
        model.to(DEVICE)
        assert all(
            parameter.device.type == "flagos" for parameter in model.parameters()
        )
        for name, value in model.named_parameters():
            assert torch.equal(value.cpu(), expected[name])

        model.to("mps")
        assert all(parameter.device.type == "mps" for parameter in model.parameters())
        for name, value in model.named_parameters():
            assert torch.equal(value.cpu(), expected[name])
    finally:
        torch.__future__.set_swap_module_params_on_conversion(previous)


@pytest.mark.parametrize(
    "code, expected",
    [
        pytest.param(
            """
import sys
import torch

mps_built = torch.backends.mps.is_built()
mps_available = torch.backends.mps.is_available()
mps_backward = False
if mps_available:
    x = torch.ones(3, device="mps", requires_grad=True)
    (x * x).sum().backward()
    mps_backward = torch.equal(x.grad.cpu(), torch.full((3,), 2.0))
print("torch_fl" in sys.modules, torch.accelerator.current_accelerator(False),
      mps_built, mps_available, mps_backward)
""",
            "False mps",
            id="bare-torch-mps-baseline",
        ),
        pytest.param(
            """
import sys
import torch
import torch_fl

mps_built = torch.backends.mps.is_built()
mps_available = torch.backends.mps.is_available()
forward = False
round_trip = False
if mps_available:
    x = torch.ones(3, device="mps")
    (x * x).sum()
    round_trip = torch.equal(x.cpu(), torch.ones(3))
    forward = True
    value = x.detach().requires_grad_(True)
    try:
        (value * value).sum().backward()
    except RuntimeError as error:
        message = str(error)
        if "opt_ready_stream && opt_parent_stream" not in message:
            raise
    else:
        raise AssertionError("MPS autograd unexpectedly accepted flagos")
print("torch_fl" in sys.modules, torch._C._get_privateuse1_backend_name(),
      mps_built, mps_available, forward, round_trip)
""",
            "True flagos",
            id="explicit-torch-fl-torch",
        ),
    ],
)
def test_mps_process_boundaries(code, expected, tmp_path):
    """A fresh process proves both the MPS baseline and explicit coexistence."""
    env = _child_env(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith(expected), result.stdout


def _child_env(tmp_path):
    """Remove import paths from this test process without hiding the installation.

    The integration harness imports torch_fl in conftest, so a baseline in this
    process would already carry the PrivateUse1 registration. A wheel run
    reaches its installed torch_fl from site-packages; an editable install keeps
    its import hook for the explicit-import scenario.
    """
    lib = active_torch_lib()
    if lib is None:
        pytest.skip("PyTorch is unavailable")

    version = Path(lib).parent / "version.py"
    if not version.is_file() or "_pytest" in version.read_text(encoding="utf-8"):
        pytest.skip("This test requires a real PyTorch installation")
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    inherited = env.get("PYTHONPATH", "").split(os.pathsep)
    removed = {
        os.path.abspath(item)
        for item in (
            tmp_path,
            Path(__file__).resolve().parents[1],
            Path(__file__).resolve().parents[2],
        )
    }
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in inherited if item and os.path.abspath(item) not in removed
    )
    return env


def test_mps_backward_with_flagos_meets_upstream_stream_precondition():
    """Track the upstream fix this backend needs before MPS training can work."""
    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is unavailable in this torch build")
    x = torch.ones(3, device="mps", requires_grad=True)
    with pytest.raises(RuntimeError, match="opt_ready_stream && opt_parent_stream"):
        (x * x).sum().backward()
