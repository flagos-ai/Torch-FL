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

"""Focused Apple Silicon (M-series) device-runtime checks.

This is the stage-0 gate from ``.claude/skills/runtime-bringup/SKILL.md``: the
28-function C ABI plus the allocator, exercised under the same contracts as the
shared runtime-conformance suite. The hosted macOS CI lane runs that shared
suite for allocate, copy, stream, event and memory checks; this file adds the
intentional coexistence check with torch's own MPS backend. It predates the
MPSGraph operator layer, whose routing is covered by ``runtime/``.

Every case self-skips off an MPS build: the marker keeps CI manifests explicit,
and the skipif keeps a broad ``pytest tests/integration`` collection from
attempting Metal on a machine that has none.
"""

import os
import subprocess
import sys
from contextlib import nullcontext
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


def test_scalar_graph_operands_with_cache_disabled(tmp_path):
    program = """
import torch
import torch_fl
for dtype in (torch.float32, torch.bfloat16):
    value = torch.tensor([0.5], dtype=dtype)
    actual = value.to("flagos:0").exp().cpu()
    torch.testing.assert_close(actual, value.exp())
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={
            **os.environ,
            "FLAGOS_USE_CACHING_ALLOCATOR": "0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_fp32_rms_scale_matches_vendor_mps():
    generator = torch.Generator().manual_seed(42)
    values = torch.randn((1, 128, 1024), generator=generator).bfloat16().float()

    def rms_scale(device):
        squared = values.to(device).pow(2)
        scale = torch.rsqrt(squared.mean(-1, keepdim=True) + 1e-6)
        return squared.cpu(), scale.cpu()

    for result, reference in zip(rms_scale(DEVICE), rms_scale("mps")):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_host_to_device_round_trip():
    host = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    device = host.to(DEVICE)
    assert device.device.type == "flagos"
    assert torch.equal(device.cpu(), host)


@pytest.mark.parametrize("offset", [0, 16])
def test_explicit_contiguous_clone_copies_token_ids(offset):
    host = torch.arange(128).reshape(1, 128)
    source = host.to(DEVICE)[:, offset:]
    expected = host[:, offset:].clone(memory_format=torch.contiguous_format)
    actual = source.clone(memory_format=torch.contiguous_format)
    assert torch.equal(actual.cpu(), expected)
    assert actual.stride() == expected.stride()
    assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_float_to_bfloat16_cast_preserves_subnormals_and_rounds_ties():
    bits = torch.tensor(
        [
            0x00008000,
            0x00018000,
            0x3F808000,
            0x3F818000,
            -(1 << 31),
            0x7F800000,
            0x7F800001,
        ],
        dtype=torch.int32,
    )
    host = bits.view(torch.float32)
    actual = host.to(DEVICE).bfloat16().cpu()
    assert torch.equal(actual.view(torch.int16), host.bfloat16().view(torch.int16))


def test_bfloat16_to_float_cast_preserves_all_finite_bit_patterns():
    bits = torch.arange(1 << 16).to(torch.int16)
    finite = (bits & 0x7F80) != 0x7F80
    host = bits[finite].view(torch.bfloat16)
    actual = host.to(DEVICE).float().cpu()
    assert torch.equal(actual.view(torch.int32), host.float().view(torch.int32))


@pytest.mark.parametrize("use_default_stream", [True, False])
def test_metadata_view_keeps_queued_graph_inputs_alive(use_default_stream):
    streams = [None, None] if use_default_stream else [flagos.Stream(), flagos.Stream()]
    reference = torch.arange(128, dtype=torch.float32).reshape(16, 8) / 128
    device_inputs = []
    for stream in streams:
        with nullcontext() if stream is None else flagos.stream(stream):
            device_inputs.append(reference.to(DEVICE))
    results = []
    for i in range(12):
        stream = streams[i % 2]
        with nullcontext() if stream is None else flagos.stream(stream):
            value = device_inputs[i % 2] + i
            view = torch.ops.aten._unsafe_view.default(value, [8, 16])
            results.append(view.neg())
            del value, view
            torch.empty_like(device_inputs[i % 2]).zero_()
    flagos.synchronize()
    for i, result in enumerate(results):
        torch.testing.assert_close(result.cpu(), -(reference + i).reshape(8, 16))


def test_stream_and_timing_event_synchronize():
    from torch_fl.flagos._streams import get_raw_stream

    stream = flagos.Stream()
    default = get_raw_stream(0)
    assert stream.cuda_stream != default
    event = flagos.Event(enable_timing=True)
    with flagos.stream(stream):
        assert get_raw_stream(0) == flagos.current_stream(0).cuda_stream
        assert get_raw_stream(0) == stream.cuda_stream
        y = torch.empty(1 << 20, device=DEVICE)
        y.fill_(1.0)
    assert get_raw_stream(0) == default
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


def test_coexists_with_torch_backends_apple():
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
forward = False
round_trip = False
if mps_available:
    x = torch.ones(3, device="mps")
    (x * x).sum()
    round_trip = torch.equal(x.cpu(), torch.ones(3))
    forward = True
print("torch_fl" in sys.modules, torch.accelerator.current_accelerator(False),
      mps_built, mps_available, forward, round_trip)
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
    if expected.startswith("False "):
        pytest.skip(
            "The bare baseline's torch accelerators are pinned by "
            "test_plain_torch_reports_the_accelerators_this_torch_finds, which "
            "compares the child's output with this process's own torch."
        )
    # The suite has already run flagos and torch op tests against the one
    # process-wide MPS budget. Let go of their tensors and caches before a
    # child process allocates on the same device.
    import gc

    flagos.synchronize()
    gc.collect()
    flagos.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="torch.backends.mps is unavailable in this torch build",
)
def test_mps_backward_on_plain_torch_is_measured_not_assumed(tmp_path):
    """Pin the bare baseline's backward to whatever this torch build does.

    This used to be a child of test_mps_process_boundaries, asserting that the
    bare baseline's backward works -- and the hosted macOS image failed it with

      RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT
      FAILED at .../autograd/engine.cpp

    although the child imported no torch_fl at all. A child that only imports
    torch cannot be affected by flagos, so which behaviour it shows is a
    property of that torch build rather than of this suite, and hard-coding
    one of them fails the lane on a build that does the other. The suite still
    pins a boundary deliberately: the backward either succeeds and must give
    the right gradient, or fails with exactly that precondition message.
    """
    code = """
import sys
import torch

x = torch.ones(3, device="mps", requires_grad=True)
(x * x).sum().backward()
ok = torch.equal(x.grad.cpu(), torch.full((3,), 2.0))
print("ok" if ok else "wrong-gradient")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=_child_env(tmp_path),
    )
    if result.returncode == 0:
        assert result.stdout.strip() == "ok", result.stdout
    else:
        assert "opt_ready_stream && opt_parent_stream" in result.stderr, result.stderr


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="torch.backends.mps is unavailable in this torch build",
)
def test_plain_torch_reports_the_accelerators_this_torch_finds(tmp_path):
    """Measure the bare baseline against this process's own torch, not a pinned set.

    The child imports no torch_fl, so what it reports is a property of the
    installed torch build. Every wheel carries the `torch.backends` autoload
    hook (pyproject.toml), so `import torch` alone also imports torch_fl and
    `torch.accelerator.current_accelerator()` answers "flagos" -- PyTorch's own
    torch-first import (the Inductor compile workers rely on it). The hosted
    macOS image and a Linux box with this wheel therefore disagree, and a
    hard-coded "mps" fails whichever build it was not written on.

    This process imported torch_fl first, so its torch answers "mps" unless the
    build reports something else; the child, importing torch alone, says
    "flagos". That difference is the whole assertion: the hook ran, and it did
    not stop the child from finding the accelerators torch itself finds.
    """
    code = """
import sys
import torch

print("torch_fl" in sys.modules, torch.accelerator.current_accelerator(False),
      torch.backends.mps.is_built(), torch.backends.mps.is_available())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=_child_env(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    hooked, accelerator, mps_built, mps_available = result.stdout.split()
    assert hooked == "True", result.stdout  # the wheel's hook ran in bare torch
    assert (mps_built, mps_available) == (
        str(torch.backends.mps.is_built()),
        str(torch.backends.mps.is_available()),
    ), result.stdout
    # The only difference the hook makes to bare torch is the PrivateUse1 name
    # it registers; the accelerator it reports is this build's own choice.
    assert accelerator in {"mps", "flagos"}, result.stdout
