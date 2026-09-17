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

"""
Non-zero-device dtype cast: a boxed op must not relocate the process.

Regression test for flagos-ai/Torch-FL#313.

A same-device dtype cast on a *non-zero* logical flagos device hands its tensors
to the native CUDA path (`csrc/aten/copy_ops.cc` boxes them to CUDA and calls
`at::native::copy_`). c10's device guards then move the CUDA device and do not
move it back: `c10::cuda::MaybeSetDevice` only calls `cudaSetDevice` when the
previous device has a primary context, and otherwise parks the index in its
`targetDeviceIndex` TLS for a `SetTargetDevice()` that never comes on this path.
The flagos runtime keys its own decisions on that same "current device"
(`empty_memory_format` emits a device guard only when the requested index
differs from the one it reads back, and `DeviceAllocator` tags each block with
the index read back at allocation time), so from then on the runtime reports the
cast device for the whole process, stops binding it for that device's tensors,
and lets their allocations follow whichever context happens to be current. The
per-device pool then holds segments belonging to two contexts, and Triton's
launcher rejects the odd one out with

    ValueError: Pointer argument (at N) cannot be accessed from Triton
    (cpu tensor?)

from `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_POINTER)`, or the
process dies inside `direct_copy_kernel_cuda` with an invalid page / illegal
memory access.

Device 0 is exempt -- it already has a primary context, so the missing restore is
invisible there -- which is why the device below is non-zero and the current
device is pinned to 0: with the cast device already current, the missing restore
would be invisible too.

The issue's own repro uses `flagos:15` after a 1.06 GiB allocation; both are
reproduced here (`INDEX` may be any non-zero index -- index 1 fails identically,
measured on PPU, so the test does not require a 16-device machine). The large
predecessor is kept because the issue calls it out as part of the repro and it is
what puts the allocator's per-context state where the split shows up, but the
small-predecessor case is covered as well so the regression is caught even on a
device that cannot hold the big tensor.

Both halves are asserted: the cast's *values* after a synchronize -- not merely
that dispatch succeeded -- and the process-level invariant that a boxed op
leaves the current device where it found it.

The corrupted state tolerates itself: once one op has run against the split
contexts, the current context agrees with the pointers again and the *next* op
of the same shape succeeds. So only the first such op in a process trips, which
makes the in-process tests order-dependent. `test_issue_313_repro_in_a_fresh_process`
runs the repro in its own interpreter for that reason, and is the one to trust
for "did this regress".

Only meaningful with 2+ devices, so it skips otherwise.

Usage:
    pytest tests/integration/test_nonzero_device_dtype_cast.py -v
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch_fl


pytestmark = pytest.mark.skipif(
    torch_fl.flagos.device_count() < 2,
    reason="needs at least 2 flagos devices",
)

# Index 1 specifically: index 0 is the device the bug does not affect, so
# asserting on it would pass either way.
INDEX = 1
DEVICE = f"flagos:{INDEX}"

# The issue's repro shape: 152064 * 3584 bfloat16 = 1.06 GiB.
LARGE_SHAPE = (152064, 3584)
SMALL_SHAPE = (1024,)


@pytest.fixture(autouse=True)
def current_device_zero():
    """Pin the current device to 0 for every test in this file.

    An op on a flagos:1 tensor has to switch to device 1 itself, and the bug is
    that it never switches back. If the test ran with device 1 already current,
    it would pass either way.
    """
    prev = torch_fl.flagos.current_device()
    torch_fl.flagos.set_device(0)
    try:
        yield
    finally:
        torch_fl.flagos.set_device(prev)


def _predecessor(shape):
    """The large allocation that precedes the cast in the issue's repro.

    Skipped rather than failed when the device is too small to hold it: the
    point of the tensor is to precede the cast, not to measure this machine.
    """
    try:
        return torch.empty(shape, dtype=torch.bfloat16, device=DEVICE)
    except RuntimeError as exc:
        if "memory" not in str(exc).lower():
            raise
        pytest.skip(f"{DEVICE} cannot hold the issue's {shape} repro tensor: {exc}")


# ---------------------------------------------------------------------------
# The issue's repro
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [LARGE_SHAPE, SMALL_SHAPE], ids=["large_predecessor", "small_predecessor"]
)
def test_int64_to_bool_cast_on_nonzero_device(shape):
    """int64 -> bool on flagos:1 after a preceding allocation.

    Asserts the values (`(1, 160)` of all-ones casts to all-True) and not just
    that the cast returned, plus the FlagGems reduction the issue failed on --
    `sum` allocates a scratch block per call and used to get one from the other
    context.
    """
    _predecessor(shape)
    m = torch.ones((1, 160), dtype=torch.int64, device=DEVICE)

    b = m.to(dtype=torch.bool)
    torch.cuda.synchronize()

    # The issue's failing line, and it has to come first: copying `b` to the host
    # beforehand re-binds the device and hides the bug.
    counted = int(b.sum().cpu())

    assert b.dtype == torch.bool
    assert b.device.type == "flagos"
    assert b.device.index == INDEX, f"result on flagos:{b.device.index}"
    assert counted == 160, f"expected 160 True entries, got {counted}"
    assert b.cpu().tolist() == [[True] * 160]


def test_cast_values_are_exact_on_nonzero_device():
    """Wrong data, not only crashes: a cross-context read returns whatever the
    other context's memory holds, which a crash-free run would not catch."""
    _predecessor(LARGE_SHAPE)
    base = torch.arange(160, dtype=torch.int64).reshape(1, 160)
    base = base * (base % 3 != 1)  # a fixed pattern of zeros and non-zeros
    m = base.to(DEVICE)

    b = m.to(dtype=torch.bool)
    torch.cuda.synchronize()

    assert b.cpu().tolist() == (base != 0).tolist()


# ---------------------------------------------------------------------------
# The same repro, in a fresh interpreter
# ---------------------------------------------------------------------------

_REPRO = textwrap.dedent(
    """
    import torch, torch_fl

    dev = "flagos:{index}"
    w = torch.empty((152064, 3584), dtype=torch.bfloat16, device=dev)
    m = torch.ones((1, 160), dtype=torch.int64, device=dev)
    b = m.to(dtype=torch.bool)
    torch.cuda.synchronize()
    print("true count:", int(b.sum().cpu()))
    """
).format(index=INDEX)


def test_issue_313_repro_in_a_fresh_process():
    """The issue's repro verbatim, in its own process.

    In-process is not enough: after the first op has run against the split
    contexts the process settles and later ops succeed, so whether the tests
    above trip depends on what ran before them. A fresh interpreter is the state
    the issue was reported against and the state CI must see fail, so this runs
    `python -c` and checks both the exit status and the printed value --
    the issue's `true count: 160`.
    """
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", _REPRO],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if "out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
        pytest.skip("the device cannot hold the issue's 152064x3584 repro tensor")
    assert result.returncode == 0, (
        f"repro exited {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )
    assert "true count: 160" in result.stdout, (
        f"unexpected repro output:\n{result.stdout}\n{result.stderr}"
    )


def test_cast_on_nonzero_device_leaves_current_device_alone():
    """The issue's workload, asserted on the invariant rather than the symptom.

    The corrupted allocator state needs an op order the repro above may not hit
    on every platform, so the leak itself is checked directly here: the cast is
    boxed to CUDA and must put the device back when it returns.
    """
    _predecessor(LARGE_SHAPE)
    assert torch_fl.flagos.current_device() == 0
    m = torch.ones((1, 160), dtype=torch.int64, device=DEVICE)

    m.to(dtype=torch.bool)
    torch.cuda.synchronize()

    assert torch_fl.flagos.current_device() == 0, (
        f"the cast left the flagos current device at {torch_fl.flagos.current_device()}"
    )


# ---------------------------------------------------------------------------
# The invariant behind it: a boxed op puts the current device back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,run",
    [
        ("to(bool)", lambda t: t.to(torch.bool)),
        ("to(float16)", lambda t: t.to(torch.float16)),
        ("to(same dtype)", lambda t: t.t().to(torch.int64)),
        ("cat", lambda t: torch.cat([t, t], 0)),
        ("contiguous", lambda t: t.t().contiguous()),
    ],
)
def test_boxed_op_leaves_current_device_untouched(name, run):
    """Every boxing op must leave the current device where it found it.

    The cast is only the case the issue hit; `torch.cat` leaked the device too,
    and any boxing op can, since all of them hand the tensors to c10's CUDA
    machinery. This is the root-cause assertion -- the corrupted allocator state
    is only the downstream symptom, and it needs a large allocation and a
    particular op order to surface, so asserting the invariant directly catches a
    regression that the repro above might miss.
    """
    t = torch.ones((8, 8), dtype=torch.int64, device=DEVICE)
    assert torch_fl.flagos.current_device() == 0

    run(t)
    torch.cuda.synchronize()

    assert torch_fl.flagos.current_device() == 0, (
        f"{name} left the flagos current device at {torch_fl.flagos.current_device()}"
    )
    assert torch.cuda.current_device() == 0, (
        f"{name} left the CUDA current device at {torch.cuda.current_device()}"
    )
