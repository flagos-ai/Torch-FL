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
Multi-device factory regression: a factory op must honor its `device` index.

The FlagGems factory path (CallPythonOp_Factory / CallPythonOp_LikeFactory in
csrc/aten/backends/flagos/python_op_caller.cc) used to inject a hardcoded
`device=flagos:0` kwarg, dropping the `device` argument the generated kernels
already had in scope. Two consequences, both platform-independent:

  1. torch.ones(4, device="flagos:1").device reported flagos:0.
  2. Worse, the output was allocated on device 0 while gems' Triton kernel
     launched on the *current* device -- a cross-device write, which faults the
     GPU (observed on DCU as "Invalid address access" / KERNEL VMFault) rather
     than raising a Python error. That made every rank>0 worker crash under
     FlagGems, since DDP workers build their tensors via factories.

Only meaningful with 2+ devices, so it skips otherwise. It is also worth running
with FLAGOS_USE_FLAGGEMS unset: the boxing path must stay correct too.

Usage:
    FLAGOS_USE_FLAGGEMS=1 pytest tests/integration/test_factory_device_index.py -v
"""

import pytest
import torch
import torch_fl


# `multi_device` is the selection hook for the manifests' shared
# `Multi-device contracts` step (issue #391); the skipif below stays here so the
# hardware guard lives with the test rather than in the manifest.
pytestmark = [
    pytest.mark.multi_device,
    pytest.mark.skipif(
        torch_fl.flagos.device_count() < 2,
        reason="needs at least 2 flagos devices",
    ),
]

# Device 1 specifically: index 0 is what the bug produced, so asserting on it
# would pass either way.
DEVICE = "flagos:1"
INDEX = 1


def _assert_on_device1(t: torch.Tensor, what: str) -> None:
    assert t.device.type == "flagos", f"{what}: got device type {t.device.type}"
    assert t.device.index == INDEX, (
        f"{what}: allocated on flagos:{t.device.index}, expected flagos:{INDEX} "
        f"-- the factory dropped its device argument"
    )


# ---------------------------------------------------------------------------
# Shape factories: torch.<fn>(size, device=...)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [torch.ones, torch.zeros, torch.empty, torch.rand, torch.randn]
)
def test_shape_factory_honors_device_index(fn):
    t = fn(4, 8, device=DEVICE)
    _assert_on_device1(t, fn.__name__)
    assert t.shape == (4, 8)


@pytest.mark.parametrize(
    "fn,expected",
    [(torch.ones, 1.0), (torch.zeros, 0.0)],
    ids=["ones", "zeros"],
)
def test_fill_factory_values_are_correct(fn, expected):
    """Values, not just metadata: a cross-device write can leave the buffer
    untouched (or corrupt), which the device assertion alone would not catch."""
    t = fn(64, device=DEVICE)
    torch.testing.assert_close(t.cpu(), torch.full((64,), expected))


def test_full_honors_device_index():
    t = torch.full((3, 5), 2.5, device=DEVICE)
    _assert_on_device1(t, "full")
    torch.testing.assert_close(t.cpu(), torch.full((3, 5), 2.5))


def test_arange_honors_device_index():
    t = torch.arange(0, 10, device=DEVICE)
    _assert_on_device1(t, "arange")
    torch.testing.assert_close(t.cpu(), torch.arange(0, 10))


def test_eye_honors_device_index():
    t = torch.eye(4, device=DEVICE)
    _assert_on_device1(t, "eye")
    torch.testing.assert_close(t.cpu(), torch.eye(4))


def test_linspace_honors_device_index():
    t = torch.linspace(0, 1, 11, device=DEVICE)
    _assert_on_device1(t, "linspace")
    torch.testing.assert_close(t.cpu(), torch.linspace(0, 1, 11), rtol=1e-4, atol=1e-4)


def test_random_factory_values_are_plausible():
    """randn on a non-zero device must produce real samples, not zeros/garbage."""
    t = torch.randn(4096, device=DEVICE)
    _assert_on_device1(t, "randn")
    c = t.cpu()
    assert abs(c.mean().item()) < 0.15, c.mean().item()
    assert 0.8 < c.std().item() < 1.2, c.std().item()


# ---------------------------------------------------------------------------
# *_like factories: device defaults to the source tensor's device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [torch.ones_like, torch.zeros_like, torch.rand_like, torch.randn_like]
)
def test_like_factory_defaults_to_source_device(fn):
    src = torch.empty(4, 8, device=DEVICE)
    _assert_on_device1(src, "empty (source)")
    _assert_on_device1(fn(src), f"{fn.__name__} (no explicit device)")


def test_full_like_defaults_to_source_device():
    src = torch.empty(2, 3, device=DEVICE)
    out = torch.full_like(src, 7.0)
    _assert_on_device1(out, "full_like")
    torch.testing.assert_close(out.cpu(), torch.full((2, 3), 7.0))


# ---------------------------------------------------------------------------
# Interaction with the current device
# ---------------------------------------------------------------------------


def test_explicit_device_wins_over_current_device():
    """An explicit index must be honored regardless of the current device, and
    setting the current device must not be required to reach device 1."""
    prev = torch_fl.flagos.current_device()
    try:
        torch_fl.flagos.set_device(0)
        _assert_on_device1(torch.ones(16, device=DEVICE), "ones with current=0")
    finally:
        torch_fl.flagos.set_device(prev)


def test_index_less_device_follows_current_device():
    """device="flagos" (no index) means "current device", matching aten factory
    semantics -- the same rule csrc/aten/empty.cc applies via device_or_default."""
    prev = torch_fl.flagos.current_device()
    try:
        torch_fl.flagos.set_device(INDEX)
        t = torch.ones(16, device="flagos")
        _assert_on_device1(t, "ones with index-less device")
        torch.testing.assert_close(t.cpu(), torch.ones(16))
    finally:
        torch_fl.flagos.set_device(prev)
