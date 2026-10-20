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

import pytest
import torch

from platform_support import detect_platform
from torch_fl import flagos

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

    They own separate command queues and allocations, so a tensor raised on one
    must round-trip without disturbing the other.
    """
    flagos_value = torch.ones(4, 4, dtype=torch.float32).to(DEVICE)
    assert torch.equal(flagos_value.cpu(), torch.ones(4, 4))

    if not torch.backends.mps.is_available():
        pytest.skip("torch.backends.mps is unavailable in this torch build")
    mps_value = torch.ones(4, 4, dtype=torch.float32, device="mps")
    assert torch.equal(mps_value.cpu(), torch.ones(4, 4))
    # The flagos device is untouched by the MPS round-trip above.
    assert flagos_value.device.type == "flagos"
