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

"""Shared fixtures and helpers for the cross-backend profiler contract."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from platform_support import detect_platform


@dataclass(frozen=True)
class ProfilerCapabilities:
    """Observable profiler features expected from the active backend."""

    platform: str
    device: bool
    kernel: bool
    runtime: bool
    memcpy: bool
    memset: bool
    flow: bool
    linkage: bool
    metadata: bool


def _torch_device():
    import torch

    return torch.device("flagos", 0)


def _torch_module():
    import torch

    return torch


def capabilities_for_platform(platform: str) -> ProfilerCapabilities:
    """Describe public profiler features currently emitted by each tracer.

    The capability table is intentionally about observable behavior, not vendor
    library names. Ascend currently exposes CPU/Trace records only in CI; all
    other supported tracers are expected to provide kernel and runtime records.
    """
    device = platform != "ascend"
    runtime = device
    return ProfilerCapabilities(
        platform=platform,
        device=device,
        kernel=device,
        runtime=runtime,
        memcpy=device and platform in {"cuda", "metax", "ppu", "musa"},
        memset=device and platform in {"cuda", "metax"},
        flow=device,
        linkage=device,
        metadata=device,
    )


@pytest.fixture(scope="session")
def profiler_capabilities():
    """Capabilities for the active hardware/backend."""
    return capabilities_for_platform(detect_platform())


@pytest.fixture(scope="module")
def profile_result():
    """Capture one common workload and export it as a Chrome trace.

    Shape and iteration count are kept identical to ``_run_traced_ops()`` in
    test_profiler_parity.py, which is the workload proven to emit every activity
    class this module asserts on. It matters for memsets specifically: cuBLAS
    only allocates (and zeroes) a gemm workspace once the matmul is large enough
    and repeated enough to pick a workspace-using kernel. A 256x256 x3 loop stays
    under that threshold on CUDA and produced no gpu_memset events at all, while
    still passing on backends whose sort allocates zeroed scratch -- so shrinking
    this workload silently converts the memset assertion into a no-op on some
    vendors and a failure on others.
    """
    torch = _torch_module()
    device = _torch_device()
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    small = torch.randn(16, device=device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        with_stack=False,
    ) as prof:
        for _ in range(5):
            z = (x @ y).relu()
        torch.sort(small)
        z.sum().item()  # force sync so device activity lands inside the window

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as trace_file:
        trace_path = Path(trace_file.name)
    try:
        prof.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
    finally:
        trace_path.unlink(missing_ok=True)
    return prof, trace


def events_in(trace, category, *, completed_only=True):
    """Return trace events in one category."""
    events = [
        event for event in trace.get("traceEvents", []) if event.get("cat") == category
    ]
    if completed_only:
        return [event for event in events if event.get("ph") == "X"]
    return events


def event_categories(trace):
    """Return categories for completed trace events."""
    return {
        event.get("cat")
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X"
    }


def arg_key_union(trace, category):
    """Return the union of argument keys for all events in a category."""
    keys = set()
    for event in events_in(trace, category):
        keys.update((event.get("args") or {}).keys())
    return keys


def op_name_by_external_id(trace):
    """Map Kineto External ids to CPU operation names."""
    return {
        (event.get("args") or {}).get("External id"): event.get("name")
        for event in events_in(trace, "cpu_op")
        if (event.get("args") or {}).get("External id") is not None
    }
