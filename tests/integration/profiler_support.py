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


def _mspti_in_link_map() -> bool:
    """Whether libmspti.so is mapped into this process right now."""
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    except OSError:
        return False
    return "libmspti.so" in maps


# Sampled at import, which is the only moment that answers the question being
# asked. This module is loaded as a pytest plugin before torch_fl preloads its
# device assets, so nothing has run a profiler session yet -- meaning a hit here
# can only come from a process-start preload. Sampling later would also see the
# lazy dlopen that CannDeviceTracer::start() performs, which does *not* enable
# interposition; measured on 910, a late check reports "preloaded" and then the
# memcpy assertion fails on a workload that produced no records.
_MSPTI_PRELOADED_AT_STARTUP = _mspti_in_link_map()


def mspti_preload_active() -> bool:
    """Whether CANN's process-start MSPTI interposer was loaded at startup.

    CANN 9.0 intercepts ``aclrtMemcpy*``/``aclrtMemset*`` by symbol
    interposition, so ``libmspti.so`` must already be in the ELF link map when
    ``libascendcl.so`` resolves those calls; a later ``dlopen`` cannot
    substitute, even when it happens before the first ACL call. CI establishes
    this in ``.github/scripts/set_env_ascend.sh``, alongside the other Ascend
    environment prerequisites, so the profiler contract is invoked with the same
    command on every platform.

    This reads the link map rather than ``LD_PRELOAD`` on purpose: ``ld.so``
    only warns and continues when a preloaded path does not exist, so a mistyped
    ``LD_PRELOAD`` yields the env string without the library, and an env-based
    check would claim a capability the process does not have.
    """
    return _MSPTI_PRELOADED_AT_STARTUP


def capabilities_for_platform(platform: str) -> ProfilerCapabilities:
    """Describe public profiler features currently emitted by each tracer.

    The capability table is intentionally about observable behavior, not vendor
    library names. Ascend currently exposes CPU/Trace records only in CI; all
    other supported tracers are expected to provide kernel and runtime records.
    """
    device = platform != "ascend"
    runtime = device
    # Ascend memcpy interception is gated on process-start LD_PRELOAD rather
    # than on `device` above: measured on Ascend 910 with CANN 9.0, the shared
    # profile_result() workload produces a real positive-duration gpu_memcpy
    # record when libmspti.so is preloaded at process start, and none at all
    # otherwise. Treat that as the sole memcpy capability signal for ascend so
    # the memcpy test skips (rather than silently passing or failing) when the
    # prerequisite is absent.
    ascend_memcpy = platform == "ascend" and mspti_preload_active()
    return ProfilerCapabilities(
        platform=platform,
        device=device,
        kernel=device,
        runtime=runtime,
        memcpy=(device and platform in {"cuda", "metax", "ppu", "musa"})
        or ascend_memcpy,
        # Ascend memset stays off even with the MSPTI preload present, and this
        # is a deliberate backend property rather than a gap to close. A direct
        # ctypes probe of aclrtMemset/aclrtMemsetAsync under process-start
        # preload does produce real positive-duration, positive-byte memset
        # records, so the CANN interception itself works. But torch.zeros() --
        # the only zeroing op the shared profile_result() workload and any
        # current Ascend op registration reach -- routes through the
        # aclnnInplaceZero kernel, not the allocator's aclrtMemset calls in
        # csrc/runtime/accelerator/ascend/memory.cc. Rerouting it to match the
        # MetaX allocator-memset path would make this record appear, and was
        # measured on Ascend 910 to cost 12.7us -> 133us at 1 MiB and
        # 17.4us -> 9080us at 64 MiB (aclrtMemsetAsync is worse still), so the
        # kernel routing stays and the capability stays off.
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
