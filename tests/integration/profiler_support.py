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
    cbid: bool
    memcpy: bool
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


# Every tracer names a runtime record from the record's own callback id, and every
# tracer has a placeholder for "this id has no name" -- one label that stands in
# for the whole id space. A trace that is mostly these is a trace whose runtime
# identity has collapsed: the records are real, but distinct API calls are
# indistinguishable in Chrome trace, Perfetto, and key_averages().
#
# These are the placeholders each tracer falls back to, one entry per tracer so
# the contract can bound the degradation on every platform rather than only the
# one it was measured on:
#
#   cudaRuntime    cupti_shim.h, whenever the vendor table does not name the id
#   mcRuntime      mcptiRuntimeCbidToName, for MCPTI builds without
#                  mcptiActivityGetApiName
#   musaRuntime    musa_mupti_device_tracer.cc, when MuptiShim::GetCallbackName
#                  does not resolve the id
#   topsRuntime    gcu_topspti_device_tracer.cc, when topsptiGetCallbackName does
#                  not resolve the id
#   AscendRuntime  cann_device_tracer.cc, for a record with no name field
#   Unknown        roctracer_device_tracer.cc, when roctracer_op_string returns
#                  null
GENERIC_RUNTIME_NAMES = frozenset(
    {
        "cudaRuntime",
        "mcRuntime",
        "musaRuntime",
        "topsRuntime",
        "AscendRuntime",
        "Unknown",
    }
)

# Ceiling on the share of runtime events allowed to carry a placeholder. Stock
# torch+CUDA names every one of its cuda_runtime events, so the honest bound is
# zero; this budget exists so that one id a newer toolkit introduces -- or one
# gap in a vendor resolver this repository does not control -- does not red the
# suite on a platform where the table itself is correct. It is far below the
# degradation this bound was written to catch: flagos reported 180/203 (89%) on
# CUDA and 72/117 (62%) on PPU before the cbid table was generated.
MAX_GENERIC_RUNTIME_FRACTION = 0.05


def capabilities_for_platform(platform: str) -> ProfilerCapabilities:
    """Describe public profiler features currently emitted by each tracer.

    The capability table is intentionally about observable behavior, not vendor
    library names.

    Ascend was the last backend held out of the device-side rows, and it was held
    out by a stale observation rather than by a measured gap: its row predates
    the CANN MSPTI tracer and left device/kernel/runtime/flow/metadata False, so
    ten of the twelve cases in tests/integration/test_profiler_contract.py
    skipped and the two that ran could not fail. Measured on Ascend 910 with
    CANN 9.0 and the process-start preload that .github/configs/ascend.yml
    installs, nine of those ten pass; the tenth asserted a runtime ``cbid``
    argument that MSPTI has no field for, and is now its own row below.

    Device-time linkage was the one row that did not open with them: MSPTI
    stamped every launch with the CPU op that started *after* it, so the
    launching operator kept ``self_device_time_total == 0.0``. That turned out
    to be this repository's defect rather than the vendor's -- ``popCorrelation``
    passed a null out-parameter, which CANN rejects without unwinding, leaving
    the external-correlation stack to grow for the life of the process (issue
    #425). With the pop fixed the launching operator owns its device events, so
    the ``linkage`` row below is True on its measured merits and
    ``test_profiler_device_time_linkage`` runs as an ordinary case on the Ascend
    job instead of reporting as an XFAIL.
    """
    device = True
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
        # The tracers that resolve a runtime record's identity through a vendor
        # callback-id table also stamp that id onto the event's args (cupti,
        # roctracer, mupti, topspti all set a `cbid`). CANN's MSPTI runtime
        # record has no callback-id field at all -- it carries the API name the
        # vendor already resolved -- so cann_device_tracer.cc names the event
        # from `record->name` and the arg union is {"External id", "correlation",
        # "thread"}. Measured with the contract's own workload on Ascend 910:
        # 0 of 15 privateuse1_runtime events carry a placeholder name.
        #
        # So this is the same property reached a different way, not a gap. The
        # property the arg protects is that distinct runtime calls stay
        # distinguishable in the trace, and test_profiler_runtime_names_are_not_all_fallback
        # asserts it directly on every platform. Synthesizing a cbid would mean
        # keying it on the activity correlationId, and CANN 9.0's callback
        # surface returns a repeated stack-address-shaped value there -- see the
        # reproduction recorded in issue #195.
        cbid=platform != "ascend",
        memcpy=(device and platform in {"cuda", "metax", "ppu", "musa"})
        or ascend_memcpy,
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

    The MetaX MCPTI tracer currently segfaults while Kineto processes the
    captured trace, before any contract assertion can run. Keep the shared
    contract honest by skipping the fixture on that platform until the tracer
    can safely export this workload.

    """
    if detect_platform() == "metax":
        pytest.skip("MetaX profiler trace export is currently unstable")

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


def device_memcpy_count(fn):
    """Count device copies by activity category, including unnamed directions.

    CUPTI calls its records simply ``Memcpy``. Count all device copy records
    conservatively: no direction inferred from a display name can hide a copy.
    A missing device trace cannot establish that an operation is transfer-free.
    """
    torch = _torch_module()
    fn()
    torch.flagos.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        fn()
        torch.flagos.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
    copies = events_in(trace, "gpu_memcpy")
    assert copies or events_in(trace, "kernel"), (
        "Profiler captured no device activities; zero copies is not evidence"
    )
    return len(copies)


@pytest.fixture(scope="module")
def device_copy_profiler():
    """Prove host transfers are observable before measuring a zero-copy op."""
    torch = _torch_module()
    host = torch.randn(64, 64)
    copies = device_memcpy_count(lambda: host.to("flagos:0").cpu())
    assert copies >= 2, (
        "Profiler missed the known H2D/D2H round trip; cannot validate zero copies"
    )
    return device_memcpy_count


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
