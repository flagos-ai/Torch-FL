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

"""One public profiler contract shared by every FlagOS hardware backend."""

from collections import Counter

import pytest

from profiler_support import (
    arg_key_union,
    event_categories,
    events_in,
    op_name_by_external_id,
)

pytestmark = pytest.mark.profiler


@pytest.mark.anyplatform
def test_profiler_cpu_api_and_trace_export(profile_result):
    """The same public profiler API produces CPU operations and a valid trace."""
    prof, trace = profile_result

    assert prof.key_averages(), "profiler produced no key averages"
    events = trace.get("traceEvents")
    assert isinstance(events, list) and events, "Chrome trace has no traceEvents"
    assert events_in(trace, "cpu_op"), "Chrome trace has no cpu_op events"
    assert "Trace" in event_categories(trace), "Chrome trace has no capture span"


@pytest.mark.profiler_device
@pytest.mark.profiler_kernel
def test_profiler_kernel_events(profile_result, profiler_capabilities):
    """Device profiling exposes named kernels with positive durations."""
    if not profiler_capabilities.kernel:
        pytest.skip(f"{profiler_capabilities.platform} does not emit kernel events")

    kernels = events_in(profile_result[1], "kernel")
    assert kernels, "profiler produced no kernel events"
    assert all(event.get("name") for event in kernels)
    assert all(event.get("dur", 0) > 0 for event in kernels)


@pytest.mark.profiler_device
@pytest.mark.profiler_runtime
def test_profiler_runtime_events(profile_result, profiler_capabilities):
    """Device runtime activity uses the neutral PrivateUse1 runtime category."""
    if not profiler_capabilities.runtime:
        pytest.skip(f"{profiler_capabilities.platform} does not emit runtime events")

    runtimes = events_in(profile_result[1], "privateuse1_runtime")
    assert runtimes, "profiler produced no privateuse1_runtime events"
    assert all(event.get("name") for event in runtimes)
    assert all(event.get("dur", 0) > 0 for event in runtimes)
    assert "cbid" in arg_key_union(profile_result[1], "privateuse1_runtime")


@pytest.mark.profiler_device
@pytest.mark.profiler_metadata
def test_profiler_kernel_device_metadata(profile_result, profiler_capabilities):
    """Kernel records expose usable device and stream metadata."""
    if not profiler_capabilities.metadata:
        pytest.skip(f"{profiler_capabilities.platform} has no kernel metadata contract")

    kernels = events_in(profile_result[1], "kernel")
    assert kernels, "metadata check has no kernel events"
    for event in kernels:
        args = event.get("args") or {}
        assert args.get("device") is not None
        assert args.get("stream") is not None


@pytest.mark.profiler_device
@pytest.mark.profiler_flow
def test_profiler_flow_events_are_paired(profile_result, profiler_capabilities):
    """CPU-to-device flow arrows are renderable start/finish pairs."""
    if not profiler_capabilities.flow:
        pytest.skip(f"{profiler_capabilities.platform} does not emit flow events")

    flows = [
        event
        for event in profile_result[1].get("traceEvents", [])
        if event.get("cat") == "ac2g" and event.get("ph") in {"s", "f"}
    ]
    assert flows, "profiler produced no ac2g flow events"
    starts = {event.get("id") for event in flows if event.get("ph") == "s"}
    finishes = {event.get("id") for event in flows if event.get("ph") == "f"}
    assert starts == finishes
    assert starts


@pytest.mark.profiler_device
@pytest.mark.profiler_linkage
def test_profiler_device_time_linkage(profile_result, profiler_capabilities):
    """key_averages device time equals the linked device events in the trace."""
    if not profiler_capabilities.linkage:
        pytest.skip(f"{profiler_capabilities.platform} has no linkage contract")

    prof, trace = profile_result
    mm = next((event for event in prof.key_averages() if event.key == "aten::mm"), None)
    assert mm is not None, "aten::mm is absent from key_averages"
    assert mm.self_device_time_total > 0

    op_names = op_name_by_external_id(trace)
    linked = [
        event
        for category in ("kernel", "gpu_memcpy", "gpu_memset")
        for event in events_in(trace, category)
        if op_names.get((event.get("args") or {}).get("External id")) == "aten::mm"
    ]
    assert linked, "no device events link to aten::mm"
    truth = sum(event.get("dur", 0) for event in linked)
    assert abs(mm.self_device_time_total - truth) <= max(1.0, truth * 0.01)


@pytest.mark.profiler_device
@pytest.mark.profiler_memcpy
def test_profiler_memcpy_events(profile_result, profiler_capabilities):
    """Backends advertising memcpy support expose positive byte-count records."""
    if not profiler_capabilities.memcpy:
        pytest.skip(f"{profiler_capabilities.platform} does not emit memcpy events")

    copies = events_in(profile_result[1], "gpu_memcpy")
    assert copies, "profiler produced no gpu_memcpy events"
    assert all(event.get("dur", 0) > 0 for event in copies)
    assert all((event.get("args") or {}).get("bytes", 0) > 0 for event in copies)


@pytest.mark.profiler_device
@pytest.mark.profiler_memset
def test_profiler_memset_events(profile_result, profiler_capabilities):
    """Backends advertising memset support expose positive byte-count records."""
    if not profiler_capabilities.memset:
        pytest.skip(f"{profiler_capabilities.platform} does not emit memset events")

    memsets = events_in(profile_result[1], "gpu_memset")
    assert memsets, "profiler produced no gpu_memset events"
    assert all(event.get("dur", 0) > 0 for event in memsets)
    assert all((event.get("args") or {}).get("bytes", 0) > 0 for event in memsets)


@pytest.mark.profiler_device
@pytest.mark.profiler_metadata
def test_profiler_device_arg_keys(profile_result, profiler_capabilities):
    """Required neutral metadata fields remain present across device events."""
    if not profiler_capabilities.metadata:
        pytest.skip(f"{profiler_capabilities.platform} has no metadata contract")

    trace = profile_result[1]
    for category in ("kernel", "privateuse1_runtime"):
        events = events_in(trace, category)
        assert events, f"no {category} events for metadata validation"
        assert "External id" in arg_key_union(trace, category) or category == "kernel"


@pytest.mark.profiler_device
@pytest.mark.profiler_runtime
def test_profiler_capture_window_contains_device_events(
    profile_result, profiler_capabilities
):
    """Collected device/runtime events remain inside the Trace capture span."""
    if not profiler_capabilities.device:
        pytest.skip(f"{profiler_capabilities.platform} has no device activity")

    trace = profile_result[1]
    spans = events_in(trace, "Trace")
    assert spans, "trace has no capture window"
    span = max(spans, key=lambda event: event.get("dur", 0))
    lo = span["ts"]
    hi = lo + span.get("dur", 0)
    tracked_categories = {"privateuse1_runtime", "kernel", "gpu_memcpy", "gpu_memset"}
    tracked = [
        event
        for event in trace.get("traceEvents", [])
        if event.get("cat") in tracked_categories and event.get("ph") == "X"
    ]
    assert tracked, "capture-window check has no device/runtime events"
    assert all(
        event["ts"] >= lo and event["ts"] + event.get("dur", 0) <= hi
        for event in tracked
    )


@pytest.mark.profiler_device
@pytest.mark.profiler_kernel
def test_profiler_kernel_names_are_demangled(profile_result, profiler_capabilities):
    """Kernel names are readable rather than raw Itanium symbols."""
    if not profiler_capabilities.kernel:
        pytest.skip(f"{profiler_capabilities.platform} has no kernel activity")

    kernels = events_in(profile_result[1], "kernel")
    assert kernels
    assert not [event for event in kernels if event.get("name", "").startswith("_ZN")]


@pytest.mark.profiler_device
@pytest.mark.profiler_runtime
def test_profiler_runtime_names_are_not_all_fallback(
    profile_result, profiler_capabilities
):
    """Runtime records preserve callback identity instead of one hard-coded name."""
    if not profiler_capabilities.runtime:
        pytest.skip(f"{profiler_capabilities.platform} has no runtime activity")

    names = Counter(
        event.get("name")
        for event in events_in(profile_result[1], "privateuse1_runtime")
    )
    assert len(names) > 1 or "cbid" in arg_key_union(
        profile_result[1], "privateuse1_runtime"
    )
