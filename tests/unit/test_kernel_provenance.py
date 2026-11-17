# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "kernel_provenance_test", ROOT / "benchmarks/host_compiler/kernel_provenance.py"
)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def trace():
    return [
        {
            "ph": "X",
            "name": "flagos::nvidia::mm",
            "pid": 1,
            "tid": 2,
            "ts": 10,
            "dur": 5,
        },
        {
            "cat": "privateuse1_runtime",
            "name": "cudaLaunchKernel",
            "pid": 1,
            "tid": 2,
            "ts": 11,
            "args": {"correlation": 3},
        },
        {"cat": "kernel", "name": "vendor_gemm", "dur": 2, "args": {"correlation": 3}},
        {"cat": "kernel", "name": "published_gems", "dur": 1},
    ]


@pytest.mark.parametrize(
    "launch_name", ["cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx"]
)
def test_vendor_requires_approved_scope_and_correlated_launch(launch_name):
    events = trace()
    events[1]["name"] = launch_name
    report = provenance.audit_kernels(
        events, {"published_gems"}, {"catalog": [{"op": "mm"}]}
    )
    assert report["flaggems_counts"] == {"published_gems": 1}
    assert report["vendor_counts"] == {"vendor_gemm": 1}
    assert report["vendor_entry_launches"] == {"flagos::nvidia::mm": 1}


@pytest.mark.parametrize(
    "mutation", ["correlation", "thread", "time", "no_launch", "no_policy"]
)
def test_vendor_kernel_name_alone_never_proves_origin(mutation):
    events = trace()
    policy = {"catalog": [{"op": "mm"}]}
    if mutation == "correlation":
        events[2]["args"]["correlation"] = 4
    elif mutation == "thread":
        events[1]["tid"] = 3
    elif mutation == "time":
        events[1]["ts"] = 16
    elif mutation == "no_launch":
        del events[1]
    else:
        policy = None
    with pytest.raises(RuntimeError):
        provenance.audit_kernels(events, {"published_gems"}, policy)


def test_unknown_kernel_and_empty_trace_are_rejected():
    for events in ([], [{"cat": "kernel", "name": "unknown", "dur": 1}]):
        with pytest.raises(RuntimeError, match="provenance audit failed"):
            provenance.audit_kernels(events, {"published_gems"})
