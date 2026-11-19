# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
@pytest.mark.parametrize("vendor", ["nvidia", "metax"])
def test_vendor_requires_approved_scope_and_correlated_launch(launch_name, vendor):
    events = trace()
    events[0]["name"] = f"flagos::{vendor}::mm"
    events[1]["name"] = launch_name
    report = provenance.audit_kernels(
        events, {"published_gems"}, {"vendor": vendor, "catalog": [{"op": "mm"}]}
    )
    assert report["flaggems_counts"] == {"published_gems": 1}
    assert report["vendor_counts"] == {"vendor_gemm": 1}
    assert report["vendor_entry_launches"] == {f"flagos::{vendor}::mm": 1}


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


def correlated_trace():
    events = trace()
    for event in events[:3]:
        event.setdefault("args", {})["External id"] = 389
    return events


def test_external_correlation_handles_independent_trace_clocks():
    events = correlated_trace()
    events[1]["ts"] = events[0]["ts"] - 0.78
    report = provenance.audit_kernels(
        events, {"published_gems"}, {"catalog": [{"op": "mm"}]}
    )
    assert report["vendor_entry_launches"] == {"flagos::nvidia::mm": 1}


@pytest.mark.parametrize(
    "mutation",
    [
        "scope_id",
        "launch_id",
        "kernel_id",
        "thread",
        "correlation",
        "missing_launch",
        "unknown_scope",
    ],
)
def test_external_ids_preserve_scope_thread_and_api_admission(mutation):
    events = correlated_trace()
    policy = {"catalog": [{"op": "mm"}]}
    if mutation.endswith("_id"):
        index = {"scope_id": 0, "launch_id": 1, "kernel_id": 2}[mutation]
        events[index]["args"]["External id"] = 390
    elif mutation == "thread":
        events[1]["tid"] += 1
    elif mutation == "correlation":
        events[2]["args"]["correlation"] += 1
    elif mutation == "missing_launch":
        del events[1]
    else:
        policy = None
    with pytest.raises(RuntimeError):
        provenance.audit_kernels(events, {"published_gems"}, policy)


@pytest.mark.parametrize("value", [None, 0, True])
def test_unknown_external_ids_do_not_prove_out_of_scope_launches(value):
    events = trace()
    events[1]["ts"] = events[0]["ts"] - 0.78
    for event in events[:3]:
        event.setdefault("args", {})["External id"] = value
    with pytest.raises(RuntimeError):
        provenance.audit_kernels(
            events, {"published_gems"}, {"catalog": [{"op": "mm"}]}
        )


@pytest.mark.parametrize(
    "native_hosts,native_vendor,expected",
    [(True, False, True), (True, True, True), (False, False, False)],
)
def test_qwen_reports_sdk_precision_without_a_bundled_vendor_module(
    monkeypatch, native_hosts, native_vendor, expected
):
    package = ModuleType("torch_fl")
    package._build_config = SimpleNamespace(
        NATIVE_HOSTS=native_hosts, NATIVE_VENDOR=native_vendor
    )
    native = ModuleType("torch_fl.native_host")
    report = {"vendor": "metax", "hits": {"mm": 1}}
    native.vendor = SimpleNamespace(audit=lambda: report)
    monkeypatch.setitem(sys.modules, "torch_fl", package)
    monkeypatch.setitem(sys.modules, "torch_fl.native_host", native)
    source = ROOT / "benchmarks/host_compiler/qwen_eager.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "vendor_audit"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    assert namespace["vendor_audit"]() == (report if expected else None)
