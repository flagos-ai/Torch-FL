#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a diagnostic TOPSPTI trace without using it as acceptance timing.

Run after qwen_strict_eager.py --profile-trace in a process with the profiling
capabilities required by the installed TOPS driver. API traces are optional and
come from enflame_api_trace.cc in a separate, deliberately instrumented run.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def union_us(intervals):
    total = 0.0
    end = float("-inf")
    for begin, stop in sorted(intervals):
        total += max(0.0, stop - max(begin, end))
        end = max(end, stop)
    return total


def artifact(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    copies = [e for e in events if "memcpy" in e.get("cat", "").lower()]
    scopes = [e for e in events if e.get("name") in ("qwen::prefill", "qwen::decode")]
    result = {"artifact": artifact(path), "scopes": {}}
    for scope in scopes:
        begin, end = scope["ts"], scope["ts"] + scope["dur"]
        selected = [e for e in kernels if begin <= e["ts"] < end]
        transfers = [e for e in copies if begin <= e["ts"] < end]
        intervals = [
            (max(begin, e["ts"]), min(end, e["ts"] + e["dur"]))
            for e in selected + transfers
        ]
        busy = union_us(intervals)
        names = defaultdict(lambda: {"calls": 0, "duration_ms": 0.0})
        for event in selected:
            names[event["name"]]["calls"] += 1
            names[event["name"]]["duration_ms"] += event["dur"] / 1000
        result["scopes"][scope["name"].split("::")[1]] = {
            "profiled_wall_ms": scope["dur"] / 1000,
            "kernel_count": len(selected),
            "kernel_sum_ms": sum(e["dur"] for e in selected) / 1000,
            "copy_count": len(transfers),
            "copy_sum_ms": sum(e["dur"] for e in transfers) / 1000,
            "device_busy_union_ms": busy / 1000,
            "host_or_device_idle_ms": (scope["dur"] - busy) / 1000,
            "top_kernels": sorted(
                ({"name": key, **value} for key, value in names.items()),
                key=lambda e: e["duration_ms"],
                reverse=True,
            )[:20],
        }
    if len(scopes) != 2 or not kernels:
        raise ValueError(
            "expected two synchronized model scopes and nonempty device kernel events"
        )
    return result


def summarize_api(path):
    events = json.loads(path.read_text())["events"]
    libraries = defaultdict(set)
    for event in events:
        for frame in event.get("frames", []):
            if Path(frame["library"]).name == "libtorch_fl.so":
                libraries[frame["library"]].add(frame["offset"])
    symbols = {}
    for library, offsets in libraries.items():
        offsets = sorted(offsets)
        output = subprocess.run(
            ["addr2line", "-f", "-C", "-e", library, *(hex(o) for o in offsets)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        symbols.update(
            {
                (library, offset): output[2 * i].split("(")[0]
                for i, offset in enumerate(offsets)
            }
        )
    counts = Counter()
    cpu_ms = Counter()
    transfers = Counter()
    tensor_readbacks = []
    for event in events:
        counts[event["api"]] += 1
        cpu_ms[event["api"]] += (event["end_ns"] - event["start_ns"]) / 1e6
        if event["api"] != "topsMemcpy":
            continue
        stack = [symbols.get((f["library"], f["offset"]), "") for f in event["frames"]]
        roots = tuple(
            dict.fromkeys(s for s in stack if "KernelGcu" in s or "gcu::DtypeCast" in s)
        )
        transfers[(event["kind"], event["bytes"], roots)] += 1
        # The measured device-side range checks export a single bool. This is
        # a necessary audit check, not a claim that arbitrary 1-byte transfers
        # prove absence of all possible host computation.
        if event["kind"] == 2 and event["bytes"] != 1:
            tensor_readbacks.append({"bytes": event["bytes"], "roots": roots})
    return {
        "artifact": artifact(path),
        "api_cpu_times": [
            {"api": name, "calls": count, "cpu_ms": cpu_ms[name]}
            for name, count in counts.items()
        ],
        "transfers": [
            {"kind": kind, "bytes": size, "roots": roots, "calls": count}
            for (kind, size, roots), count in transfers.items()
        ],
        "tensor_readbacks": tensor_readbacks,
        "only_scalar_bool_readbacks": not tensor_readbacks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--api-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_note": "Profiler and API observer overhead are included; use separate unprofiled runs for the 80% gate. Device idle time includes host submission gaps, waits and instrumentation.",
        "device_profile": summarize_trace(args.trace),
    }
    if args.api_trace:
        result["api_profile"] = summarize_api(args.api_trace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
