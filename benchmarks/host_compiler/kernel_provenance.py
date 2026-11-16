# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute GPU launches to audited FlagGems or typed vendor entries."""

from collections import Counter, defaultdict


def audit_kernels(events, admitted, vendor=None):
    approved = {
        "flagos::nvidia::" + row["op"] for row in (vendor or {}).get("catalog", [])
    }
    scopes = defaultdict(list)
    launches = defaultdict(list)
    for event in events:
        name = event.get("name", "")
        if name.startswith("flagos::nvidia::"):
            if name not in approved or event.get("ph") != "X":
                raise RuntimeError(f"unapproved native vendor scope: {name}")
            scopes[event["pid"], event["tid"]].append(event)
        if (
            event.get("cat") in {"privateuse1_runtime", "cuda_runtime"}
            and "Launch" in name
        ):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation].append(event)
    kernels = [event for event in events if event.get("cat") == "kernel"]
    counts, gems, native, unmatched = Counter(), Counter(), Counter(), Counter()
    vendor_entries = Counter()
    for kernel in kernels:
        name = kernel["name"]
        counts[name] += 1
        if name in admitted:
            gems[name] += 1
            continue
        owners = set()
        for launch in launches.get(kernel.get("args", {}).get("correlation"), []):
            for scope in scopes[launch["pid"], launch["tid"]]:
                if scope["ts"] <= launch["ts"] <= scope["ts"] + scope["dur"]:
                    owners.add(scope["name"])
        if len(owners) == 1:
            native[name] += 1
            vendor_entries[owners.pop()] += 1
        else:
            unmatched[name] += 1
    if not kernels or unmatched:
        raise RuntimeError(f"GPU kernel provenance audit failed: {dict(unmatched)}")
    return {
        "model_calls": 2,
        "counts": dict(counts),
        "flaggems_counts": dict(gems),
        "vendor_counts": dict(native),
        "vendor_entry_launches": dict(vendor_entries),
        "unmatched": [],
        "summed_device_ms": sum(event["dur"] for event in kernels) / 1000,
    }
