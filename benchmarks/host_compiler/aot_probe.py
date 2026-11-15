# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audit precompiled host functions on fresh inputs without native compilation."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", type=Path)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.oracle and args.reference is None:
        parser.error("native mode requires --reference from an oracle run")
    if not args.oracle and args.hosts is not None:
        sys.path.insert(0, str(args.hosts.resolve().parent))
        loader_path = (
            Path(__file__).resolve().parents[2] / "torch_fl/native_host/load.py"
        )
        spec = importlib.util.spec_from_file_location("host_loader", loader_path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        loader.install_launcher()
    import torch

    start = time.perf_counter()
    import torch_fl
    import flag_gems

    hosts = None
    load_ms = None
    active = getattr(getattr(torch_fl, "native_host", None), "_hosts", None)
    if args.oracle and active is not None:
        parser.error("oracle mode requires an ordinary FlagGems wheel")
    if not args.oracle:
        if args.hosts is None:
            hosts = active
            if hosts is None:
                parser.error("install a native-host wheel or specify --hosts")
        else:
            start = time.perf_counter()
            hosts = loader.Hosts(args.hosts)
            hosts.activate()
        load_ms = (time.perf_counter() - start) * 1000
    expected = torch.load(args.reference, weights_only=True) if args.reference else None
    torch.manual_seed(17)
    cases = []
    for shape in [(), (0,), (131,), (3, 263), (2, 5, 37)]:
        x = torch.randn(shape).to("flagos")
        cases.append(("cos", (x,)))
        cases.append(("sin", (x,)))
    x = torch.randn(31, 17).to("flagos").t()
    cases.extend([("cos", (x,)), ("add", (x, torch.randn(31).to("flagos")))])
    for m, n, k in [(17, 32, 64), (31, 80, 96)]:
        a = torch.randn(m, k, dtype=torch.bfloat16).to("flagos")
        b = torch.randn(k, n, dtype=torch.bfloat16).to("flagos")
        cases.append(("mm", (a, b)))
    cases.append(
        (
            "bmm",
            (torch.randn(3, 17, 32).to("flagos"), torch.randn(3, 32, 23).to("flagos")),
        )
    )
    cases.append(("softmax", (torch.randn(7, 131).to("flagos"), -1, False)))
    results = []
    outputs = []
    for index, (name, values) in enumerate(cases):
        compiled = getattr(flag_gems, name) if args.oracle else hosts.op(name)
        start = time.perf_counter()
        actual = compiled(*values)
        torch_fl.flagos.synchronize()
        first_ms = (time.perf_counter() - start) * 1000
        outputs.append(actual.cpu())
        if expected is not None:
            torch.testing.assert_close(outputs[-1], expected[index], rtol=0, atol=0)
        compiled(*values)
        frames = Counter()

        def profile(frame, event, result):
            if event == "call":
                code = frame.f_code
                frames[f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}"] += 1

        sys.setprofile(profile)
        try:
            compiled(*values)
        finally:
            sys.setprofile(None)
        timings = {}
        for mode, function in [("original" if args.oracle else "compiled", compiled)]:
            samples = []
            for _ in range(5):
                torch_fl.flagos.synchronize()
                start = time.perf_counter_ns()
                for _ in range(100):
                    _output = function(*values)
                samples.append((time.perf_counter_ns() - start) / 100 / 1000)
            timings[mode] = statistics.median(samples)
        results.append(
            {
                "op": name,
                "shape": list(values[0].shape),
                "first_ms": first_ms,
                "cpu_us": timings,
                "python_frames": dict(frames),
                "equal": True if expected is not None else None,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    report = {
        "host_load_ms": load_ms,
        "results": results,
        "native_audit": hosts.audit() if hosts else None,
    }
    torch.save(outputs, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
