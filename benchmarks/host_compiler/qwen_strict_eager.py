# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run qwen_eager.py with model graph-compilation entry points forbidden."""

import argparse
import json
from pathlib import Path
import runpy

import torch
import torch._dynamo
import torch._dynamo.eval_frame

called = []


def reject(*args, **kwargs):
    called.append(True)
    raise RuntimeError("model graph capture is forbidden in strict-eager acceptance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", default="plain")
    parser.add_argument("--model-host", action="store_true")
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    if args.model_host or args.mode != "plain":
        parser.error("strict eager requires --mode plain without --model-host")
    blocked = []
    for owner, names in (
        (torch, ("compile",)),
        (torch._dynamo, ("optimize", "export", "optimize_assert")),
        (torch._dynamo.eval_frame, ("optimize", "_optimize", "optimize_assert")),
        (torch.jit, ("script", "trace", "trace_module")),
    ):
        for name in names:
            if hasattr(owner, name):
                setattr(owner, name, reject)
                blocked.append(owner.__name__ + "." + name)
    runpy.run_path(str(Path(__file__).with_name("qwen_eager.py")), run_name="__main__")
    report = json.loads(args.output.read_text())
    if report["execution"] != "eager" or called:
        raise RuntimeError("model graph capture occurred during strict eager execution")
    if any(report.get("host_program_stats", [])):
        raise RuntimeError("legacy host plans were used during strict eager execution")
    report["model_capture_audit"] = {
        "blocked_entry_points": blocked,
        "calls": len(called),
    }
    args.output.write_text(json.dumps(report, indent=2))
