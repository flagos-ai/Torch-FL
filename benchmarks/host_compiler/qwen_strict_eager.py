# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run qwen_eager.py with model graph-compilation entry points forbidden."""

# Device libraries must precede Torch import on the CPU-Torch Hygon build.
# ruff: noqa: E402

import argparse
import json
from pathlib import Path
import runpy
import sys

# Hygon's CPU frontend must load its device libraries before Torch caches hooks.
_backend_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_backend_parser.add_argument("--device")
if (_backend_parser.parse_known_args()[0].device or "").startswith("flagos"):
    import torch_fl  # noqa: F401

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
    parser.add_argument("--device")
    args, _ = parser.parse_known_args()
    if args.model_host or args.mode != "plain":
        parser.error("strict eager requires --mode plain without --model-host")
    if args.device == "gcu:0":
        # Initialize the vendor plugin before blocking model capture APIs, as
        # Torch itself is initialized above. The model is loaded after the gate.
        import torch_gcu  # noqa: F401

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
    # runpy does not add the target's sibling helper directory to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from logits_validation import LogitsValidationFailed

    exit_code = 0
    try:
        runpy.run_path(
            str(Path(__file__).with_name("qwen_eager.py")), run_name="__main__"
        )
    except LogitsValidationFailed as error:
        # Failed accuracy is still an audited result, never a successful run.
        print(str(error), file=sys.stderr)
        exit_code = 1
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
    if exit_code:
        raise SystemExit(exit_code)
