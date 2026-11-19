# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure the non-boxing Hygon model with explicit CPU fallback diagnostics.

This is ineligible for strict native-host acceptance. Run qwen_strict_eager.py
separately to test the required all-device route first.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys

from torch_fl import _build_config
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.device != "flagos:0" or _build_config.ACCELERATOR != "hygon":
        parser.error("diagnostic requires the Hygon flagos device")
    if not _build_config.NATIVE_HOSTS or "boxing" in _build_config.KERNELS:
        parser.error("diagnostic requires a non-boxing native-host wheel")
    torch.ops.flagos.strict_host_routes(False)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        runpy.run_path(
            str(Path(__file__).with_name("qwen_strict_eager.py")), run_name="__main__"
        )
    finally:
        if args.output.is_file():
            report = json.loads(args.output.read_text())
            report["diagnostic_policy"] = {
                "cpu_fallback_allowed": True,
                "boxing_allowed": False,
                "strict_native_host_acceptance_eligible": False,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
