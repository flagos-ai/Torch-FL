# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Record real eager-model ATen metadata without copying tensor values.

Optionally reject Python Tensor.cpu calls inside model forwards, including the
published FlagGems scalar fallback. SDK/API tracing is still required to audit
C++ readbacks. This wrapper supplies diagnostics, never acceptance timings.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": list(value.shape),
            "stride": list(value.stride()),
        }
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class Census(TorchDispatchMode):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.forwards = 0
        self.active = False
        self.in_forward = False
        self.counts = Counter()
        self.readbacks = []
        self.api_directory = None
        self.api_start = 0
        self.api_count = 0
        self.api_context = None
        self.api_files = []

    def begin(self, module, inputs):
        self.active = self.forwards < self.limit
        self.in_forward = True
        if (
            self.api_directory
            and self.api_start <= self.forwards < self.api_start + self.api_count
        ):
            from qwen_eager import api_capture

            path = self.api_directory / f"forward-{self.forwards:03d}.json"
            self.api_context = api_capture(path)
            self.api_context.__enter__()
            self.api_files.append(str(path))

    def end(self, module, inputs, result):
        if self.api_context is not None:
            self.api_context.__exit__(None, None, None)
            self.api_context = None
        self.active = False
        self.in_forward = False
        self.forwards += 1

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = function(*args, **kwargs)
        if self.active:
            key = json.dumps(
                {
                    "forward": self.forwards,
                    "op": str(function),
                    "args": describe(args),
                    "kwargs": describe(kwargs),
                    "result": describe(result),
                },
                sort_keys=True,
            )
            self.counts[key] += 1
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--census-output", type=Path, required=True)
    parser.add_argument("--census-forwards", type=int, default=2)
    parser.add_argument("--reject-model-readbacks", action="store_true")
    parser.add_argument("--forward-api-dir", type=Path)
    parser.add_argument("--forward-api-start", type=int, default=0)
    parser.add_argument("--forward-api-count", type=int, default=9)
    options, remaining = parser.parse_known_args()
    if options.census_forwards < 0:
        parser.error("--census-forwards must be nonnegative")
    if options.forward_api_start < 0 or options.forward_api_count < 1:
        parser.error("forward API start/count must be nonnegative/positive")
    census = Census(options.census_forwards)
    census.api_directory = options.forward_api_dir
    census.api_start, census.api_count = (
        options.forward_api_start,
        options.forward_api_count,
    )
    original_load = AutoModelForCausalLM.from_pretrained
    original_cpu = torch.Tensor.cpu

    def load(cls, *args, **kwargs):
        model = original_load(*args, **kwargs)
        model.register_forward_pre_hook(census.begin)
        model.register_forward_hook(census.end, always_call=True)
        return model

    def cpu(value, *args, **kwargs):
        if census.in_forward and value.device.type != "cpu":
            census.readbacks.append(describe(value))
            if options.reject_model_readbacks:
                raise RuntimeError("Python tensor CPU readback inside model forward")
        return original_cpu(value, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = classmethod(load)
    torch.Tensor.cpu = cpu
    sys.argv = [sys.argv[0], *remaining]
    try:
        with census if options.census_forwards else nullcontext():
            runpy.run_path(
                str(Path(__file__).with_name("qwen_strict_eager.py")),
                run_name="__main__",
            )
    finally:
        torch.Tensor.cpu = original_cpu
        options.census_output.parent.mkdir(parents=True, exist_ok=True)
        options.census_output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "diagnostic_only": True,
                    "completed_forwards": census.forwards,
                    "recorded_forward_limit": census.limit,
                    "python_model_readbacks": census.readbacks,
                    "forward_api_files": census.api_files,
                    "records": [
                        {**json.loads(key), "calls": count}
                        for key, count in census.counts.items()
                    ],
                },
                indent=2,
            )
            + "\n"
        )
