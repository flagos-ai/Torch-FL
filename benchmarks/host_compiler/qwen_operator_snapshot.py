# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture first-attention ATen inputs/outputs for numerical diagnosis only.

The copies synchronize individual operators and intentionally perturb execution.
Use qwen_strict_eager.py without this wrapper for performance acceptance.
"""

import argparse
from pathlib import Path
import runpy
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from transformers import AutoModelForCausalLM


class Capture(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.done = False
        self.records = []

    def begin(self, module, inputs):
        self.active = not self.done

    def end(self, module, inputs, output):
        self.active = False
        self.done = True

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = function(*args, **kwargs)
        if self.active and function._schema.name in {
            "aten::mm",
            "aten::bmm",
            "aten::mul",
            "aten::add",
            "aten::neg",
            "aten::_softmax",
            "aten::_to_copy",
            "aten::mean",
            "aten::pow",
            "aten::rsqrt",
        }:

            def snapshot(value):
                return (
                    value.detach().cpu().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )

            self.records.append(
                {
                    "op": str(function),
                    "args": tree_map(snapshot, args),
                    "kwargs": tree_map(snapshot, kwargs),
                    "result": tree_map(snapshot, result),
                }
            )
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--operator-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    if (
        options.operator_snapshot.resolve()
        == options.output.with_suffix(".pt").resolve()
    ):
        parser.error("operator snapshots and benchmark tensors need separate paths")
    capture = Capture()
    original = AutoModelForCausalLM.from_pretrained

    def load(cls, *args, **kwargs):
        model = original(*args, **kwargs)
        module = model.model.layers[0].self_attn
        module.register_forward_pre_hook(capture.begin)
        module.register_forward_hook(capture.end)
        return model

    AutoModelForCausalLM.from_pretrained = classmethod(load)
    sys.argv = [sys.argv[0], *remaining, "--output", str(options.output)]
    try:
        with capture:
            runpy.run_path(
                str(Path(__file__).with_name("qwen_strict_eager.py")),
                run_name="__main__",
            )
    finally:
        options.operator_snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.save(capture.records, options.operator_snapshot)
