# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Diagnose individual FlagGems substitutions in the complete eager model.

The dispatch mode is diagnostic instrumentation. Final acceptance must use the
generated conf and qwen_strict_eager.py without this mode. One loaded checkpoint
is reused to avoid confounding every candidate with model-loading overhead.
"""

import argparse
from collections import Counter
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from transformers import AutoModelForCausalLM

from logits_validation import (
    file_sha256,
    math_settings,
    summarize_accuracy,
    tensor_accuracy,
)


CANDIDATES = {
    "mul.Tensor": "mul",
    "add.Tensor": "add",
    "neg": "neg",
    "pow.Tensor_Scalar": "pow_tensor_scalar",
    "cos": "cos",
    "sin": "sin",
    "rsqrt": "rsqrt",
    "cat": "cat",
    "where.self": "where_self",
    "mean.dim": "mean_dim",
    "silu": "silu",
    "_softmax": "softmax",
    "bmm": "bmm",
    "mm": "mm",
}
torch_fl = None


def op_name(function):
    name = function._schema.name.removeprefix("aten::")
    overload = function._schema.overload_name
    return name + ("." + overload if overload else "")


class Substitute(TorchDispatchMode):
    def __init__(self, selected, functions):
        super().__init__()
        self.selected = selected
        self.functions = functions
        self.calls = Counter()
        self.in_forward = False
        self.readbacks = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = op_name(function)
        values, _ = tree_flatten((args, kwargs))
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if name in self.selected and all(
            value.dtype not in (torch.int64, torch.uint64, torch.float64)
            for value in tensors
        ):
            self.calls[name] += 1
            return self.functions[name](*args, **kwargs)
        return function(*args, **kwargs)


def run_case(model, ids, inputs, reference, selected, launcher, functions):
    mode = Substitute(selected, functions)
    original_cpu = torch.Tensor.cpu

    def cpu(value, *args, **kwargs):
        if mode.in_forward and value.device.type != "cpu":
            mode.readbacks.append(
                {"shape": list(value.shape), "dtype": str(value.dtype)}
            )
            raise RuntimeError("FlagGems attempted a Python tensor CPU readback")
        return original_cpu(value, *args, **kwargs)

    before = launcher.enqueue_stats()
    torch.Tensor.cpu = cpu
    comparisons = []
    started = time.perf_counter()
    try:
        with torch.no_grad():
            cache = None
            for step, token in enumerate([ids, *inputs]):
                mode.in_forward = True
                try:
                    with mode:
                        out = model(token, past_key_values=cache, use_cache=True)
                        torch_fl.flagos.synchronize()
                finally:
                    mode.in_forward = False
                cache = out.past_key_values
                actual = out.logits.cpu() if step == 0 else out.logits[:, -1, :].cpu()
                expected = (
                    reference["prefill_logits"]
                    if step == 0
                    else reference["logits"][step]
                )
                comparisons.append(tensor_accuracy(actual, expected))
        result = {
            "validation": summarize_accuracy(
                comparisons, len(inputs) + 1, atol=1e-3, rtol=1e-3
            )
        }
    except Exception as error:
        result = {"error": type(error).__name__ + ": " + str(error)}
    finally:
        torch.Tensor.cpu = original_cpu
    after = launcher.enqueue_stats()
    result.update(
        selected=sorted(selected),
        operator_calls=dict(mode.calls),
        python_tensor_readbacks=mode.readbacks,
        comparisons=comparisons,
        elapsed_seconds=time.perf_counter() - started,
        enqueued_kernels={
            name: count - before.get(name, 0)
            for name, count in after.items()
            if count != before.get(name, 0)
        },
    )
    result["flag_gems_executed"] = bool(mode.calls) and bool(result["enqueued_kernels"])
    torch_fl.flagos.synchronize()
    return result


def main():
    global torch_fl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="*", choices=list(CANDIDATES), default=list(CANDIDATES)
    )
    parser.add_argument(
        "--source-profile", choices=("vendor", "generic"), default="vendor"
    )
    parser.add_argument("--square-via-mul", action="store_true")
    args = parser.parse_args()
    if "torch_fl" in sys.modules:
        raise RuntimeError(
            "run substitutions in a fresh process before importing torch_fl"
        )
    installed = Path(importlib.util.find_spec("torch_fl").origin).parent
    baseline_conf = args.output.with_suffix(".baseline.conf").resolve()
    baseline_conf.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in (installed / "configs/backends_enflame.conf").read_text().splitlines():
        name, sep, _ = line.split("#", 1)[0].partition("=")
        lines.append(
            f"{name.strip()} = enflame" if sep and name.strip() in CANDIDATES else line
        )
    baseline_conf.write_text("\n".join(lines) + "\n")
    os.environ["FLAGOS_BACKEND_CONFIG"] = str(baseline_conf)
    import torch_fl
    import flag_gems

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    assert reference["producer"]["backend"] == "vendor"
    assert reference["contract"]["math_settings"] == math_settings()
    assert (
        file_sha256(args.model / "config.json")
        == reference["contract"]["config_sha256"]
    )
    for name, digest in reference["contract"]["checkpoint_sha256"].items():
        assert file_sha256(args.model / name) == digest
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .eval()
        .to("flagos:0")
    )
    ids = reference["prompt_ids"].to("flagos:0")
    inputs = [token.to("flagos:0") for token in reference["inputs"]]
    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )
    hosts = torch_fl.native_host._hosts
    functions, unavailable = {}, {}
    for name, function in CANDIDATES.items():
        if args.source_profile == "vendor":
            functions[name] = getattr(flag_gems, function)
            continue
        module_name = "flag_gems.ops." + name.split(".")[0].lstrip("_")
        module = hosts.originals.get(module_name)
        if module is not None and hasattr(module, function):
            functions[name] = hosts.resolve(getattr(module, function))
        else:
            unavailable[name] = (
                "Generic published module is absent from the prebuilt host closure: "
                + module_name
            )
    if args.square_via_mul:
        mul = functions["mul.Tensor"]

        def square(value, exponent):
            if exponent != 2:
                raise ValueError("square diagnostic expects exponent two")
            return mul(value, value)

        functions["pow.Tensor_Scalar"] = square
        unavailable.pop("pow.Tensor_Scalar", None)
    report = {
        "schema_version": 1,
        "diagnostic_only": True,
        "source_profile": args.source_profile,
        "square_via_mul": args.square_via_mul,
        "unavailable": unavailable,
        "reference_sha256": file_sha256(args.reference),
        "baseline_config": {
            "path": str(baseline_conf),
            "sha256": file_sha256(baseline_conf),
        },
        "layers": len(model.model.layers),
        "parameters": sum(p.numel() for p in model.parameters()),
        "contract": reference["contract"],
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def case(name, selected):
        print(json.dumps({"starting": name, "selected": sorted(selected)}), flush=True)
        result = run_case(model, ids, inputs, reference, selected, launcher, functions)
        report["cases"][name] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "finished": name,
                    "passed": result.get("validation", {}).get("passed"),
                    "violations": result.get("validation", {}).get("violations"),
                    "submissions": sum(result["enqueued_kernels"].values()),
                    "elapsed_seconds": result["elapsed_seconds"],
                    "error": result.get("error"),
                }
            ),
            flush=True,
        )
        gc.collect()
        return result

    baseline = case("baseline", set())
    if not baseline.get("validation", {}).get("passed"):
        raise RuntimeError("baseline does not reproduce the saved vendor reference")
    if baseline["enqueued_kernels"]:
        raise RuntimeError(
            "the vendor control submitted FlagGems kernels; substitutions are confounded"
        )
    passing = []
    for name in args.cases:
        if name in unavailable:
            continue
        result = case(name, {name})
        if result.get("validation", {}).get("passed") and result["flag_gems_executed"]:
            passing.append(name)
    if passing:
        combined = case("combined", set(passing))
        if not combined.get("validation", {}).get("passed"):
            selected = set()
            for name in passing:
                result = case("greedy:" + name, selected | {name})
                if result.get("validation", {}).get("passed"):
                    selected.add(name)
            report["greedy_selected"] = sorted(selected)
    report["individually_passing"] = passing
    report["admitted_kernel_sources"] = torch_fl.native_host.audit()["launched_kernels"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
