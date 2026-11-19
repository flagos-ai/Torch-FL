# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Paired eager Hygon operator correctness and warm submission measurements."""

# Device libraries must precede Torch import on the CPU-Torch Hygon build.
# ruff: noqa: E402

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

_backend_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_backend_parser.add_argument("--device")
if (_backend_parser.parse_known_args()[0].device or "").startswith("flagos"):
    import torch_fl  # noqa: F401

import torch

from model_operator_submission import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda:0", "flagos:0"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    native = args.device.startswith("flagos")
    native_scope = nullcontext()
    report = {
        "torch": torch.__version__,
        "torch_distribution": importlib.metadata.version("torch"),
        "device": args.device,
        "execution": "eager",
        "numerical_tolerance": {"rtol": 0.03, "atol": 0.05},
        "cases": {},
    }
    if native:
        import torch_fl
        from torch_fl import _build_config

        assert _build_config.NATIVE_HOSTS and "boxing" not in _build_config.KERNELS
        report["kernels"] = list(_build_config.KERNELS)
        report["backend"] = torch_fl.native_host._hosts.manifest["backend"]
        synchronize = torch_fl.flagos.synchronize
        stats = torch.ops.flagos.native_source_typed_stats
        native_scope = torch_fl.native_host.execution_scope(0)
        report["native_execution_scope"] = "current DTK library stream"
    else:
        assert "torch_fl" not in sys.modules and torch.version.hip is not None
        synchronize = torch.cuda.synchronize
        stats = None
    reference = (
        torch.load(args.reference, weights_only=True) if args.reference else None
    )
    if native and (reference is None or reference["producer"] != "vendor"):
        raise ValueError(
            "candidate correctness requires an independent vendor reference"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    with torch.no_grad(), native_scope:
        x = torch.randn(23, 1024, dtype=torch.bfloat16).to(args.device)
        weight = torch.randn(3072, 1024, dtype=torch.bfloat16).to(args.device).t()
        activation = torch.randn(1, 23, 3072, dtype=torch.bfloat16).to(args.device)
        norm = torch.randn(1, 1, 1024, dtype=torch.bfloat16).to(args.device)
        scores = torch.randn(1, 16, 1, 24).to(args.device)
        query = torch.randn(16, 1, 128, dtype=torch.bfloat16).to(args.device)
        keys = torch.randn(16, 128, 481, dtype=torch.bfloat16).to(args.device)
        probabilities = torch.randn(16, 1, 481, dtype=torch.bfloat16).to(args.device)
        values = torch.randn(16, 481, 128, dtype=torch.bfloat16).to(args.device)
        add_out, mul_out = torch.empty_like(x), torch.empty_like(x)
        mean_out = torch.empty(1, 16, 1, dtype=torch.float32, device=args.device)
        cat_out = torch.empty(46, 1024, dtype=x.dtype, device=args.device)
        operations = {
            "cos_fp32": lambda: torch.cos(scores),
            "mm_bf16_23_1024_3072": lambda: torch.mm(x, weight),
            "mul_bf16_1_23_3072": lambda: activation * activation,
            "mul_fp32_1_16_1_24": lambda: scores * scores,
            "mul_scalar_fp32_1_16_1_24": lambda: scores * 0.08838834764831845,
            "pow_square_fp32_1_16_1_24": lambda: scores.pow(2),
            "bmm_bf16_16_1_128_481": lambda: torch.bmm(query, keys),
            "bmm_bf16_16_1_481_128": lambda: torch.bmm(probabilities, values),
            "add_bf16_23_1024": lambda: x + x,
            "mean_fp32_1_16_1_24": lambda: scores.mean(-1),
            "cat_bf16_46_1024": lambda: torch.cat([x, x], dim=0),
            "add_out_bf16_23_1024": lambda: torch.add(x, x, out=add_out),
            "mul_out_bf16_23_1024": lambda: torch.mul(x, x, out=mul_out),
            "mean_out_fp32_1_16_1_24": lambda: torch.mean(scores, [-1], out=mean_out),
            "cat_out_bf16_46_1024": lambda: torch.cat([x, x], dim=0, out=cat_out),
            "silu_bf16_1_23_3072": lambda: torch.nn.functional.silu(activation),
            "softmax_fp32_1_16_1_24": lambda: torch.softmax(scores, -1),
            "cast_bf16_to_fp32_1_1_1024": lambda: norm.float(),
            "arange_int64_24": lambda: torch.arange(24, device=args.device),
            "scalar_tensor_fp32": lambda: torch.scalar_tensor(0.0, device=args.device),
        }
        input_hashes = {
            name: hashlib.sha256(
                value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, value in {
                "x": x,
                "weight": weight,
                "activation": activation,
                "norm": norm,
                "scores": scores,
                "query": query,
                "keys": keys,
                "probabilities": probabilities,
                "values": values,
            }.items()
        }
        report["input_sha256"] = input_hashes
        if reference and input_hashes != reference["input_sha256"]:
            raise ValueError("candidate and vendor operator inputs differ")
        for name, call in operations.items():
            try:
                actual = call()
                synchronize()
                assert str(actual.device) == args.device
                saved[name] = actual.cpu()
                row = {"status": "passed", "output_device": str(actual.device)}
                if reference:
                    expected = reference["outputs"][name]
                    torch.testing.assert_close(
                        saved[name], expected, rtol=0.03, atol=0.05
                    )
                    row["max_abs_error"] = float(
                        (saved[name].float() - expected.float()).abs().max()
                    )
                    row["bitwise_equal"] = torch.equal(saved[name], expected)
                before = stats()[1] if stats else None
                row.update(measure(call, synchronize))
                if stats:
                    row["typed_entry_hits"] = stats()[1] - before
            except Exception as error:
                row = {
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            report["cases"][name] = row
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(name, json.dumps(row), flush=True)
    if native:
        report["native_audit"] = torch_fl.native_host.audit()
    report["passed"] = all(
        row["status"] == "passed" for row in report["cases"].values()
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(
        {
            "producer": "native" if native else "vendor",
            "outputs": saved,
            "input_sha256": input_hashes,
        },
        args.output.with_suffix(".pt"),
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
