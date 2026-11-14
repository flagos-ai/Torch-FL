# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare ordinary eager submission on representative Qwen3 operator shapes."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def measure(call, synchronize):
    for _ in range(30):
        call()
    synchronize()
    rounds = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(200):
            output = call()
        rounds.append((time.perf_counter_ns() - start) / 200 / 1000)
        synchronize()
        del output
    return {"median_us": statistics.median(rounds), "rounds_us": rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("flagos"):
        import torch_fl
        from torch_fl import host_compiler

        synchronize = torch_fl.flagos.synchronize
        if args.cache:
            host_compiler.enable(args.cache, mode="load")
    else:
        synchronize = torch.cuda.synchronize
    with torch.no_grad():
        x = torch.randn(23, 1024, dtype=torch.bfloat16).to(args.device)
        weight = torch.randn(3072, 1024, dtype=torch.bfloat16).to(args.device).t()
        norm = torch.randn(1, 1, 1024, dtype=torch.bfloat16).to(args.device)
        activation = torch.randn(1, 23, 3072, dtype=torch.bfloat16).to(args.device)
        scores = torch.randn(1, 16, 1, 24).to(args.device)
        operations = {
            "mm_bf16_23_1024_3072": lambda: torch.mm(x, weight),
            "cast_bf16_to_fp32_1_1_1024": lambda: norm.float(),
            "silu_bf16_1_23_3072": lambda: torch.nn.functional.silu(activation),
            "softmax_fp32_1_16_1_24": lambda: torch.softmax(scores, -1),
            "arange_int64_24": lambda: torch.arange(24, device=args.device),
            "scalar_tensor_fp32": lambda: torch.scalar_tensor(0.0, device=args.device),
        }
        report = {"torch": torch.__version__, "device": args.device, "cases": {}}
        for name, call in operations.items():
            before = torch.ops.flagos.host_program_stats()[2] if args.cache else None
            row = measure(call, synchronize)
            if before is not None:
                row["native_hits"] = torch.ops.flagos.host_program_stats()[2] - before
            report["cases"][name] = row
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
