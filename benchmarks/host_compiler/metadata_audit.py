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

"""Measure real metadata dispatch; never simulate fallback with host copies.

Run in separate processes with --device flagos:0 and --device cuda:0.
The flagos process requires the CPU-torch, non-boxing wheel and uses its
unmodified default configuration. JSON retains correctness and alias results
alongside timing, so a fast but invalid view cannot count as a success.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
import warnings
from pathlib import Path

import torch


def measure(operation, synchronize, iterations, rounds):
    for _ in range(10):
        operation()
    synchronize()
    submission, completion = [], []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            result = operation()
        submitted = time.perf_counter_ns()
        synchronize()
        completed = time.perf_counter_ns()
        submission.append((submitted - start) / iterations / 1000)
        completion.append((completed - start) / iterations / 1000)
        del result
    return {
        "submission_us_per_call": submission,
        "batch_completion_us_per_call": completion,
        "submission_median_us": statistics.median(submission),
        "batch_completion_median_us": statistics.median(completion),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["flagos:0", "cuda:0"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    assert args.iterations > 0 and args.rounds > 0

    environment = {"torch": torch.__version__, "device": args.device}
    if args.device.startswith("flagos"):
        import torch_fl
        from torch_fl import _build_config

        assert torch.version.cuda is None, "measure the CPU-torch contract"
        assert "boxing" not in _build_config.KERNELS, "requires a non-boxing wheel"
        config = Path(torch_fl.backend_config_path())
        assert config.name == "backends_nvidia.conf"
        environment.update(
            kernels=list(_build_config.KERNELS),
            config=str(config),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            flag_gems=importlib.metadata.version("flag_gems"),
            flagtree=importlib.metadata.version("flagtree"),
        )
        synchronize = torch_fl.flagos.synchronize
    else:
        synchronize = torch.cuda.synchronize

    # Inputs are constructed on CPU outside timing, then transferred once.
    cases = [
        ("transpose.int", (32, 1024, 128), lambda x: x.transpose(0, 1)),
        ("t", (4096, 4096), lambda x: x.t()),
        ("unsqueeze", (32, 1024, 128), lambda x: x.unsqueeze(0)),
        ("select.int", (1, 1024, 4096), lambda x: x.select(1, 0)),
        ("detach", (32, 1024, 128), lambda x: x.detach()),
        (
            "_unsafe_view",
            (32, 1024, 128),
            lambda x: torch.ops.aten._unsafe_view(x, [-1]),
        ),
    ]
    report = {"environment": environment, "parameters": vars(args).copy(), "cases": []}
    report["parameters"]["output"] = str(args.output)
    for name, shape, operation in cases:
        host = torch.ones(shape, dtype=torch.float32)
        value = host.to(args.device)
        synchronize()
        row = {"op": name, "shape": shape}
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                result = operation(value)
                expected = operation(host)
                row.update(
                    device=str(result.device),
                    aliases_input=(
                        result.untyped_storage().data_ptr()
                        == value.untyped_storage().data_ptr()
                    ),
                    strides_match=result.stride() == expected.stride(),
                    offset_matches=result.storage_offset() == expected.storage_offset(),
                )
                torch.testing.assert_close(result.cpu(), expected)
                row["values_match"] = True
                row.update(
                    measure(
                        lambda: operation(value),
                        synchronize,
                        args.iterations,
                        args.rounds,
                    )
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["warnings"] = sorted({str(w.message) for w in recorded})
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
