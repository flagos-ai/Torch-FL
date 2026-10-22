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

"""Micro-benchmark: argmax latency on vocab-sized logits.

Usage:
    # torch_fl
    CUDA_VISIBLE_DEVICES=2 FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
        python tests/perf/bench_argmax.py

    # CUDA baseline
    CUDA_VISIBLE_DEVICES=2 python tests/perf/bench_argmax.py --baseline
"""

import argparse
import time

import torch


def bench_argmax(device: torch.device, warmup: int = 50, rounds: int = 200):
    # Simulate logits shape: (1, vocab_size) typical for greedy decoding
    vocab_size = 152064  # Qwen3 vocab
    x = torch.randn(1, vocab_size, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        torch.argmax(x, dim=-1)
    torch.cuda.synchronize(device)

    # Benchmark
    times = []
    for _ in range(rounds):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        torch.argmax(x, dim=-1)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    print(f"argmax({1}x{vocab_size}) on {device}")
    print(f"  Median: {median:.3f} ms")
    print(f"  P10:    {p10:.3f} ms")
    print(f"  P90:    {p90:.3f} ms")
    print(f"  Min:    {times[0]:.3f} ms")
    print(f"  Max:    {times[-1]:.3f} ms")
    return median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Use native CUDA")
    args = parser.parse_args()

    if args.baseline:
        device = torch.device("cuda:0")
        print("=== CUDA Baseline ===")
    else:
        import torch_fl  # noqa: F401

        device = torch.device("flagos:0")
        print("=== torch_fl (flagos) ===")

    bench_argmax(device)


if __name__ == "__main__":
    main()
