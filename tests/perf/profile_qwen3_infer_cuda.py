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

"""
Profile Qwen3 inference on torch_fl to diagnose performance bottlenecks.

Measures:
1. Op-level dispatch overhead (TorchDispatchMode hook)
2. Per-op execution time (top 20 slowest ops)
3. CPU fallback detection (ops going to CPU)
4. Synchronization overhead

Usage:
    python tests/manual/profile_qwen3_infer.py [--model PATH] [--tokens N] [--rounds N]
"""

import argparse
import time
from collections import defaultdict

import torch
import torch_fl
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer


class OpProfiler(TorchDispatchMode):
    """Profile op calls, execution time, and device placement."""

    def __init__(self):
        self.op_counts = defaultdict(int)
        self.op_times = defaultdict(float)  # total time per op
        self.cpu_fallbacks = defaultdict(int)  # ops that returned CPU tensors

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)

        # Track call count
        self.op_counts[op_name] += 1

        # Time the op execution
        torch_fl.flagos.synchronize()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        torch_fl.flagos.synchronize()
        elapsed = time.perf_counter() - t0
        self.op_times[op_name] += elapsed

        # Detect CPU fallback
        if isinstance(result, torch.Tensor) and result.device.type == "cpu":
            self.cpu_fallbacks[op_name] += 1
        elif isinstance(result, (list, tuple)):
            for r in result:
                if isinstance(r, torch.Tensor) and r.device.type == "cpu":
                    self.cpu_fallbacks[op_name] += 1
                    break

        return result

    def report(self, total_tokens):
        print("\n=== Op Profile Summary ===")
        print(f"Total unique ops: {len(self.op_counts)}")
        print(f"Total op calls: {sum(self.op_counts.values())}")
        print(f"Total op time: {sum(self.op_times.values()):.3f}s")

        if self.cpu_fallbacks:
            print(f"\n⚠️  CPU fallbacks detected: {len(self.cpu_fallbacks)} ops")
            print("\nTop CPU fallback ops:")
            for op, count in sorted(
                self.cpu_fallbacks.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                print(f"  {op:60s} {count:5d} calls")

        print("\n=== Top 20 slowest ops (by total time) ===")
        for op, total_time in sorted(
            self.op_times.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            count = self.op_counts[op]
            avg_time_us = (total_time / count) * 1e6
            pct = (total_time / sum(self.op_times.values())) * 100
            print(
                f"{op:50s} {total_time:6.3f}s ({pct:5.1f}%)  "
                f"{count:5d} calls  {avg_time_us:7.1f}µs/call"
            )

        print("\n=== Top 20 most frequent ops ===")
        for op, count in sorted(
            self.op_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            total_time = self.op_times[op]
            avg_time_us = (total_time / count) * 1e6
            print(f"{op:50s} {count:5d} calls  {avg_time_us:7.1f}µs/call")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Qwen3 inference on torch_fl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Path to model")
    parser.add_argument(
        "--tokens", type=int, default=64, help="Exact number of new tokens to generate"
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of profiling rounds (take median)"
    )
    parser.add_argument(
        "--warmup-rounds", type=int, default=2, help="Number of warmup rounds"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "flagos:0"

    print(f"Device: {device}")
    print(f"Device count: {torch_fl.flagos.device_count()}")
    print(f"FlagGems enabled: {torch_fl.is_flaggems_enabled()}")
    print(f"Registered ops: {len(torch_fl.get_registered_ops())}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()
    # Force eager attention to match baseline
    model.model.layers[0].self_attn.config._attn_implementation = "eager"
    print(f"Model loaded in {time.time() - t0:.2f}s")
    print("Attention: eager")
    print()

    # Prepare input
    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Give me a short introduction to large language model.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_len}")
    print(f"Output tokens: {args.tokens} (fixed, greedy)")
    print(f"Warmup rounds: {args.warmup_rounds}, Profiling rounds: {args.rounds}")
    print()

    # Generation config: greedy, deterministic, fixed length
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.tokens,
        min_new_tokens=args.tokens,  # force exact token count
        do_sample=False,  # greedy decoding
        temperature=None,
        top_p=None,
        top_k=None,
    )

    # Warmup runs (no profiling)
    print("Warmup...")
    for i in range(args.warmup_rounds):
        torch_fl.flagos.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**gen_kwargs)
        torch_fl.flagos.synchronize()
        print(f"  Round {i + 1}: {time.time() - t0:.2f}s")
    print()

    # Profiled runs - collect timing from multiple rounds
    print(f"Profiling ({args.rounds} rounds)...")
    round_times = []
    round_tps = []

    # Use a single profiler that accumulates across all rounds
    profiler = OpProfiler()
    total_tokens_generated = 0

    for i in range(args.rounds):
        torch_fl.flagos.synchronize()
        t0 = time.time()
        with profiler:
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
        torch_fl.flagos.synchronize()
        elapsed = time.time() - t0

        new_tokens = output.shape[1] - input_len
        total_tokens_generated += new_tokens
        tps = new_tokens / elapsed
        round_times.append(elapsed)
        round_tps.append(tps)
        print(f"  Round {i + 1}: {elapsed:.2f}s, {new_tokens} tokens, {tps:.2f} tok/s")

    # Statistics
    round_times.sort()
    round_tps.sort()
    median_time = round_times[len(round_times) // 2]
    median_tps = round_tps[len(round_tps) // 2]
    min_time = round_times[0]
    max_time = round_times[-1]

    print(f"\n=== Timing Summary ({args.rounds} rounds) ===")
    print(f"Median: {median_time:.3f}s ({median_tps:.2f} tok/s)")
    print(f"Min:    {min_time:.3f}s ({round_tps[-1]:.2f} tok/s)")
    print(f"Max:    {max_time:.3f}s ({round_tps[0]:.2f} tok/s)")
    print(f"Spread: {(max_time - min_time) / median_time * 100:.1f}%")

    # Report aggregated profiler results
    profiler.report(total_tokens_generated)

    # Overhead analysis (averaged across rounds)
    print("\n=== Overhead Analysis (aggregated) ===")
    total_wall = sum(round_times)
    total_op_time = sum(profiler.op_times.values())
    unaccounted = total_wall - total_op_time
    print(f"Total wall-clock:    {total_wall:.3f}s ({args.rounds} rounds)")
    print(
        f"Op execution time:   {total_op_time:.3f}s ({total_op_time / total_wall * 100:.1f}%)"
    )
    print(
        f"Unaccounted overhead: {unaccounted:.3f}s ({unaccounted / total_wall * 100:.1f}%)"
    )
    print("  (dispatch, Python overhead, host-device sync, framework bookkeeping)")

    # Per-token breakdown (use median round)
    print("\n=== Per-Token Breakdown (median round) ===")
    tokens_per_round = args.tokens
    ops_per_round = sum(profiler.op_counts.values()) / args.rounds
    op_time_per_round = total_op_time / args.rounds
    print(f"Time per token: {median_time / tokens_per_round * 1000:.1f}ms")
    print(f"Ops per token:  {ops_per_round / tokens_per_round:.0f}")
    avg_op_time_us = (op_time_per_round / ops_per_round) * 1e6
    print(f"Avg op time:    {avg_op_time_us:.1f}µs")


if __name__ == "__main__":
    main()
