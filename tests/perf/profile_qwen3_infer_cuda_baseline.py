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
Profile Qwen3 inference on native CUDA to provide op-level baseline.

Same profiling methodology as profile_qwen3_infer.py but runs on CUDA.
Compare outputs to identify which ops are slow in torch_fl.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tests/manual/profile_qwen3_infer_cuda_baseline.py
"""

import argparse
import time
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer


class OpProfiler(TorchDispatchMode):
    """Profile op calls and execution time."""

    def __init__(self, device):
        self.device = device
        self.op_counts = defaultdict(int)
        self.op_times = defaultdict(float)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)
        self.op_counts[op_name] += 1

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0
        self.op_times[op_name] += elapsed

        return result

    def report(self, total_tokens):
        print("\n=== Op Profile Summary (CUDA baseline) ===")
        print(f"Total unique ops: {len(self.op_counts)}")
        print(f"Total op calls: {sum(self.op_counts.values())}")
        print(f"Total op time: {sum(self.op_times.values()):.3f}s")

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


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()
    device = args.device

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()
    model.model.layers[0].self_attn.config._attn_implementation = "eager"
    print("Model loaded, attention: eager")

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

    # Warmup
    print("Warmup...")
    for i in range(args.warmup_rounds):
        torch.cuda.synchronize(device)
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**gen_kwargs)
        torch.cuda.synchronize(device)
        print(f"  Round {i + 1}: {time.time() - t0:.2f}s")
    print()

    # Profiled runs
    print(f"Profiling ({args.rounds} rounds)...")
    round_times = []
    round_tps = []

    profiler = OpProfiler(device)
    total_tokens_generated = 0

    for i in range(args.rounds):
        torch.cuda.synchronize(device)
        t0 = time.time()
        with profiler:
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
        torch.cuda.synchronize(device)
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

    # Report
    profiler.report(total_tokens_generated)

    # Per-token breakdown
    print("\n=== Per-Token Breakdown (median round) ===")
    total_op_time = sum(profiler.op_times.values())
    total_wall = sum(round_times)
    tokens_per_round = args.tokens
    ops_per_round = sum(profiler.op_counts.values()) / args.rounds
    op_time_per_round = total_op_time / args.rounds
    unaccounted = total_wall - total_op_time

    print(f"Total wall-clock:    {total_wall:.3f}s ({args.rounds} rounds)")
    print(
        f"Op execution time:   {total_op_time:.3f}s ({total_op_time / total_wall * 100:.1f}%)"
    )
    print(
        f"Unaccounted overhead: {unaccounted:.3f}s ({unaccounted / total_wall * 100:.1f}%)"
    )
    print(f"Time per token: {median_time / tokens_per_round * 1000:.1f}ms")
    print(f"Ops per token:  {ops_per_round / tokens_per_round:.0f}")
    avg_op_time_us = (op_time_per_round / ops_per_round) * 1e6
    print(f"Avg op time:    {avg_op_time_us:.1f}µs")


if __name__ == "__main__":
    main()
