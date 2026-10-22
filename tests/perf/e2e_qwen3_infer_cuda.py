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
End-to-end Qwen3 inference benchmark on torch_fl (no per-op profiling).

Measures pure wall-clock latency and throughput without TorchDispatchMode
overhead, providing a fair e2e comparison with native CUDA.

Usage:
    python tests/perf/e2e_qwen3_infer_cuda.py [--model PATH] [--tokens N] [--rounds N]
"""

import argparse
import time

import torch
import torch_fl
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    print(f"Warmup rounds: {args.warmup_rounds}, Benchmark rounds: {args.rounds}")
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
        torch_fl.flagos.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**gen_kwargs)
        torch_fl.flagos.synchronize()
        print(f"  Round {i + 1}: {time.perf_counter() - t0:.3f}s")
    print()

    # Benchmark runs
    print(f"Benchmarking ({args.rounds} rounds)...")
    round_times = []

    for i in range(args.rounds):
        torch_fl.flagos.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**gen_kwargs)
        torch_fl.flagos.synchronize()
        elapsed = time.perf_counter() - t0

        new_tokens = output.shape[1] - input_len
        tps = new_tokens / elapsed
        round_times.append(elapsed)
        print(f"  Round {i + 1}: {elapsed:.3f}s, {new_tokens} tokens, {tps:.2f} tok/s")

    # Statistics
    round_times.sort()
    median_time = round_times[len(round_times) // 2]
    min_time = round_times[0]
    max_time = round_times[-1]
    median_tps = args.tokens / median_time

    print("\n=== E2E Benchmark Results (torch_fl) ===")
    print(f"Tokens generated: {args.tokens} (greedy, fixed)")
    print(f"Median: {median_time:.3f}s ({median_tps:.2f} tok/s)")
    print(f"Min:    {min_time:.3f}s ({args.tokens / min_time:.2f} tok/s)")
    print(f"Max:    {max_time:.3f}s ({args.tokens / max_time:.2f} tok/s)")
    print(f"Spread: {(max_time - min_time) / median_time * 100:.1f}%")
    print(f"Time per token: {median_time / args.tokens * 1000:.2f}ms")


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2E Qwen3 inference benchmark on torch_fl"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Path to model")
    parser.add_argument(
        "--tokens", type=int, default=64, help="Exact number of new tokens to generate"
    )
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of benchmark rounds (take median)"
    )
    parser.add_argument(
        "--warmup-rounds", type=int, default=3, help="Number of warmup rounds"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
