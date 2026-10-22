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
Count actual tensor allocations per token during inference.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--tokens", type=int, default=64, help="Number of tokens to generate"
    )
    args = parser.parse_args()

    model_name = args.model
    num_tokens = args.tokens

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

    # Count allocations via torch.cuda.memory_stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    stats_before = torch.cuda.memory_stats()

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

    torch.cuda.synchronize()
    stats_after = torch.cuda.memory_stats()

    total_allocs = (
        stats_after["allocation.all.allocated"]
        - stats_before["allocation.all.allocated"]
    )
    total_frees = (
        stats_after["allocation.all.freed"] - stats_before["allocation.all.freed"]
    )

    print(
        f"=== Allocation count during {num_tokens}-token generation ({model_name}) ==="
    )
    print(f"Total allocations: {total_allocs}")
    print(f"Total frees:       {total_frees}")
    print(f"Allocs per token:  {total_allocs / num_tokens:.1f}")
    print(f"Frees per token:   {total_frees / num_tokens:.1f}")
    print()
    print("Per-token allocator overhead estimate:")
    print(
        f"  @ 506ns/alloc saving: {total_allocs / num_tokens * 506 / 1e6:.3f} ms/token"
    )


if __name__ == "__main__":
    main()
