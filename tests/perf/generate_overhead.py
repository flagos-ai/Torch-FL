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
Isolate HuggingFace generate() framework overhead for PrivateUse1 devices.

Previous findings:
- Single forward pass: flagos only +0.44ms (2.1%) vs CUDA
- generate(1 token):   flagos +5.08ms (22.4%) vs CUDA
- Framework overhead:  +4.63ms/step comes from generate() internals

This script breaks down generate() into components to find the bottleneck:
1. Forward pass (model.__call__)
2. Logits processing (temperature, top_k, etc.)
3. Token selection (argmax for greedy)
4. Stopping criteria (EOS check)
5. KV cache update
6. Input preparation for next step

Usage:
    CUDA_VISIBLE_DEVICES=2 FLAGOS_BACKEND_CONFIG=... python tests/perf/generate_overhead.py
"""

import argparse
import time

import torch
import torch_fl
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteriaList,
)


def sync(device_type):
    if device_type == "flagos":
        torch_fl.flagos.synchronize()
    else:
        torch.cuda.synchronize()


def bench(fn, device_type, warmup=50, rounds=200):
    for _ in range(warmup):
        fn()
    sync(device_type)
    t0 = time.perf_counter()
    for _ in range(rounds):
        fn()
    sync(device_type)
    return (time.perf_counter() - t0) / rounds * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model path")
    parser.add_argument("-n", type=int, default=200, help="Rounds per test")
    args = parser.parse_args()
    N = args.n

    model_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load two copies
    model_cuda = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu"
        )
        .to("cuda:0")
        .eval()
    )
    model_cuda.model.layers[0].self_attn.config._attn_implementation = "eager"

    model_fl = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu"
        )
        .to("flagos:0")
        .eval()
    )
    model_fl.model.layers[0].self_attn.config._attn_implementation = "eager"

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
    input_ids_cuda = tokenizer([text], return_tensors="pt")["input_ids"].to("cuda:0")
    input_ids_fl = tokenizer([text], return_tensors="pt")["input_ids"].to("flagos:0")
    seq_len = input_ids_cuda.shape[1]

    print(f"Model: {model_path}")
    print(f"Input tokens: {seq_len}")
    print(f"Rounds per test: {N}")
    print()
    print(f"{'Test':<45s} {'CUDA':>8s} {'flagos':>8s} {'delta':>10s}")
    print("-" * 75)

    # === Test 1: Pure forward (prefill, no cache) ===
    def fwd_cuda():
        with torch.no_grad():
            model_cuda(input_ids_cuda)

    def fwd_fl():
        with torch.no_grad():
            model_fl(input_ids_fl)

    t_cuda = bench(fwd_cuda, "cuda", rounds=N)
    t_fl = bench(fwd_fl, "flagos", rounds=N)
    print(
        f"{'1. Forward pass (prefill, no cache)':<45s} {t_cuda:>6.2f}ms {t_fl:>6.2f}ms {t_fl - t_cuda:>+8.2f}ms"
    )

    # === Test 2: Forward with KV cache (decode step) ===
    # First do prefill to get cache
    with torch.no_grad():
        out_cuda = model_cuda(input_ids_cuda, use_cache=True)
        cache_cuda = out_cuda.past_key_values
        out_fl = model_fl(input_ids_fl, use_cache=True)
        cache_fl = out_fl.past_key_values

    # Single new token
    next_tok_cuda = out_cuda.logits[:, -1:].argmax(dim=-1)
    next_tok_fl = out_fl.logits[:, -1:].argmax(dim=-1)

    def decode_cuda():
        with torch.no_grad():
            model_cuda(next_tok_cuda, past_key_values=cache_cuda, use_cache=True)

    def decode_fl():
        with torch.no_grad():
            model_fl(next_tok_fl, past_key_values=cache_fl, use_cache=True)

    t_cuda = bench(decode_cuda, "cuda", rounds=N)
    t_fl = bench(decode_fl, "flagos", rounds=N)
    print(
        f"{'2. Decode step (1 token, with KV cache)':<45s} {t_cuda:>6.2f}ms {t_fl:>6.2f}ms {t_fl - t_cuda:>+8.2f}ms"
    )

    # === Test 3: argmax (greedy token selection) ===
    logits_cuda = out_cuda.logits[:, -1:]  # [1, 1, vocab_size]
    logits_fl = out_fl.logits[:, -1:]

    def argmax_cuda():
        logits_cuda.argmax(dim=-1)

    def argmax_fl():
        logits_fl.argmax(dim=-1)

    t_cuda = bench(argmax_cuda, "cuda", rounds=N * 5)
    t_fl = bench(argmax_fl, "flagos", rounds=N * 5)
    print(
        f"{'3. argmax (greedy selection, vocab=151k)':<45s} {t_cuda:>6.2f}ms {t_fl:>6.2f}ms {t_fl - t_cuda:>+8.2f}ms"
    )

    # === Test 4: Stopping criteria (EOS check via isin) ===
    eos_token_id = tokenizer.eos_token_id
    criteria_cuda = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=seq_len + 64),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    criteria_fl = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=seq_len + 64),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )

    # Simulate generated ids so far
    gen_ids_cuda = torch.cat([input_ids_cuda, next_tok_cuda], dim=1)
    gen_ids_fl = torch.cat([input_ids_fl, next_tok_fl], dim=1)
    scores_cuda = logits_cuda[:, -1, :]
    scores_fl = logits_fl[:, -1, :]

    def stopping_cuda():
        criteria_cuda(gen_ids_cuda, scores_cuda)

    def stopping_fl():
        criteria_fl(gen_ids_fl, scores_fl)

    t_cuda = bench(stopping_cuda, "cuda", rounds=N * 5)
    t_fl = bench(stopping_fl, "flagos", rounds=N * 5)
    print(
        f"{'4. Stopping criteria (EOS + MaxLength)':<45s} {t_cuda:>6.2f}ms {t_fl:>6.2f}ms {t_fl - t_cuda:>+8.2f}ms"
    )

    # === Test 5: isin alone ===
    eos_ids_cuda = torch.tensor([eos_token_id], device="cuda:0")
    eos_ids_fl = torch.tensor([eos_token_id], device="flagos:0")

    def isin_cuda():
        torch.isin(next_tok_cuda, eos_ids_cuda)

    def isin_fl():
        torch.isin(next_tok_fl, eos_ids_fl)

    t_cuda = bench(isin_cuda, "cuda", rounds=N * 5)
    t_fl = bench(isin_fl, "flagos", rounds=N * 5)
    print(
        f"{'5. isin (EOS token check)':<45s} {t_cuda:>6.2f}ms {t_fl:>6.2f}ms {t_fl - t_cuda:>+8.2f}ms"
    )

    # === Test 6: Full generate (1 token) ===
    gen_kwargs_cuda = dict(
        input_ids=input_ids_cuda,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    gen_kwargs_fl = dict(
        input_ids=input_ids_fl,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    def gen1_cuda():
        with torch.no_grad():
            model_cuda.generate(**gen_kwargs_cuda)

    def gen1_fl():
        with torch.no_grad():
            model_fl.generate(**gen_kwargs_fl)

    t_cuda = bench(gen1_cuda, "cuda", rounds=N // 2)
    t_fl = bench(gen1_fl, "flagos", rounds=N // 2)
    print(
        f"{'6. Full generate(max_new_tokens=1)':<45s} {t_cuda:>6.2f}ms {t_fl:>6.2f}ms {t_fl - t_cuda:>+8.2f}ms"
    )

    # === Test 7: Full generate (4 tokens, measure per-decode-step) ===
    gen_kwargs_cuda_4 = dict(
        input_ids=input_ids_cuda,
        max_new_tokens=4,
        min_new_tokens=4,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    gen_kwargs_fl_4 = dict(
        input_ids=input_ids_fl,
        max_new_tokens=4,
        min_new_tokens=4,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    def gen4_cuda():
        with torch.no_grad():
            model_cuda.generate(**gen_kwargs_cuda_4)

    def gen4_fl():
        with torch.no_grad():
            model_fl.generate(**gen_kwargs_fl_4)

    t_cuda_4 = bench(gen4_cuda, "cuda", rounds=N // 2)
    t_fl_4 = bench(gen4_fl, "flagos", rounds=N // 2)
    print(
        f"{'7. Full generate(max_new_tokens=4)':<45s} {t_cuda_4:>6.2f}ms {t_fl_4:>6.2f}ms {t_fl_4 - t_cuda_4:>+8.2f}ms"
    )

    # Derived: per-decode-step overhead in generate loop
    per_step_cuda = (t_cuda_4 - t_cuda) / 3
    per_step_fl = (t_fl_4 - t_fl) / 3
    print(
        f"{'   -> per decode step (derived)':<45s} {per_step_cuda:>6.2f}ms {per_step_fl:>6.2f}ms {per_step_fl - per_step_cuda:>+8.2f}ms"
    )

    print()
    print("=" * 75)
    print("Interpretation:")
    print("  - Test 1-2: model compute overhead (should be near zero)")
    print("  - Test 3-5: per-token ops in generate loop")
    print("  - Test 6-7: full generate including all framework logic")
    print("  - Derived:  per-step overhead = (gen4 - gen1) / 3")


if __name__ == "__main__":
    main()
