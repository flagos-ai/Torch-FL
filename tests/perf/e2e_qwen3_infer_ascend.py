"""
End-to-end Qwen3 inference benchmark on Ascend 910, comparing backends.

Two backends share one identical measurement harness (same model, same prompt,
same fixed token count, same warmup/round counts) so the numbers are directly
comparable:

  --backend torch_fl   torch_fl + aclnn C++ kernels (device flagos:0)
  --backend torch_npu  Huawei torch_npu baseline    (device npu:0)

Usage:
    # aclnn path (env ascend_p0_210)
    FLAGOS_ACCELERATOR=ascend python tests/perf/e2e_qwen3_infer_ascend.py \
        --backend torch_fl --model /tmp/Qwen3-0.6B --tokens 64

    # torch_npu baseline (env torch_npu_210)
    python tests/perf/e2e_qwen3_infer_ascend.py \
        --backend torch_npu --model /tmp/Qwen3-0.6B --tokens 64
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_backend(backend):
    """Import the backend module, return (device_str, synchronize_fn)."""
    if backend == "torch_fl":
        import torch_fl

        torch_fl.flagos.set_device(0)
        return "flagos:0", torch_fl.flagos.synchronize
    elif backend == "torch_npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        return "npu:0", torch.npu.synchronize
    raise ValueError(f"unknown backend {backend}")


def main():
    args = parse_args()
    device, synchronize = setup_backend(args.backend)

    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
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
    # Force eager attention so both backends run the same math path.
    model.config._attn_implementation = "eager"

    # Optional: route Qwen3RMSNorm through F.rms_norm so it lands on a single
    # fused kernel (aclnnRmsNorm on torch_fl/Ascend) instead of HF's ~6
    # elementwise ops + 2 dtype casts. Applied to BOTH backends so the
    # comparison stays fair (torch_npu also gets its fused rms_norm path).
    if args.fuse_rmsnorm:
        from transformers.models.qwen3 import modeling_qwen3 as _m

        def _fused_forward(self, hidden_states):
            return torch.nn.functional.rms_norm(
                hidden_states,
                (hidden_states.shape[-1],),
                self.weight,
                self.variance_epsilon,
            )

        _m.Qwen3RMSNorm.forward = _fused_forward
        print("RMSNorm: fused (F.rms_norm)")
    else:
        print("RMSNorm: HF default (decomposed)")
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
        synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**gen_kwargs)
        synchronize()
        print(f"  Round {i + 1}: {time.perf_counter() - t0:.3f}s")
    print()

    # Benchmark
    print(f"Benchmarking ({args.rounds} rounds)...")
    round_times = []
    for i in range(args.rounds):
        synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**gen_kwargs)
        synchronize()
        elapsed = time.perf_counter() - t0

        new_tokens = output.shape[1] - input_len
        tps = new_tokens / elapsed
        round_times.append(elapsed)
        print(f"  Round {i + 1}: {elapsed:.3f}s, {new_tokens} tokens, {tps:.2f} tok/s")

    round_times.sort()
    median_time = round_times[len(round_times) // 2]
    min_time = round_times[0]
    max_time = round_times[-1]
    median_tps = args.tokens / median_time

    print(f"\n=== E2E Inference Results ({args.backend}) ===")
    print(f"Tokens generated: {args.tokens} (greedy, fixed)")
    print(f"Median: {median_time:.3f}s ({median_tps:.2f} tok/s)")
    print(f"Min:    {min_time:.3f}s ({args.tokens / min_time:.2f} tok/s)")
    print(f"Max:    {max_time:.3f}s ({args.tokens / max_time:.2f} tok/s)")
    print(f"Spread: {(max_time - min_time) / median_time * 100:.1f}%")
    print(f"Time per token: {median_time / args.tokens * 1000:.2f}ms")


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2E Qwen3 inference benchmark (Ascend)"
    )
    parser.add_argument("--backend", choices=["torch_fl", "torch_npu"], required=True)
    parser.add_argument("--model", default="/tmp/Qwen3-0.6B", help="Path to model")
    parser.add_argument(
        "--tokens", type=int, default=64, help="Exact number of new tokens to generate"
    )
    parser.add_argument(
        "--rounds", type=int, default=5, help="Benchmark rounds (take median)"
    )
    parser.add_argument(
        "--fuse-rmsnorm",
        action="store_true",
        help="Route Qwen3RMSNorm through F.rms_norm (fused kernel on both backends)",
    )
    parser.add_argument("--warmup-rounds", type=int, default=3, help="Warmup rounds")
    return parser.parse_args()


if __name__ == "__main__":
    main()
