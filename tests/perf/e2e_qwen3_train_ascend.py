"""
End-to-end Qwen3 training benchmark on Ascend 910, comparing backends.

One identical harness (same model, seed, batch/seq, optimizer, step count) runs
against two backends so step-time and throughput are directly comparable:

  --backend torch_fl   torch_fl + aclnn C++ kernels (device flagos:0)
  --backend torch_npu  Huawei torch_npu baseline    (device npu:0)

AdamW defaults to its foreach path on both backends (pass --no-foreach for the
single-tensor path); the setting is applied identically either way so the
optimizer contributes the same op mix to both measurements.

Usage:
    FLAGOS_ACCELERATOR=ascend python tests/perf/e2e_qwen3_train_ascend.py \
        --backend torch_fl --model /tmp/Qwen3-0.6B --steps 10

    python tests/perf/e2e_qwen3_train_ascend.py \
        --backend torch_npu --model /tmp/Qwen3-0.6B --steps 10
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dummy_dataset import DummyTextDataset  # noqa: E402


def setup_backend(backend):
    if backend == "torch_fl":
        import torch_fl

        torch_fl.flagos.set_device(0)
        return "flagos:0", torch_fl.flagos.synchronize
    elif backend == "torch_npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        return "npu:0", torch.npu.synchronize
    raise ValueError(f"unknown backend {backend}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="E2E Qwen3 training benchmark (Ascend)"
    )
    parser.add_argument("--backend", choices=["torch_fl", "torch_npu"], required=True)
    parser.add_argument("--model", default="/tmp/Qwen3-0.6B", help="Path to model")
    parser.add_argument("--steps", type=int, default=10, help="Benchmark steps")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--foreach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="AdamW foreach path (--no-foreach for the single-tensor path)",
    )
    args = parser.parse_args()

    device, synchronize = setup_backend(args.backend)
    set_seed(42)

    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print()

    print("[1] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)
    model.train()
    print(f"    Load time: {time.time() - t0:.2f}s")

    # Freeze unused parameters (embedding tie etc.) so grads match across backends.
    dummy = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(input_ids=dummy, use_cache=False)
        out.logits.sum().backward()
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
            unused.append(name)
        else:
            param.grad = None
    print(f"    Frozen {len(unused)} unused parameters")

    synchronize()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"    Parameters: {total:.2f}M total, {trainable:.2f}M trainable")
    print()

    # Both backends get the same foreach setting so the optimizer contributes the
    # same op mix to each measurement.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        foreach=args.foreach,
    )
    print(f"    Optimizer: AdamW(foreach={args.foreach})")
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=args.seq_len)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    tokens_per_step = args.batch_size * args.seq_len

    def run_step(batch):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    print(f"[2] Warmup ({args.warmup_steps} steps)...")
    data_iter = iter(dataloader)
    for i in range(args.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        synchronize()
        t0 = time.perf_counter()
        loss = run_step(batch)
        synchronize()
        elapsed = time.perf_counter() - t0
        print(
            f"    Step {i + 1}: loss={loss:.4f}, time={elapsed:.2f}s, "
            f"{tokens_per_step / elapsed:.1f} tok/s"
        )
    print()

    print(f"[3] Benchmarking ({args.steps} steps)...")
    step_times = []
    step_losses = []
    for i in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        synchronize()
        t0 = time.perf_counter()
        loss = run_step(batch)
        synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)
        step_losses.append(loss)
        print(
            f"    Step {i + 1}: loss={loss:.4f}, time={elapsed:.3f}s, "
            f"{tokens_per_step / elapsed:.1f} tok/s"
        )

    step_times_sorted = sorted(step_times)
    median_time = step_times_sorted[len(step_times_sorted) // 2]
    min_time = step_times_sorted[0]
    max_time = step_times_sorted[-1]
    median_tps = tokens_per_step / median_time

    print(f"\n=== E2E Training Results ({args.backend}) ===")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"Tokens per step: {tokens_per_step}")
    print(f"Steps: {args.steps}")
    print(f"Median step time: {median_time:.3f}s ({median_tps:.1f} tok/s)")
    print(f"Min:              {min_time:.3f}s ({tokens_per_step / min_time:.1f} tok/s)")
    print(f"Max:              {max_time:.3f}s ({tokens_per_step / max_time:.1f} tok/s)")
    print(f"Spread: {(max_time - min_time) / median_time * 100:.1f}%")
    print(f"Time per token: {median_time / tokens_per_step * 1000:.2f}ms")
    print()
    print("=== Loss Trend ===")
    print(f"First loss: {step_losses[0]:.4f}")
    print(f"Last loss:  {step_losses[-1]:.4f}")
    print(f"Avg loss:   {sum(step_losses) / len(step_losses):.4f}")


if __name__ == "__main__":
    main()
