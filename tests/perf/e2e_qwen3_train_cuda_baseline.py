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
End-to-end Qwen3 training benchmark on native CUDA (no per-op profiling).

Measures pure wall-clock step time and throughput without TorchDispatchMode
overhead, providing a fair e2e comparison with torch_fl.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/perf/e2e_qwen3_train_cuda_baseline.py \
        --model /nfs/hcr/models/Qwen/Qwen3-0.6B --steps 10
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


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="E2E Qwen3 training benchmark on native CUDA"
    )
    parser.add_argument(
        "--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B", help="Path to model"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of benchmark steps"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=3, help="Number of warmup steps"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = args.device
    assert torch.cuda.is_available(), "CUDA not available"

    set_seed(42)

    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print()

    # Load model
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

    # Freeze unused parameters (same logic as profile script)
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

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"    Parameters: {total:.2f}M total, {trainable:.2f}M trainable")
    print()

    # Optimizer and data
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
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

    # Warmup
    print(f"[2] Warmup ({args.warmup_steps} steps)...")
    data_iter = iter(dataloader)
    for i in range(args.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss = run_step(batch)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        tps = tokens_per_step / elapsed
        print(
            f"    Step {i + 1}: loss={loss:.4f}, time={elapsed:.2f}s, {tps:.1f} tok/s"
        )
    print()

    # Benchmark
    print(f"[3] Benchmarking ({args.steps} steps)...")
    step_times = []
    step_losses = []

    for i in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss = run_step(batch)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        step_times.append(elapsed)
        step_losses.append(loss)
        tps = tokens_per_step / elapsed
        print(
            f"    Step {i + 1}: loss={loss:.4f}, time={elapsed:.3f}s, {tps:.1f} tok/s"
        )

    # Statistics
    step_times_sorted = sorted(step_times)
    median_time = step_times_sorted[len(step_times_sorted) // 2]
    min_time = step_times_sorted[0]
    max_time = step_times_sorted[-1]
    median_tps = tokens_per_step / median_time

    print("\n=== E2E Training Benchmark Results (CUDA baseline) ===")
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
