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
Profile Qwen3 training on native CUDA (baseline).

Measures op-level execution time during forward + backward + optimizer step.
Compare with profile_qwen3_train_cuda.py to identify torch_fl overhead.

Usage:
    python tests/perf/profile_qwen3_train_cuda_baseline.py [--model PATH] [--steps N]
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dummy_dataset import DummyTextDataset


class OpProfiler(TorchDispatchMode):
    """Profile op calls and execution time."""

    def __init__(self, device):
        self.device = device
        self.op_counts = defaultdict(int)
        self.op_times = defaultdict(float)
        self.cpu_fallbacks = defaultdict(int)

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

        # Detect CPU fallback
        if isinstance(result, torch.Tensor) and result.device.type == "cpu":
            self.cpu_fallbacks[op_name] += 1
        elif isinstance(result, (list, tuple)):
            for r in result:
                if isinstance(r, torch.Tensor) and r.device.type == "cpu":
                    self.cpu_fallbacks[op_name] += 1
                    break

        return result

    def report(self, total_steps):
        print("\n=== Op Profile Summary (CUDA baseline, training) ===")
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
        total_op_time = sum(self.op_times.values())
        for op, op_time in sorted(
            self.op_times.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            count = self.op_counts[op]
            avg_time_us = (op_time / count) * 1e6
            pct = (op_time / total_op_time) * 100
            print(
                f"{op:50s} {op_time:6.3f}s ({pct:5.1f}%)  "
                f"{count:5d} calls  {avg_time_us:7.1f}µs/call"
            )

        print("\n=== Top 20 most frequent ops ===")
        for op, count in sorted(
            self.op_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            op_time = self.op_times[op]
            avg_time_us = (op_time / count) * 1e6
            print(f"{op:50s} {count:5d} calls  {avg_time_us:7.1f}µs/call")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Path to model")
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of training steps to profile"
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
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # Freeze unused parameters (same as torch_fl version for fair comparison)
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
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
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
        t0 = time.time()
        loss = run_step(batch)
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        tps = tokens_per_step / elapsed
        print(
            f"    Step {i + 1}: loss={loss:.4f}, time={elapsed:.2f}s, {tps:.1f} tok/s"
        )
    print()

    # Profiled training
    print(f"[3] Profiling ({args.steps} steps)...")
    profiler = OpProfiler(device)
    step_times = []
    step_losses = []

    for i in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        torch.cuda.synchronize(device)
        t0 = time.time()
        with profiler:
            loss = run_step(batch)
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0

        step_times.append(elapsed)
        step_losses.append(loss)
        tps = tokens_per_step / elapsed
        print(
            f"    Step {i + 1}: loss={loss:.4f}, time={elapsed:.2f}s, {tps:.1f} tok/s"
        )

    # Statistics
    step_times_sorted = sorted(step_times)
    median_time = step_times_sorted[len(step_times_sorted) // 2]
    min_time = step_times_sorted[0]
    max_time = step_times_sorted[-1]
    median_tps = tokens_per_step / median_time

    print(f"\n=== Timing Summary ({args.steps} steps) ===")
    print(f"Median step time: {median_time:.3f}s ({median_tps:.1f} tok/s)")
    print(f"Min:   {min_time:.3f}s")
    print(f"Max:   {max_time:.3f}s")
    print(f"Spread: {(max_time - min_time) / median_time * 100:.1f}%")
    print(f"Avg loss: {sum(step_losses) / len(step_losses):.4f}")

    # Report
    profiler.report(args.steps)

    # Overhead analysis
    print("\n=== Overhead Analysis ===")
    total_wall = sum(step_times)
    total_op_time = sum(profiler.op_times.values())
    unaccounted = total_wall - total_op_time
    print(f"Total wall-clock:     {total_wall:.3f}s ({args.steps} steps)")
    print(
        f"Op execution time:    {total_op_time:.3f}s "
        f"({total_op_time / total_wall * 100:.1f}%)"
    )
    print(
        f"Unaccounted overhead: {unaccounted:.3f}s "
        f"({unaccounted / total_wall * 100:.1f}%)"
    )
    print("  (dispatch, Python overhead, host-device sync, framework bookkeeping)")

    # Per-step breakdown
    print("\n=== Per-Step Breakdown (median step) ===")
    ops_per_step = sum(profiler.op_counts.values()) / args.steps
    op_time_per_step = total_op_time / args.steps
    avg_op_time_us = (op_time_per_step / ops_per_step) * 1e6
    print(f"Time per step:   {median_time * 1000:.1f}ms")
    print(f"Tokens per step: {tokens_per_step}")
    print(f"Ops per step:    {ops_per_step:.0f}")
    print(f"Avg op time:     {avg_op_time_us:.1f}µs")
    print(f"Time per token:  {median_time / tokens_per_step * 1000:.2f}ms")

    # Loss convergence check
    print("\n=== Loss Trend ===")
    print(f"First loss: {step_losses[0]:.4f}")
    print(f"Last loss:  {step_losses[-1]:.4f}")
    print(f"Avg loss:   {sum(step_losses) / len(step_losses):.4f}")


if __name__ == "__main__":
    main()
