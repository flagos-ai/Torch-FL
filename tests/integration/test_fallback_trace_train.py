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
Trace which aten ops hit CPU fallback during Qwen3 training (forward + backward).

Usage:
    pytest tests/integration/test_fallback_trace_train.py -v -s
    pytest tests/integration/test_fallback_trace_train.py -v -s --steps 3
"""

import os
import sys
import time
from collections import defaultdict

import pytest
import torch
import torch_fl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dummy_dataset import DummyTextDataset
from torch.utils.data import DataLoader


def sync():
    torch_fl.flagos.synchronize()


def _has_privateuse1_kernel(op_name: str) -> bool:
    """Check if an op has a dedicated PrivateUse1 kernel (not just the catch-all fallback)."""
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1")
    except RuntimeError:
        return False


class FallbackTracer(torch.utils._python_dispatch.TorchDispatchMode):
    """Intercept all aten dispatches and classify as native vs fallback."""

    def __init__(self):
        super().__init__()
        self._cache: dict[str, bool] = {}
        self.fallback_ops = defaultdict(int)
        self.native_ops = defaultdict(int)

    def _is_native(self, op_name: str) -> bool:
        if op_name not in self._cache:
            self._cache[op_name] = _has_privateuse1_kernel(op_name)
        return self._cache[op_name]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func.name())

        has_flagos = False
        for a in torch.utils._pytree.tree_leaves((args, kwargs)):
            if isinstance(a, torch.Tensor) and a.device.type in (
                "privateuseone",
                "flagos",
            ):
                has_flagos = True
                break

        if has_flagos:
            if self._is_native(op_name):
                self.native_ops[op_name] += 1
            else:
                self.fallback_ops[op_name] += 1

        return func(*args, **(kwargs or {}))

    def report(self, title="CPU FALLBACK TRACE REPORT"):
        lines = []
        lines.append(f"\n{'=' * 70}")
        lines.append(title)
        lines.append(f"{'=' * 70}")

        total_native = sum(self.native_ops.values())
        total_fallback = sum(self.fallback_ops.values())
        total = total_native + total_fallback

        lines.append(f"Total op calls on flagos tensors: {total}")
        lines.append(
            f"  Native (GPU):      {total_native} ({100 * total_native / max(total, 1):.1f}%)"
        )
        lines.append(
            f"  CPU fallback:      {total_fallback} ({100 * total_fallback / max(total, 1):.1f}%)"
        )

        if self.fallback_ops:
            lines.append(f"\n{'─' * 70}")
            lines.append("FALLBACK OPS (sorted by call count):")
            lines.append(f"{'─' * 70}")
            for op, count in sorted(self.fallback_ops.items(), key=lambda x: -x[1]):
                lines.append(f"  {count:>6d}x  {op}")

        if self.native_ops:
            lines.append(f"\n{'─' * 70}")
            lines.append("NATIVE OPS (sorted by call count):")
            lines.append(f"{'─' * 70}")
            for op, count in sorted(self.native_ops.items(), key=lambda x: -x[1]):
                lines.append(f"  {count:>6d}x  {op}")

        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)


@pytest.fixture(scope="module")
def train_ctx(request):
    model_path = request.config.getoption("--model")
    steps = request.config.getoption("--steps")
    batch_size = request.config.getoption("--batch-size")
    seq_len = request.config.getoption("--seq-len")
    lr = request.config.getoption("--lr")

    torch_fl.flagos.set_device(0)
    device = "flagos:0"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[setup] Loading model ({model_path})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)
    model.train()

    # Freeze unused parameters
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

    sync()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"    Parameters: {total:.2f}M total, {trainable:.2f}M trainable")
    print(f"    Load time: {time.time() - t0:.2f}s")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=seq_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "dataloader": dataloader,
        "device": device,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def test_trace_fallback_train(train_ctx):
    """Run training steps and report which ops hit CPU fallback."""
    model = train_ctx["model"]
    optimizer = train_ctx["optimizer"]
    device = train_ctx["device"]
    steps = train_ctx["steps"]
    data_iter = iter(train_ctx["dataloader"])

    # Warm-up step (first step may compile Triton kernels, not traced)
    batch = next(data_iter)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    sync()
    print(f"\n[warmup] loss={outputs.loss.item():.4f}")

    # Traced training steps
    tracer = FallbackTracer()
    losses = []

    with tracer:
        t0 = time.time()
        for i in range(steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_ctx["dataloader"])
                batch = next(data_iter)

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
            sync()

            losses.append(loss.item())
            print(f"    [Step {i + 1}/{steps}] loss={loss.item():.4f}")

        elapsed = time.time() - t0

    total_tokens = train_ctx["batch_size"] * train_ctx["seq_len"] * steps
    print(
        f"\n[traced] {steps} steps in {elapsed:.2f}s, "
        f"throughput={total_tokens / elapsed:.1f} tok/s"
    )
    print(tracer.report(title="CPU FALLBACK TRACE — TRAINING (forward + backward)"))

    # Assert: all losses finite
    assert all(torch.isfinite(torch.tensor(val)) for val in losses), (
        "Some losses are not finite"
    )

    # Warn (but don't fail) if there are fallback ops
    if tracer.fallback_ops:
        print(
            f"\n⚠️  WARNING: {len(tracer.fallback_ops)} unique ops still hit CPU fallback "
            f"({sum(tracer.fallback_ops.values())} total calls)"
        )
