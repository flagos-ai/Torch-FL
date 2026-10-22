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
Trace which aten ops hit CPU fallback during Qwen3 inference.

Usage:
    pytest tests/integration/test_fallback_trace.py -v -s
"""

import time
from collections import defaultdict

import pytest
import torch
import torch_fl


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

    def report(self):
        lines = []
        lines.append(f"\n{'=' * 70}")
        lines.append("CPU FALLBACK TRACE REPORT")
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
def model_ctx(request):
    model_path = request.config.getoption("--model")
    max_new_tokens = request.config.getoption("--max-new-tokens")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to("flagos:0")
    model.eval()

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
    inputs = tokenizer([text], return_tensors="pt").to("flagos:0")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "max_new_tokens": max_new_tokens,
    }


def test_trace_fallback_ops(model_ctx):
    """Run one inference pass and report which ops hit CPU fallback."""
    tracer = FallbackTracer()

    model = model_ctx["model"]
    inputs = model_ctx["inputs"]

    # Warm up (first run may compile Triton kernels)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=8)
    torch_fl.flagos.synchronize()

    # Traced run
    with torch.no_grad(), tracer:
        t0 = time.time()
        output = model.generate(**inputs, max_new_tokens=model_ctx["max_new_tokens"])
        torch_fl.flagos.synchronize()
        elapsed = time.time() - t0

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    print(
        f"\nInference: {elapsed:.2f}s, {new_tokens} tokens, {new_tokens / elapsed:.2f} tok/s"
    )
    print(tracer.report())

    assert new_tokens > 0
