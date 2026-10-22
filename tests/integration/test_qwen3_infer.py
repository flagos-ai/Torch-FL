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
Qwen3 End-to-End Inference Test (pytest)

All tests run on the flagos device.

Usage:
    pytest tests/integration/test_qwen3_infer.py -v -s --model /path/to/model
"""

import time

import pytest
import torch
import torch_fl


@pytest.fixture(scope="module")
def ctx(request):
    model_path = request.config.getoption("--model")
    max_new_tokens = request.config.getoption("--max-new-tokens")

    print(
        f"\nflagos device count={torch_fl.flagos.device_count()}  "
        f"FlagGems enabled={torch_fl.is_flaggems_enabled()}  "
        f"registered ops={len(torch_fl.get_registered_ops())}"
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
    )
    device = "flagos:0"
    model = model.to(device)
    model.eval()
    print(
        f"Model device: {next(model.parameters()).device}  "
        f"load time: {time.time() - t0:.2f}s"
    )

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
    print(f"Input tokens: {inputs.input_ids.shape[1]}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "max_new_tokens": max_new_tokens,
        "model_path": model_path,
    }


def sync():
    torch_fl.flagos.synchronize()


def run_inference(ctx):
    model, inputs = ctx["model"], ctx["inputs"]
    sync()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=ctx["max_new_tokens"])
    sync()
    elapsed = time.time() - t0
    new_tokens = output.shape[1] - inputs.input_ids.shape[1]
    return output, elapsed, new_tokens


class TestQwen3Inference:
    def test_first_inference(self, ctx):
        """First run — may include kernel compilation."""
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  First: {elapsed:.2f}s, {new_tokens} tokens, "
            f"{new_tokens / elapsed:.2f} tok/s"
        )
        assert new_tokens > 0, "No tokens generated on first run"

    def test_second_inference(self, ctx):
        """Second run — kernel cache warm."""
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  Second: {elapsed:.2f}s, {new_tokens} tokens, "
            f"{new_tokens / elapsed:.2f} tok/s"
        )
        assert new_tokens > 0

    def test_third_inference(self, ctx):
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  Third: {elapsed:.2f}s, {new_tokens} tokens, "
            f"{new_tokens / elapsed:.2f} tok/s"
        )
        assert new_tokens > 0

    def test_fourth_inference_and_output(self, ctx):
        """Fourth run — verify non-empty decoded text."""
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  Fourth: {elapsed:.2f}s, {new_tokens} tokens, "
            f"{new_tokens / elapsed:.2f} tok/s"
        )
        decoded = ctx["tokenizer"].decode(
            output[0][ctx["inputs"].input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        print(f"\nGenerated output:\n{decoded}")
        assert new_tokens > 0
        assert len(decoded.strip()) > 0, "Generated text is empty"
