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
Qwen3 Profiler Integration Test

Validates that torch.profiler captures both CPU ops and GPU kernels
for a real model inference workload.

Usage:
    pytest tests/integration/test_profiler_qwen3_infer.py -v -s
"""

import json
import tempfile

import pytest
import torch
import torch_fl  # noqa: F401  -- registers the "flagos" PrivateUse1 device
from torch.profiler import ProfilerActivity, profile


MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def model_and_tokenizer(request):
    """Load Qwen3 model once for all tests.

    Uses the shared --model option (default Qwen/Qwen3-0.6B) like the other
    integration tests, so a local path can be supplied. On a host without
    network access, point HF at a pre-populated cache and run offline:
        HF_HOME=<your-hf-cache-dir> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = request.config.getoption("--model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).to(
        "flagos"
    )
    model.eval()
    return model, tokenizer


def test_profiler_qwen3_basic(model_and_tokenizer):
    """Basic profiler test: confirm no crash under profiling."""
    model, tokenizer = model_and_tokenizer
    ids = tokenizer("Hello world", return_tensors="pt").input_ids.to("flagos")

    torch.flagos.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        with torch.no_grad():
            model(ids)
        torch.flagos.synchronize()

    # Should complete without crash
    events = prof.key_averages()
    assert len(events) > 0, "No profiler events captured"
    print(f"Captured {len(events)} profiler events")


def test_profiler_qwen3_chrome_trace_kernels(model_and_tokenizer):
    """Chrome trace test: verify GPU kernel events appear in trace."""
    model, tokenizer = model_and_tokenizer
    ids = tokenizer("Hello world", return_tensors="pt").input_ids.to("flagos")

    torch.flagos.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        with torch.no_grad():
            model(ids)
        torch.flagos.synchronize()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)

    with open(path) as fh:
        trace_data = json.load(fh)

    events = trace_data["traceEvents"]
    kernels = [
        e
        for e in events
        if "kernel" in str(e.get("name", "")).lower()
        or e.get("cat") in ("kernel", "Kernel")
    ]

    print(f"qwen3 trace: {len(kernels)} kernel events, saved to {path}")
    print(f"Total trace events: {len(events)}")

    assert len(events) > 0, "Chrome trace is empty"
    assert "traceEvents" in trace_data, "Invalid chrome trace format"

    # Stage B core goal: a real model inference must surface GPU kernels in the
    # trace, with real CUPTI-decoded names and non-zero durations, on the
    # CPU-torch + external libtorch_cuda stack (no CUDA wheel). A Qwen3 forward
    # pass runs many matmuls/elementwise kernels, so expect a substantial count.
    assert len(kernels) > 0, "no GPU kernel events captured in qwen3 trace"
    named = [
        e
        for e in kernels
        if e.get("name") and e.get("name") not in ("kernel", "Memcpy")
    ]
    assert named, "GPU kernel events have no real names (CUPTI record decode broken)"
    assert any(e.get("dur", 0) > 0 for e in named), (
        "all GPU kernel durations are zero (CUPTI timestamp decode broken)"
    )
    print(
        f"✓ GPU kernel timeline captured: {len(named)} named kernels, "
        f"sample={named[0].get('name')[:60]!r} dur={named[0].get('dur')}"
    )
