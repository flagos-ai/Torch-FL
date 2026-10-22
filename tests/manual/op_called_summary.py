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

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_PATH = "/nfs/hcr/models/Qwen/Qwen3-0.6B"


class AtenOpCollector(TorchDispatchMode):
    def __init__(self):
        self.ops = defaultdict(int)  # op -> call count

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[str(func)] += 1
        return func(*args, **(kwargs or {}))


model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
inputs = tokenizer("hello world", return_tensors="pt").to("cuda")

# default output: 'sdpa'
print(
    "Attention implementation:",
    model.model.layers[0].self_attn.config._attn_implementation,
)
model.model.layers[0].self_attn.config._attn_implementation = "eager"

# Create separate collectors for inference and training
inference_collector = AtenOpCollector()
training_collector = AtenOpCollector()

# --- inference ---
print("=== Running inference ===")
with inference_collector:
    with torch.no_grad():
        out = model(**inputs)

# --- training (forward + backward) ---
print("=== Running training ===")
model.train()
with training_collector:
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()

# Calculate incremental ops in training (not present in inference)
inference_ops = set(inference_collector.ops.keys())
training_ops = set(training_collector.ops.keys())
incremental_ops = training_ops - inference_ops

# Output results
print("\n=== Inference Operators ===")
for op, count in sorted(inference_collector.ops.items()):
    print(f"{op:60s} {count}")

print("\n=== All Training Operators ===")
for op, count in sorted(training_collector.ops.items()):
    print(f"{op:60s} {count}")

print("\n=== Incremental Training Operators (not in inference) ===")
if incremental_ops:
    for op in sorted(incremental_ops):
        print(f"{op:60s} {training_collector.ops[op]}")
else:
    print("No incremental operators found")

# Summary
print("\n=== Summary ===")
print(f"Inference operators: {len(inference_collector.ops)}")
print(f"Training operators: {len(training_collector.ops)}")
print(f"Incremental operators: {len(incremental_ops)}")
