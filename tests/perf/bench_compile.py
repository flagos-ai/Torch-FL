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
Performance benchmark: torch.compile fusion gains on flagos device.

Measures speedup from kernel fusion (compiled vs eager) to validate we achieve
parity with inductor+triton performance gains.

Usage:
    python tests/perf/bench_compile.py
    python tests/perf/bench_compile.py --model=mlp --batch-size=128
    FLAGOS_USE_FLAGTREE=1 python tests/perf/bench_compile.py  # require FlagTree
"""

import argparse
import time
import torch
import torch.nn as nn
import torch_fl


class MLPModel(nn.Module):
    """Multi-layer perceptron with many fusible ops."""

    def __init__(self, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Many elementwise ops that should fuse
        x = self.fc1(x)
        x = torch.relu(x)
        x = x * 2.0
        x = x + 1.0

        x = self.fc2(x)
        x = torch.gelu(x)
        x = x / 2.0

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class ConvModel(nn.Module):
    """Convolutional model with fusible activation patterns."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x * 0.5 + 0.5  # Normalization pattern

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block (attention + FFN)."""

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual (many fusible ops)
        ffn = self.linear1(x)
        ffn = torch.relu(ffn)
        ffn = self.linear2(ffn)
        x = self.norm2(x + ffn)

        return x


def benchmark_model(model, inputs, warmup=10, rounds=100):
    """
    Benchmark model execution time.

    Returns average time per forward pass in milliseconds.
    """
    device = next(model.parameters()).device

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

    # Sync before timing
    if device.type == "privateuseone":
        torch_fl.flagos.synchronize()
    else:
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(rounds):
        with torch.no_grad():
            _ = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

    # Sync after timing
    if device.type == "privateuseone":
        torch_fl.flagos.synchronize()
    else:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / rounds) * 1000  # Convert to ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["mlp", "conv", "transformer"], default="mlp"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--compare-cuda", action="store_true", help="Compare with CUDA baseline"
    )
    args = parser.parse_args()

    device_flagos = "flagos:0"
    device_cuda = "cuda:0" if args.compare_cuda else None

    # Build model and inputs
    if args.model == "mlp":
        model = MLPModel(hidden_size=args.hidden_size)
        inputs_shape = (args.batch_size, args.hidden_size)
        inputs_flagos = torch.randn(*inputs_shape, device=device_flagos)
        inputs_cuda = (
            torch.randn(*inputs_shape, device=device_cuda) if device_cuda else None
        )
    elif args.model == "conv":
        model = ConvModel()
        inputs_shape = (args.batch_size, 3, 224, 224)
        inputs_flagos = torch.randn(*inputs_shape, device=device_flagos)
        inputs_cuda = (
            torch.randn(*inputs_shape, device=device_cuda) if device_cuda else None
        )
    elif args.model == "transformer":
        model = TransformerBlock(d_model=args.hidden_size)
        inputs_shape = (args.batch_size, 128, args.hidden_size)
        inputs_flagos = torch.randn(*inputs_shape, device=device_flagos)
        inputs_cuda = (
            torch.randn(*inputs_shape, device=device_cuda) if device_cuda else None
        )

    print("=== torch.compile Performance Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Rounds: {args.rounds}")
    print(
        f"FlagTree enabled: {bool(int(torch.os.environ.get('FLAGOS_USE_FLAGTREE', '0')))}"
    )
    print()

    # === Flagos Device ===
    print("--- Flagos Device ---")

    # Eager mode (baseline)
    model_flagos_eager = model.to(device_flagos)
    time_eager = benchmark_model(
        model_flagos_eager, inputs_flagos, warmup=args.warmup, rounds=args.rounds
    )
    print(f"Eager mode:        {time_eager:>8.3f} ms/iter")

    # Compiled mode (inductor fusion)
    try:
        model_flagos_compiled = torch.compile(model.to(device_flagos), backend="flagos")
        time_compiled = benchmark_model(
            model_flagos_compiled, inputs_flagos, warmup=args.warmup, rounds=args.rounds
        )
        speedup = time_eager / time_compiled
        print(
            f"Compiled (flagos): {time_compiled:>8.3f} ms/iter  ({speedup:.2f}x speedup)"
        )
    except Exception as e:
        print(f"Compiled (flagos): FAILED - {e}")
        speedup = 1.0

    print()

    # === CUDA Baseline (for comparison) ===
    if args.compare_cuda:
        print("--- CUDA Device (baseline) ---")

        model_cuda_eager = model.to(device_cuda)
        time_cuda_eager = benchmark_model(
            model_cuda_eager, inputs_cuda, warmup=args.warmup, rounds=args.rounds
        )
        print(f"Eager mode:          {time_cuda_eager:>8.3f} ms/iter")

        try:
            model_cuda_compiled = torch.compile(
                model.to(device_cuda), backend="inductor"
            )
            time_cuda_compiled = benchmark_model(
                model_cuda_compiled, inputs_cuda, warmup=args.warmup, rounds=args.rounds
            )
            speedup_cuda = time_cuda_eager / time_cuda_compiled
            print(
                f"Compiled (inductor): {time_cuda_compiled:>8.3f} ms/iter  ({speedup_cuda:.2f}x speedup)"
            )

            # Compare flagos vs CUDA speedups
            print()
            print("--- Speedup Comparison ---")
            print(f"Flagos compile speedup: {speedup:.2f}x")
            print(f"CUDA compile speedup:   {speedup_cuda:.2f}x")
            parity = (speedup / speedup_cuda) * 100
            print(f"Parity:                 {parity:.1f}% (flagos vs CUDA)")

        except Exception as e:
            print(f"Compiled (inductor): FAILED - {e}")

    # === Summary ===
    print()
    print("=== Summary ===")
    if speedup >= 1.5:
        print(f"✅ Fusion gain achieved: {speedup:.2f}x speedup")
    elif speedup >= 1.1:
        print(f"⚠️  Modest fusion gain: {speedup:.2f}x speedup")
    else:
        print(f"❌ No significant fusion gain: {speedup:.2f}x")

    if args.compare_cuda and speedup_cuda > 0:
        if parity >= 80:
            print(f"✅ Parity with inductor+triton: {parity:.1f}%")
        elif parity >= 60:
            print(f"⚠️  Approaching parity: {parity:.1f}%")
        else:
            print(f"❌ Below parity: {parity:.1f}%")


if __name__ == "__main__":
    main()
