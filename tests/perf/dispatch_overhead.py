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
Micro-benchmark: dispatch overhead isolation.

Measures per-op overhead of PrivateUse1 dispatch vs native CUDA dispatch
by running trivial ops (that finish instantly) and measuring wall-clock.

This isolates dispatch/framework overhead from actual kernel execution time.

Usage:
    CUDA_VISIBLE_DEVICES=2 FLAGOS_BACKEND_CONFIG=... python tests/perf/dispatch_overhead.py
"""

import argparse
import time

import torch
import torch_fl


def bench_op(fn, warmup=100, rounds=10000):
    """Run fn() many times, return average time in microseconds."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(rounds):
        fn()
    return (time.perf_counter() - t0) / rounds * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10000, help="Rounds per op")
    args = parser.parse_args()
    N = args.n

    device_flagos = "flagos:0"
    device_cuda = "cuda:0"

    # Pre-allocate tensors
    fc = torch.randn(64, 64, device=device_cuda, dtype=torch.float16)
    ff = torch.randn(64, 64, device=device_flagos, dtype=torch.float16)

    fc_big = torch.randn(256, 256, device=device_cuda, dtype=torch.float16)
    ff_big = torch.randn(256, 256, device=device_flagos, dtype=torch.float16)

    # Sync before starting
    torch.cuda.synchronize()
    torch_fl.flagos.synchronize()

    print(f"=== Dispatch Overhead Micro-Benchmark ({N} rounds/op) ===\n")
    print(f"{'Op':<30s} {'CUDA':>10s} {'flagos':>10s} {'delta':>10s}")
    print("-" * 65)

    # --- 1. Metadata-only ops (no kernel launch) ---
    t_cuda = bench_op(lambda: fc.view(64, 64), rounds=N)
    t_flagos = bench_op(lambda: ff.view(64, 64), rounds=N)
    print(
        f"{'view (no-op reshape)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(lambda: fc.t(), rounds=N)
    t_flagos = bench_op(lambda: ff.t(), rounds=N)
    print(
        f"{'t (transpose)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(lambda: fc.unsqueeze(0), rounds=N)
    t_flagos = bench_op(lambda: ff.unsqueeze(0), rounds=N)
    print(
        f"{'unsqueeze':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(lambda: fc[0:32], rounds=N)
    t_flagos = bench_op(lambda: ff[0:32], rounds=N)
    print(
        f"{'slice [0:32]':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(lambda: fc.expand(64, 64), rounds=N)
    t_flagos = bench_op(lambda: ff.expand(64, 64), rounds=N)
    print(
        f"{'expand':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    print()

    # --- 2. Simple element-wise ops (lightweight kernel) ---
    t_cuda = bench_op(lambda: fc + fc, rounds=N)
    t_flagos = bench_op(lambda: ff + ff, rounds=N)
    print(
        f"{'add (64x64)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(lambda: fc * fc, rounds=N)
    t_flagos = bench_op(lambda: ff * ff, rounds=N)
    print(
        f"{'mul (64x64)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(lambda: fc.neg(), rounds=N)
    t_flagos = bench_op(lambda: ff.neg(), rounds=N)
    print(
        f"{'neg (64x64)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    print()

    # --- 3. Matmul (heavier kernel) ---
    t_cuda = bench_op(lambda: torch.mm(fc, fc), rounds=N)
    t_flagos = bench_op(lambda: torch.mm(ff, ff), rounds=N)
    print(
        f"{'mm (64x64)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(lambda: torch.mm(fc_big, fc_big), rounds=N)
    t_flagos = bench_op(lambda: torch.mm(ff_big, ff_big), rounds=N)
    print(
        f"{'mm (256x256)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    print()

    # --- 4. Allocation overhead ---
    t_cuda = bench_op(
        lambda: torch.empty(64, 64, device=device_cuda, dtype=torch.float16),
        rounds=N,
    )
    t_flagos = bench_op(
        lambda: torch.empty(64, 64, device=device_flagos, dtype=torch.float16),
        rounds=N,
    )
    print(
        f"{'empty (64x64, alloc only)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    t_cuda = bench_op(
        lambda: torch.empty(1024, 1024, device=device_cuda, dtype=torch.float16),
        rounds=N,
    )
    t_flagos = bench_op(
        lambda: torch.empty(1024, 1024, device=device_flagos, dtype=torch.float16),
        rounds=N,
    )
    print(
        f"{'empty (1024x1024, alloc only)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    print()

    # --- 5. clone (involves alloc + kernel) ---
    t_cuda = bench_op(lambda: fc.clone(), rounds=N)
    t_flagos = bench_op(lambda: ff.clone(), rounds=N)
    print(
        f"{'clone (64x64)':<30s} {t_cuda:>8.2f}µs {t_flagos:>8.2f}µs {t_flagos - t_cuda:>+8.2f}µs"
    )

    print()

    # --- Summary ---
    print("=" * 65)
    print("Positive delta = flagos slower than CUDA.")
    print("Consistent delta across metadata ops = per-op dispatch tax.")
    print("Growing delta with op complexity = allocator or boxing overhead.")


if __name__ == "__main__":
    main()
