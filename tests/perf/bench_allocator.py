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
Micro-benchmark: allocation throughput comparison.

Measures the overhead of tensor allocation/deallocation
between native CUDA (PyTorch CUDACachingAllocator) and flagos (our allocator).
"""

import torch
import time


def bench_alloc_free(device, N=100_000, sizes=None):
    """Measure alloc+free cycle time for tensors of various sizes."""
    if sizes is None:
        sizes = [(64, 128), (128, 128), (1, 4096), (32, 32, 128)]

    results = {}
    for shape in sizes:
        # Warmup: fill the cache
        tensors = [torch.empty(shape, device=device) for _ in range(100)]
        del tensors
        if device == "cuda":
            torch.cuda.synchronize()

        # Bench: alloc + immediate dealloc (hits cache)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(N):
            t = torch.empty(shape, device=device)
            del t
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        results[shape] = (t1 - t0) / N

    return results


def main():

    print("=== Allocation throughput (alloc + free, cached) ===")
    print(
        f"{'Shape':<20} {'CUDA (ns)':<12} {'flagos (ns)':<12} {'Δ (ns)':<10} {'Δ (µs)'}"
    )
    print("-" * 70)

    sizes = [(64, 128), (128, 128), (1, 4096), (32, 32, 128), (1, 1)]

    cuda_results = bench_alloc_free("cuda", sizes=sizes)
    flagos_results = bench_alloc_free("flagos", sizes=sizes)

    total_cuda = 0
    total_flagos = 0
    for shape in sizes:
        c = cuda_results[shape]
        f = flagos_results[shape]
        total_cuda += c
        total_flagos += f
        print(
            f"{str(shape):<20} {c:<12.0f} {f:<12.0f} {c - f:<10.0f} {(c - f) / 1000:.2f}"
        )

    avg_cuda = total_cuda / len(sizes)
    avg_flagos = total_flagos / len(sizes)
    print("-" * 70)
    print(
        f"{'Average':<20} {avg_cuda:<12.0f} {avg_flagos:<12.0f} {avg_cuda - avg_flagos:<10.0f} {(avg_cuda - avg_flagos) / 1000:.2f}"
    )
    print()
    print(
        f"Per-token estimate (~1500 allocs): CUDA {avg_cuda * 1500 / 1e6:.3f}ms, flagos {avg_flagos * 1500 / 1e6:.3f}ms, Δ {(avg_cuda - avg_flagos) * 1500 / 1e6:.3f}ms"
    )


if __name__ == "__main__":
    main()
