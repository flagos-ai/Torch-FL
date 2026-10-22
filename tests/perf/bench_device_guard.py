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
Micro-benchmark: cudaGetDevice() driver call vs thread_local read.

Measures the overhead of DeviceGuard in CUDA vs PrivateUse1 paths.
"""

import torch
import time


def bench_device_guard_overhead():
    """Compare per-op overhead for metadata-only ops between CUDA and flagos."""

    # --- CUDA baseline: view op ---
    x_cuda = torch.randn(64, 128, device="cuda")

    # Warmup
    for _ in range(1000):
        _ = x_cuda.view(128, 64)
    torch.cuda.synchronize()

    N = 100_000
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(N):
        _ = x_cuda.view(128, 64)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    cuda_ns = (t1 - t0) / N

    # --- flagos: view op ---
    x_fl = torch.randn(64, 128, device="flagos")

    # Warmup
    for _ in range(1000):
        _ = x_fl.view(128, 64)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(N):
        _ = x_fl.view(128, 64)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    flagos_ns = (t1 - t0) / N

    print("=== Metadata op (view) overhead ===")
    print(f"CUDA:   {cuda_ns:.0f} ns/call ({cuda_ns / 1000:.2f} µs)")
    print(f"flagos: {flagos_ns:.0f} ns/call ({flagos_ns / 1000:.2f} µs)")
    print(
        f"Δ:      {cuda_ns - flagos_ns:.0f} ns ({(cuda_ns - flagos_ns) / 1000:.2f} µs)"
    )
    print()

    # --- Also test t() and unsqueeze ---
    for op_name, cuda_op, fl_op in [
        ("t (2D transpose)", lambda: x_cuda.t(), lambda: x_fl.t()),
        ("unsqueeze(0)", lambda: x_cuda.unsqueeze(0), lambda: x_fl.unsqueeze(0)),
        (
            "expand",
            lambda: x_cuda[:1].expand(64, 128),
            lambda: x_fl[:1].expand(64, 128),
        ),
    ]:
        # Warmup
        for _ in range(1000):
            cuda_op()
            fl_op()
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(N):
            cuda_op()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        c_ns = (t1 - t0) / N

        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(N):
            fl_op()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        f_ns = (t1 - t0) / N

        print(
            f"{op_name:20s}  CUDA: {c_ns:.0f} ns  flagos: {f_ns:.0f} ns  Δ: {c_ns - f_ns:.0f} ns ({(c_ns - f_ns) / 1000:.2f} µs)"
        )


def bench_cudagetdevice_direct():
    """Directly measure cudaGetDevice overhead via a minimal CUDA op that does nothing."""
    # torch.cuda.current_device() internally calls cudaGetDevice
    N = 100_000

    # Warmup
    for _ in range(1000):
        torch.cuda.current_device()

    t0 = time.perf_counter_ns()
    for _ in range(N):
        torch.cuda.current_device()
    t1 = time.perf_counter_ns()
    ns_per_call = (t1 - t0) / N
    print("\n=== torch.cuda.current_device() (wraps cudaGetDevice) ===")
    print(f"{ns_per_call:.0f} ns/call ({ns_per_call / 1000:.2f} µs)")


if __name__ == "__main__":
    bench_device_guard_overhead()
    bench_cudagetdevice_direct()
