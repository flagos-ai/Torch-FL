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
Micro-benchmark: synchronize() call overhead comparison.

Results on H800 (100k calls each):
    flagos.synchronize():            1.83 us/call
    torch.cuda.synchronize():        5.17 us/call
    torch.cuda.synchronize(device):  4.43 us/call
    Delta (cuda - flagos):           3.34 us

This ~3.3us/call difference explains why profiler mode (TorchDispatchMode with
per-op sync) shows flagos appearing faster than native CUDA. Each profiled op
includes one sync in the measured interval, so CUDA baseline gets +3.3us/op
systematic overhead. At ~3300 ops/token, that's ~11ms/token of measurement bias.

Usage:
    CUDA_VISIBLE_DEVICES=2 FLAGOS_BACKEND_CONFIG=... python tests/perf/sync_cuda.py
"""

import argparse
import time

import torch
import torch_fl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100000, help="Number of calls")
    args = parser.parse_args()
    N = args.n

    device_flagos = "flagos:0"
    device_cuda = "cuda:0"

    # Warm up both devices
    _ = torch.randn(16, 16, device=device_flagos)
    _ = torch.randn(16, 16, device=device_cuda)
    torch_fl.flagos.synchronize()
    torch.cuda.synchronize()

    # --- flagos.synchronize ---
    torch_fl.flagos.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        torch_fl.flagos.synchronize()
    flagos_cost = (time.perf_counter() - t0) / N * 1e6

    # --- torch.cuda.synchronize() ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        torch.cuda.synchronize()
    cuda_cost = (time.perf_counter() - t0) / N * 1e6

    # --- torch.cuda.synchronize(device) ---
    torch.cuda.synchronize(device_cuda)
    t0 = time.perf_counter()
    for _ in range(N):
        torch.cuda.synchronize(device_cuda)
    cuda_dev_cost = (time.perf_counter() - t0) / N * 1e6

    # --- perf_counter baseline ---
    t0 = time.perf_counter()
    for _ in range(N):
        time.perf_counter()
    counter_cost = (time.perf_counter() - t0) / N * 1e6

    print(f"=== Synchronize Micro-Benchmark ({N} calls) ===")
    print(f"flagos.synchronize():            {flagos_cost:.2f} us/call")
    print(f"torch.cuda.synchronize():        {cuda_cost:.2f} us/call")
    print(f"torch.cuda.synchronize(device):  {cuda_dev_cost:.2f} us/call")
    print(f"Delta (cuda - flagos):           {cuda_cost - flagos_cost:.2f} us")
    print(f"time.perf_counter() overhead:    {counter_cost:.2f} us/call")


if __name__ == "__main__":
    main()
