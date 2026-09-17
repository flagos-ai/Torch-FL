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
FlagGems dispatch-overhead microbenchmark -- worker process.

Measures per-op HOST cost of the Python FlagGems path vs the vendor (cuda)
path, to quantify how much a hypothetical C++ (kFlagOs) dispatch could save.

Two timings per op:
  * submit  -- issue N ops back-to-back, synchronize once at the end. With the
               async queue hiding kernel time, this is dominated by host-side
               dispatch cost (GIL + pybind tensor/arg marshalling + CPython
               frame for the Python path; a thin C++ dispatcher for cuda).
  * e2e     -- synchronize after every op: full per-op wall time incl. kernel.

The submit-time gap (flaggems_python - cuda) is the realistic ceiling on what
C++ dispatch could recover, since both then launch the same class of GPU work.

Routing is fixed at import, so this worker benchmarks ONE backend and prints a
JSON blob. The driver (flaggems_dispatch_bench.py) runs it once per backend.

Usage (normally invoked by the driver):
    FLAGOS_FORCE_BACKEND=flaggems python flaggems_dispatch_overhead.py --backend flaggems
    python flaggems_dispatch_overhead.py --backend cuda
"""

import argparse
import json
import sys
import time

import torch_fl
import torch

DEVICE = "flagos:0"
_sync = torch_fl.flagos.synchronize


def _t(*shape):
    """Random tensor built on CPU then moved to flagos (avoids the RNG shim)."""
    return torch.randn(*shape).to(DEVICE)


# Each entry: (label, category, build-inputs -> fn, expected-flaggems-routing).
# Sizes chosen to span host-bound (small elementwise) -> compute-bound (mm).
def _ops():
    small, big = 512, 4096
    return [
        (
            "add[512x512]",
            "elementwise",
            lambda: (_t(small, small), _t(small, small)),
            lambda a, b: torch.add(a, b),
        ),
        (
            "add[4096x4096]",
            "elementwise",
            lambda: (_t(big, big), _t(big, big)),
            lambda a, b: torch.add(a, b),
        ),
        (
            "mul[512x512]",
            "elementwise",
            lambda: (_t(small, small), _t(small, small)),
            lambda a, b: torch.mul(a, b),
        ),
        (
            "abs[512x512]",
            "elementwise",
            lambda: (_t(small, small),),
            lambda a: torch.abs(a),
        ),
        (
            "neg[512x512]",
            "elementwise",
            lambda: (_t(small, small),),
            lambda a: torch.neg(a),
        ),
        (
            "sum.dim[1024x1024]",
            "reduction",
            lambda: (_t(1024, 1024),),
            lambda a: torch.sum(a, dim=1),
        ),
        (
            "softmax[1024x1024]",
            "reduction",
            lambda: (_t(1024, 1024),),
            lambda a: torch.softmax(a, dim=-1),
        ),
        (
            "mm[512x512]",
            "matmul",
            lambda: (_t(small, small), _t(small, small)),
            lambda a, b: torch.mm(a, b),
        ),
        (
            "mm[4096x4096]",
            "matmul",
            lambda: (_t(big, big), _t(big, big)),
            lambda a, b: torch.mm(a, b),
        ),
    ]


def _bench_one(build, fn, n_submit, n_e2e, warmup):
    args = build()
    for _ in range(warmup):
        fn(*args)
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_submit):
        fn(*args)
    _sync()
    submit_us = (time.perf_counter() - t0) / n_submit * 1e6
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_e2e):
        fn(*args)
        _sync()
    e2e_us = (time.perf_counter() - t0) / n_e2e * 1e6
    return submit_us, e2e_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend", required=True, help="label only; routing is env-driven"
    )
    ap.add_argument("--submit", type=int, default=300)
    ap.add_argument("--e2e", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    results = []
    for label, cat, build, fn in _ops():
        try:
            submit_us, e2e_us = _bench_one(
                build, fn, args.submit, args.e2e, args.warmup
            )
            results.append(
                {"op": label, "category": cat, "submit_us": submit_us, "e2e_us": e2e_us}
            )
        except Exception as e:  # noqa: BLE001
            results.append({"op": label, "category": cat, "error": repr(e)})

    print("BENCH_JSON_START", file=sys.stderr)
    print(json.dumps({"backend": args.backend, "results": results}))


if __name__ == "__main__":
    main()
