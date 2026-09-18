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

"""A/B `scaled_dot_product_attention` on DCU across the routes it can take.

PR #347 gave the op a FlagGems route on MetaX, where the boxing path had no
reachable fused kernel. DCU is the other CUDA-boxing platform where the same
whole-op override (`csrc/aten/sdp_choice_stub.cc`) is compiled in, so the
question of whether to follow MetaX is a conf line. This is the measurement
behind the answer, which is no: `backends_dcu.conf` keeps the `cuda` route and
the vendor CUTLASS flash adapter is 2.5x faster than FlagGems at both shapes.

Three arms, all through the same `torch.ops.aten.scaled_dot_product_attention`:

    math      FLAGOS_OP_scaled_dot_product_attention=cuda + FLAGOS_DCU_SDPA_FLASH=0
              (DTK's own math decomposition)
    cuda      shipped conf (DTK's selector, which the shim points at its
              CUTLASS flash adapter)
    flaggems  FLAGOS_OP_scaled_dot_product_attention=flaggems

One process per arm. The route is latched when the vendor conf is loaded, i.e.
during `import torch_fl`, so the environment has to be set before it and cannot
be changed inside a running process.

Two details are load-bearing, and both are mistakes this harness was written
after making them:

* `import torch_fl` must precede `import torch`. The DCU backend dlopens DTK's
  device libraries from `torch_fl/__init__.py`; letting torch's PrivateUse1
  autoload pull torch_fl in afterwards leaves device init failing with
  "Cannot initialize CUDA without ATen_cuda library" even though the kernels
  registered. Each arm therefore runs as a fresh subprocess.
* Every timing window closes with `torch.cuda.synchronize(device)`, never the
  no-argument form. The no-argument form synchronizes
  `torch.cuda.current_device()`, which is `0` however the tensors were placed,
  so on a non-zero card it returns immediately and the "ms/iter" is a submit
  time. The two entries this harness corrected in
  `docs/reference/operator-support.md` were measured with it. `--sync-check`
  prints all four windows side by side so the difference is visible rather
  than asserted.

Run, on a host whose `flagos` backend is the Hygon DCU build:

    python tests/manual/dcu_sdpa_ab.py --device flagos:7
    python tests/manual/dcu_sdpa_ab.py --routes flaggems --sync-check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Qwen-Image-2512 joint attention: the shape PR #347's envelope is drawn on, and
# the per-head shape its 1664x928 pass runs.
SHAPES = {
    "joint-4114": (1, 24, 4114, 128),
    "joint-12576": (1, 24, 12576, 128),
}
ROUTES = ("math", "cuda", "flaggems")


def _route_env(route):
    """The two env vars each arm differs by; `math` is `cuda` with flash off."""
    env = {
        "FLAGOS_OP_scaled_dot_product_attention": "cuda" if route == "math" else route
    }
    if route == "math":
        env["FLAGOS_DCU_SDPA_FLASH"] = "0"
    return env


def _median(fn, sync, iters, warmup=2):
    for _ in range(warmup):
        fn()
        sync()
    times = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return times[len(times) // 2], times[0]


def run_arm(args):
    """One route, one process. Prints one JSON line per shape."""
    for key, value in _route_env(args.route).items():
        os.environ[key] = value

    import torch_fl  # noqa: F401  - must precede `import torch`
    import torch

    dev = torch.device(args.device)
    torch.manual_seed(0)

    if args.sync_check:
        # Deliberately before any set_device: the no-argument synchronize()
        # follows the current device, and it is the untouched process default
        # that makes the mistake reproducible.
        _print_sync_windows(torch, dev, args.idle_device)
        return 0

    torch.cuda.set_device(dev)
    import flag_gems

    report = []
    for name in args.shapes:
        b, h, s, d = SHAPES[name]
        # Built on the host: the device randn path is routed through the conf and
        # does not initialise on this stack.
        q = torch.randn(b, h, s, d, dtype=torch.bfloat16).to(dev)
        k = torch.randn(b, h, s, d, dtype=torch.bfloat16).to(dev)
        v = torch.randn(b, h, s, d, dtype=torch.bfloat16).to(dev)

        def call(q=q, k=k, v=v):
            # scale/enable_gqa are keyword-only; the defaults are the argument
            # shape PR #347's envelope is drawn on.
            return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        median, best = _median(call, lambda: torch.cuda.synchronize(dev), args.iters)
        routed = call()
        # The attribution: the FlagGems kernel the stub calls by name. Bit
        # identity means the arm took the route it was asked for, so a latency
        # difference is the kernel's and not a silent fall-through.
        direct = flag_gems.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(dev)
        report.append(
            {
                "shape": name,
                "route": args.route,
                "median_ms": round(median, 2),
                "best_ms": round(best, 2),
                "bit_identical_to_direct_flag_gems": bool(torch.equal(routed, direct)),
                "maxdiff_vs_direct_flag_gems": float(
                    (routed.float() - direct.float()).abs().max().item()
                ),
            }
        )
        del q, k, v, routed, direct

    for row in report:
        print("RESULT " + json.dumps(row, sort_keys=True))
    sys.stdout.flush()
    return 0


def _print_sync_windows(torch, dev, idle):
    """Show which device the no-argument `synchronize()` actually waits on."""
    b, h, s, d = SHAPES["joint-12576"]
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16).to(dev)
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16).to(dev)
    v = torch.randn(b, h, s, d, dtype=torch.bfloat16).to(dev)

    def call():
        return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

    idle_dev = torch.device(idle)
    default_dev = torch.cuda.current_device()
    no_arg, _ = _median(call, torch.cuda.synchronize, 5)
    on_dev, _ = _median(call, lambda: torch.cuda.synchronize(dev), 5)
    on_idle, _ = _median(call, lambda: torch.cuda.synchronize(idle_dev), 5)
    torch.cuda.set_device(dev)
    on_current, _ = _median(call, torch.cuda.synchronize, 5)
    print(f"tensors_on={dev} current_device_at_start={default_dev}")
    print(
        f"SYNC no-arg(current={default_dev})={no_arg:.2f}ms "
        f"sync({dev})={on_dev:.2f}ms sync({idle})={on_idle:.2f}ms "
        f"no-arg(after set_device)={on_current:.2f}ms"
    )
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=",".join(ROUTES))
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--device", default="flagos:7")
    ap.add_argument("--idle-device", default="flagos:0")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sync-check", action="store_true")
    ap.add_argument("--route", default=None, choices=ROUTES, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.shapes = [s for s in args.shapes.split(",") if s]

    if args.route is not None:
        return run_arm(args)

    env = dict(os.environ)
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    rows = []
    for route in args.routes.split(","):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--route",
            route,
            "--device",
            args.device,
            "--idle-device",
            args.idle_device,
            "--shapes",
            ",".join(args.shapes),
            "--iters",
            str(args.iters),
        ]
        if args.sync_check:
            cmd.append("--sync-check")
        print(f"--- arm {route} on {args.device} ---", flush=True)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.startswith(("RESULT ", "SYNC ", "tensors_on=")):
                print(line, flush=True)
                if line.startswith("RESULT "):
                    rows.append(json.loads(line[len("RESULT ") :]))
        if proc.returncode != 0:
            print(f"    arm {route} exited {proc.returncode}", flush=True)
            print(proc.stderr[-2000:], flush=True)

    if len(rows) > 1:
        print("\n=== median ms/iter ===")
        by_shape = {}
        for row in rows:
            by_shape.setdefault(row["shape"], {})[row["route"]] = row["median_ms"]
        header = "shape".ljust(14) + "".join(r.rjust(12) for r in ROUTES)
        print(header)
        for shape, routes in by_shape.items():
            cells = "".join(
                (f"{routes[r]:.1f}" if r in routes else "-").rjust(12) for r in ROUTES
            )
            print(shape.ljust(14) + cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
