#!/usr/bin/env python3
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

"""Run the TileOPs route checks without pytest.

``tests/integration/ops/test_tileops_generated.py`` is the real suite and should be
preferred wherever pytest is installed. This script drives equivalent assertions
with plain python, for hosts where it is not: the H800 dev box has no direct
network route, and while an HTTP proxy makes pytest installable, that proxy is not
available everywhere.

    python scripts/run_tileops_checks.py [--full] [--filter SUBSTR]

``--full`` uses the manifest workload shapes instead of a small shape. Each
distinct shape costs a full TileLang compile (~4 s), so the default is small.
"""

from __future__ import annotations

import argparse
import time

import torch

import torch_fl  # noqa: F401  (registers the flagos device)
from torch_fl import tileops_backend
from torch_fl.generated.tileops_routes import ROUTES, WORKLOADS

TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (5e-2, 5e-2),
    torch.float32: (1e-4, 1e-5),
}

SMALL_SHAPE = (64, 32)


def aten_ref(overload: str):
    """Resolve an overload to a callable, to be used with CPU tensors.

    The TileOPs impl is bound on the flagos key, so a reference computed there
    would re-enter the code under test.
    """
    base, _, ov = overload.partition(".")
    packet = getattr(torch.ops.aten, base)
    return getattr(packet, ov) if ov else packet


def check_route(overload, recipe, module, cls_name, dtypes, extra, full: bool):
    shape = WORKLOADS.get(overload, SMALL_SHAPE) if full else SMALL_SHAPE
    dtype = getattr(torch, dtypes[0])

    impl = tileops_backend.build_impl(recipe, module, cls_name, dtypes, extra, overload)
    if impl is None:
        return "skip", "not constructible"

    cpu_args = tileops_backend.sample_inputs(
        recipe, shape, dtype, device="cpu", overload=overload
    )
    dev_args = tuple(
        a.to("flagos") if isinstance(a, torch.Tensor) else a for a in cpu_args
    )

    got = impl(*dev_args)
    want = aten_ref(overload)(*cpu_args)
    if isinstance(got, (tuple, list)):
        got = got[0]
    if isinstance(want, (tuple, list)):
        want = want[0]

    if dtype in TOL:
        rtol, atol = TOL[dtype]
        torch.testing.assert_close(
            got.cpu().float(), want.float(), rtol=rtol, atol=atol
        )
    elif not torch.equal(got.cpu(), want):
        raise AssertionError(f"integer result not bit-exact for {overload}")
    return "pass", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use manifest workload shapes")
    ap.add_argument(
        "--filter", default=None, help="only routes containing this substring"
    )
    args = ap.parse_args()

    if not tileops_backend.is_tileops_available():
        print("TileOPs unavailable or host is not SM90 -- nothing to check")
        return 0

    registered = tileops_backend.enable_tileops_for_flagos()
    again = tileops_backend.enable_tileops_for_flagos()
    assert registered == again, f"enable not idempotent: {registered} then {again}"
    bound = tileops_backend.registered_ops()

    selected = [r for r in ROUTES if args.filter is None or args.filter in r[0]]
    passed, skipped, failures = 0, 0, []
    t0 = time.perf_counter()

    for route in selected:
        overload = route[0]
        assert overload in bound, f"{overload} in ROUTES but not registered"
        try:
            status, note = check_route(*route, full=args.full)
        except Exception as exc:  # noqa: BLE001  (report, do not abort the sweep)
            failures.append((overload, route[3], f"{type(exc).__name__}: {exc}"))
            print(
                f"  FAIL {overload:<28} {type(exc).__name__}: {str(exc)[:80]}",
                flush=True,
            )
            continue
        if status == "skip":
            skipped += 1
            print(f"  SKIP {overload:<28} {note}", flush=True)
        else:
            passed += 1
            print(f"  PASS {overload:<28} {route[4][0]}", flush=True)

    # An unsupported dtype must fall back to aten rather than raise.
    guard = torch.ones(64, device="flagos", dtype=torch.int32)
    assert torch.equal(torch.relu(guard).cpu(), torch.ones(64, dtype=torch.int32)), (
        "int32 relu did not fall back correctly"
    )

    elapsed = time.perf_counter() - t0
    print(
        f"\nregistered={registered} checked={len(selected)} passed={passed} "
        f"skipped={skipped} failed={len(failures)}  ({elapsed:.0f}s)"
    )
    print("dtype-guard fallback: OK")
    if failures:
        print("\nfailures:")
        for overload, cls_name, msg in failures:
            print(f"  {overload:<28} {cls_name:<22} {msg[:150]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
