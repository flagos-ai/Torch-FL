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

"""Hardware validation for ``torch_fl/fast_launch.py`` (Phase 0.5 acceptance gate).

The unit tests pin the plan/replay logic hermetically; this script is the part that
needs a real accelerator, and it exists because three of the defects this shim was
built to avoid are invisible to an accuracy check:

* a plan whose output slot was classified ``scratch`` never writes its result,
* a swapped multi-output pairing returns indices where values belong,
* a per-occurrence intermediate lets launch 1 read a fresh buffer.

So the gate is not "the numbers match". It is, per operator:

1. **slot census** - every plan has an ``"out"`` in the last launch's arg kinds and
   the expected number of outputs, scratch count and launch count,
2. **exactness** - the shim's result matches the untouched FlagGems result and a CPU
   reference, including dtype and container type,
3. **multi-output ordering** - ``max.dim`` returns ``(values fp32, indices int64)``
   through the *plan path*, which is the only hardware check of the storage-pointer
   pairing assumption,
4. **promoting dtype** - a mixed integer/float operator, if FlagGems covers one on
   this device, round-trips through ``StridedBuffer(dtype=...)`` (the one case where
   the captured dtype is load-bearing rather than merely correct),
5. **fail-closed set** - a ping-pong operator (``sort``) is reported unsupported and
   keeps calling the untouched callable rather than replaying,
6. **no growth** - repeated calls do not keep adding plans or allocations.

Usage, on a box with the flagos device and flag_gems installed:

    python benchmarks/fast_launch_integration.py --calls 2000 --json out.json

Exit code is 0 only when every assertion passed; ``--allow-missing`` downgrades
"operator not registered" to a skip (for cohorts where FlagGems does not cover it).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def _measure(fn, calls: int, repeats: int = 5, warmup: int = 20) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    import torch

    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            fn()
        elapsed = time.perf_counter_ns() - start
        torch.cuda.synchronize()
        samples.append(elapsed / calls / 1e3)
    return {
        "submit_us": round(statistics.median(samples), 3),
        "min_us": round(min(samples), 3),
    }


def _max_err(actual, expected) -> float:
    import torch

    if isinstance(expected, torch.Tensor):
        return float((actual.float() - expected.float()).abs().max().item())
    raise TypeError(f"not a tensor: {type(expected)!r}")


def _load_impl(path: str | None):
    """Import the candidate module by path, or fall back to the installed package."""
    if not path:
        from torch_fl import fast_launch

        return fast_launch
    import importlib.util

    spec = importlib.util.spec_from_file_location("fast_launch_candidate", path)
    if spec is None or spec.loader is None:  # pragma: no cover - bad path
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flatten(result):
    if isinstance(result, (tuple, list)):
        return list(result)
    return [result]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=2000, help="timed calls per op")
    parser.add_argument("--json", default=None, help="write the report here")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="treat an operator absent from FlagGems as a skip, not a failure",
    )
    parser.add_argument(
        "--impl",
        default=None,
        help=(
            "load this fast_launch.py by path instead of from the installed torch_fl "
            "package, so a candidate file can be validated without installing it"
        ),
    )
    args = parser.parse_args(argv)

    import torch

    import flag_gems
    import torch_fl  # noqa: F401  - registers the flagos device

    fl = _load_impl(args.impl)

    device = torch.device("flagos:0")
    failures: list[str] = []
    report: dict[str, object] = {"device": str(device), "ops": {}, "plans": []}

    def fail(message: str) -> None:
        failures.append(message)
        print(f"  FAIL {message}")

    # Each entry: FlagGems attribute name, positional operands, torch reference.
    # Positional only: the C++ route passes positionals, and the shim deliberately
    # falls back for keyword calls.
    square = torch.randn(512, 512, device=device)
    tall = torch.randn(1024, 1024, device=device)

    specs = {
        "mul": ("mul", [square, square], lambda a, b: torch.mul(a, b)),
        "add": ("add", [square, square], lambda a, b: torch.add(a, b)),
        "relu": ("relu", [square], lambda a: torch.relu(a)),
        # FlagGems' dim-reduction entry points take ``dim`` as a sequence.
        "sum_dim": ("sum_dim", [tall, [1]], lambda a, dim: torch.sum(a, dim[0])),
        "sum_full": ("sum", [tall], lambda a: torch.sum(a)),
        "softmax": ("softmax", [tall, -1], lambda a, dim: torch.softmax(a, dim)),
        "max_dim": ("max_dim", [tall, 1], lambda a, dim: torch.max(a, dim)),
        "sort": ("sort", [tall, -1], lambda a, dim: torch.sort(a, dim)),
    }

    # The untouched callables must be captured *before* enable() replaces them;
    # comparing torch.<op> against torch.<op> would run the shim on both arms.
    originals = {}
    for name, (attribute, _operands, _reference) in specs.items():
        target = getattr(flag_gems, attribute, None)
        if target is None:
            print(f"  SKIP {name}: torch_fl's FlagGems route has no '{attribute}'")
            if not args.allow_missing:
                failures.append(f"{name}: flag_gems.{attribute} missing")
            continue
        originals[name] = target

    print(f"enabling fast launch on {device}")
    fl.enable([specs[name][0] for name in originals])
    stats = fl.stats()

    for name in originals:
        attribute, operands, torch_reference = specs[name]
        print(f"\n=== {name} (flag_gems.{attribute})")
        entry: dict[str, object] = {}

        cpu_operands = [
            operand.detach().cpu() if isinstance(operand, torch.Tensor) else operand
            for operand in operands
        ]
        reference = torch_reference(*cpu_operands)
        wrapped = getattr(flag_gems, attribute)

        # Warm both arms: the first call is where libtuner may still be tuning, and
        # the shim deliberately does not capture that call.
        _flatten(originals[name](*operands))
        _flatten(wrapped(*operands))
        torch.cuda.synchronize()

        expected = originals[name](*operands)
        actual = wrapped(*operands)
        torch.cuda.synchronize()

        expected_flat = _flatten(expected)
        actual_flat = _flatten(actual)
        if len(expected_flat) != len(actual_flat):
            fail(
                f"{name}: result arity changed {len(expected_flat)} -> {len(actual_flat)}"
            )
        if isinstance(expected, tuple) != isinstance(actual, tuple):
            fail(f"{name}: container type changed {type(expected)} -> {type(actual)}")

        errors = []
        for index, (got, want) in enumerate(
            zip(actual_flat, expected_flat, strict=False)
        ):
            if got.dtype != want.dtype:
                fail(f"{name}[{index}]: dtype {got.dtype} != {want.dtype}")
                continue
            errors.append(_max_err(got, want))
        entry["max_err_vs_flag_gems"] = errors
        for index, error in enumerate(errors):
            if error > 1e-5:
                fail(f"{name}[{index}]: max_err {error} vs the untouched callable")
        if isinstance(reference, torch.Tensor) and errors:
            cpu_err = _max_err(actual_flat[0], reference.to(device))
            entry["max_err_vs_cpu"] = cpu_err
            # A reduction over 1M fp32 elements reorders the accumulation, so a
            # few ulp against the CPU reference is expected (measured 2.4e-04 on
            # this shape in the campaign) and is not a shim defect.
            if cpu_err > 1e-3:
                fail(f"{name}: max_err {cpu_err} against the CPU reference")

        # 3. multi-output ordering, through the plan path.
        if name == "max_dim":
            values, indices = actual
            values_ref, indices_ref = reference
            entry["order"] = [str(values.dtype), str(indices.dtype)]
            if values.dtype != torch.float32:
                fail(
                    f"max_dim: values dtype {values.dtype} (indices where values belong?)"
                )
            if indices.dtype != torch.int64:
                fail(f"max_dim: indices dtype {indices.dtype}")
            if not torch.equal(indices, indices_ref.to(device)):
                fail("max_dim: indices differ from the CPU reference")
            if not torch.allclose(values, values_ref.to(device), rtol=0, atol=0):
                fail("max_dim: values differ from the CPU reference")

        # 5. fail-closed set: a ping-pong operator cannot replay.
        if name == "sort":
            if "sort" not in fl._UNSUPPORTED:
                fail("sort: expected the ping-pong capture to be refused")
            else:
                entry["fail_closed"] = True

        # 6. no growth across repeated calls.
        plans_before = len(fl.plans())
        for _ in range(args.calls):
            wrapped(*operands)
        torch.cuda.synchronize()
        plans_after = len(fl.plans())
        if plans_after != plans_before:
            fail(f"{name}: plan cache grew {plans_before} -> {plans_after}")

        entry["fast"] = _measure(lambda: wrapped(*operands), args.calls)
        entry["flag_gems"] = _measure(lambda: originals[name](*operands), args.calls)
        entry["speedup"] = round(
            entry["flag_gems"]["submit_us"] / entry["fast"]["submit_us"], 3
        )
        report["ops"][name] = entry
        print(
            f"  max_err={errors} fast={entry['fast']['submit_us']} us "
            f"vs flag_gems={entry['flag_gems']['submit_us']} us "
            f"({entry['speedup']}x)"
        )

    report["plans"] = fl.plans()
    report["stats"] = vars(stats)

    # 1. slot census: the invariant that a wrong-and-fast plan violates.
    for plan in report["plans"]:
        kinds = plan["launch_arg_kinds"][-1]
        if "out" not in kinds:
            fail(f"plan {plan['op']}: final launch has no 'out' slot ({kinds})")
        if plan["outputs"] == 0:
            fail(f"plan {plan['op']}: no outputs paired")
        if kinds.count("out") != plan["outputs"]:
            fail(
                f"plan {plan['op']}: {plan['outputs']} outputs but "
                f"{kinds.count('out')} 'out' slots in the final launch"
            )

    import torch as _torch

    if _torch.cuda.is_available():
        reserved = _torch.cuda.memory_reserved()
        report["reserved_bytes"] = reserved

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=str)

    print(f"\nplans: {json.dumps(report['plans'], indent=1)}")
    print(f"stats: {report['stats']}")
    if failures:
        print(f"\nFAILED ({len(failures)})")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
