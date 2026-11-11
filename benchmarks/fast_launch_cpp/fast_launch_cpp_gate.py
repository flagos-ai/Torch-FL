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

"""Hardware gate for the C++ replay, with a vendor arm for every operator.

The three arms run in separate processes. For the wrapper and cpp arms this is required:
the plugin resolves a FlagGems callable on its first dispatch and caches it for the life
of the process, so enabling the fast path after that dispatch would leave the cached
original in place and measure nothing. The vendor arm runs the same operators on the
vendor device and never imports the plugin.

    python benchmarks/fast_launch_cpp/fast_launch_cpp_gate.py --arm vendor
    python benchmarks/fast_launch_cpp/fast_launch_cpp_gate.py --arm wrapper
    python benchmarks/fast_launch_cpp/fast_launch_cpp_gate.py --arm cpp

Every arm measures the same operators at the same shapes and checks its result against a
CPU reference, and each row is emitted as a single ``RESULT`` line so the three arms can
be collated into one table without reading prose. The vendor arm is the reference the
whole design is judged against: it is the operator a user would get without FlagGems.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Operator table: name, torch call, input shape, and whether the replay is expected to
# take it. The expected-declined entries are the fallback contract under test.
OPS = (
    # Only the operators in ``fast_launch_cpp.VERIFIED_OPS`` are expected to install a
    # plan; the rest must decline and keep their untouched cost. Every operator is also
    # measured on the vendor device, so each row can be read against the operator it is
    # ultimately compared with.
    ("mul", lambda t: t["a"] * t["b"], (512, 512), True),
    ("softmax", lambda t: t["a"].softmax(dim=-1), (1024, 1024), True),
    ("add", lambda t: t["a"] + t["b"], (512, 512), False),
    ("relu", lambda t: t["a"].relu(), (512, 512), False),
    ("sum_dim", lambda t: t["a"].sum(dim=1), (1024, 1024), False),
    ("max_dim", lambda t: t["a"].max(dim=1), (1024, 1024), False),
    ("sort", lambda t: t["a"].sort(), (1024, 1024), False),
)


def measure(fn, calls=200, repeats=5, warmup=20):
    for _ in range(warmup):
        fn()
    import torch

    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            fn()
        elapsed = time.perf_counter_ns() - start
        torch.cuda.synchronize()
        values.append(elapsed / calls / 1e3)
    return round(statistics.median(values), 3)


#: Relative tolerance for a float comparison. The reference is computed on the CPU, so
#: a reduction legitimately differs in the last bits; anything larger is a real error.
RTOL = 1e-4


def _max_rel_error(actual, expected):
    difference = (actual.cpu().float() - expected.float()).abs().max()
    scale = expected.float().abs().max().clamp(min=1.0)
    return float(difference / scale)


def errors(name, actual, expected):
    """Relative error per result, or ``None`` for an entry that has no value to compare."""
    import torch

    if not isinstance(actual, tuple):
        return [_max_rel_error(actual, expected)]
    # ``max(dim)`` returns (values, indices) and ``sort`` returns (values, indices):
    # the index result legitimately differs between implementations when values tie,
    # so only the value result is compared.
    value = _max_rel_error(actual[0], expected[0])
    if torch.is_floating_point(actual[0]):
        return [value, None]
    return [value, None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("vendor", "wrapper", "cpp", "plugin"), required=True
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    # The vendor arm must not import the plugin: it measures the operator a user would
    # get without FlagGems, which is the reference the replay is judged against.
    if args.arm != "vendor":
        import torch_fl  # noqa: F401 - registers the device

    backend = None
    if args.arm in ("cpp", "plugin"):
        import fast_launch_cpp

        # Enable before the first dispatch: the wrapper performs the capture, and the
        # plugin's C++ caller replays from then on.
        backend = fast_launch_cpp.enable(
            [name for name, *_ in OPS], plugin=args.arm == "plugin"
        )

    default_device = "cuda:0" if args.arm == "vendor" else "flagos:0"
    device = torch.device(args.device or default_device)
    failures = []
    for name, call, shape, replayable in OPS:
        tensors = {
            "a": torch.randn(*shape, device=device),
            "b": torch.randn(*shape, device=device),
        }
        expected = {}
        for key, value in tensors.items():
            expected[key] = value.cpu()
        reference = call({key: value.cpu() for key, value in tensors.items()})
        value = measure(lambda: call(tensors))
        actual = call(tensors)
        torch.cuda.synchronize()
        errs = errors(name, actual, reference)
        shown = ["n/a" if err is None else f"{err:.2e}" for err in errs]
        print(
            f"  {name:10s} {value:8.3f} us  rel_err={shown}"
            f"  replay={'expected' if replayable else 'declined'}"
        )
        print(
            "RESULT "
            + json.dumps(
                {
                    "arm": args.arm,
                    "op": name,
                    "device": str(device),
                    "us": value,
                    "rel_err": errs,
                    "expected_replay": replayable,
                }
            )
        )
        if any(err is not None and err > RTOL for err in errs):
            failures.append(f"{name}: wrong result")

    if backend is not None:
        for name, plan in backend.plans.items():
            print(f"\nplan {name}: {plan.census()}")
        print(f"declined: {backend.reasons}")
        print(f"stats: {backend.stats}")
        missing = [
            name
            for name, _, _, replayable in OPS
            if replayable and name not in backend.plans
        ]
        if missing:
            failures.append(f"replay did not install for {missing}")
        unexpected = [
            name
            for name, _, _, replayable in OPS
            if not replayable and name in backend.plans
        ]
        if unexpected:
            failures.append(f"replay erroneously installed for {unexpected}")

    print("\nPASS" if not failures else f"\nFAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
