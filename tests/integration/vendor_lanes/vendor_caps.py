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

"""The vocabulary the per-platform vendor-lane suite is written in.

Every domain in this package -- the SDPA lanes, and the dtype escape, the
device-context guard and the pinned routes that follow -- used to be a file
named after the chip that needed it (``test_ascend_sdpa_mask_bias.py``,
``test_gcu_sdpa_mask.py``, ...). Each such file stated three things: the cases
the contract is about, the lane or answer the vendor was measured to produce,
and the account of why. Two files for one operator meant the third chip to
reach it authored a third file and inherited none of the coverage.

The split this module draws is the same one ``tests/integration/flaggems`` and
``tests/integration/runtime`` draw. The *case* is the call; it is written once,
in the domain's own table. The *row* is what was measured on one chip: the lane
it takes, the numbers it holds to, and the account of the mask spelling or
dispatch position that makes those numbers the right answer. A chip that starts
taking the lane adds a row and gets every case the row's peers run.

Nothing here is a platform name. ``accel_label()`` reads the accelerator from
the conf the wheel resolved, so the row a test reads is the row for the wheel it
is inside rather than for the machine someone believed it was on.

The shared test engine is :func:`run_case`, which drives one
:class:`Case` against one :class:`LaneRow`: it opens the grad context the row
states, records the leaves the call dispatched to when the row says the lane is
observable, compares the answer against the row's reference, and, where the case
is about a mask that must not change the answer, checks the baseline as well.
"""

from __future__ import annotations

import contextlib
from typing import Callable, NamedTuple

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

#: The device string ``[flagos dispatch]`` prints for the flagos device.
DEVICE = "flagos:0"

#: ``{domain name: module}`` -- filled by each domain module at import time, so
#: the conftest's summary line names the rows this platform carries without a
#: second copy of the tables.
_DOMAINS: dict[str, object] = {}


def register_domain(name: str, module: object) -> None:
    """Declare one domain's table to the package summary line."""
    _DOMAINS[name] = module


def declared_rows() -> list[str]:
    """One line per domain, naming this platform's row (or its absence)."""
    accel = accel_label()
    rows: list[str] = []
    for name, module in sorted(_DOMAINS.items()):
        describe = getattr(module, "declared_lines", None)
        if describe is None:
            continue
        rows.append(f"{name}: {describe(accel)}")
    return rows


def accel_label() -> str:
    """The accelerator the active conf names, from its basename.

    Read from ``backends_<platform>.conf`` rather than from ``detect_platform()``
    (or any vendor probe): it is the wheel's own statement of what it was built
    for, and it is the key every table here is stored under. A conf with a
    user-chosen path has no row, and the domains skip it by name.
    """
    from pathlib import Path

    import torch_fl

    name = Path(torch_fl.backend_config_path()).name
    if name.startswith("backends_") and name.endswith(".conf"):
        return name[len("backends_") : -len(".conf")]
    return ""


class RouteRecorder(TorchDispatchMode):
    """Which leaves the measured call dispatched to.

    A mode sits below the autograd keys and above every backend key, so what it
    sees is the composite's decomposition rather than a single vendor leaf --
    which is what makes "the fused lane ran" and "the math lane ran" readable
    from the same record. ``rules`` maps a leaf-name suffix to the lane name a
    row calls it, in priority order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        key = str(func)
        self.counts[key] = self.counts.get(key, 0) + 1
        return func(*args, **(kwargs or {}))

    def lane(self, rules: tuple[tuple[str, str], ...]) -> str:
        for suffix, lane in rules:
            if any(name.endswith(suffix) for name in self.counts):
                return lane
        return "other"

    def describe(self) -> str:
        """The counts, sorted, for a failure message."""
        return f"{dict(sorted(self.counts.items()))}"


class Case(NamedTuple):
    """One call the domain's contract is about.

    ``build`` returns a dict with at least:

        ``run``       a zero-argument callable returning the device result
        ``reference`` the expected value, on the host

    and optionally:

        ``baseline``  a zero-argument callable for the same operands with the
                      feature under test removed (no mask; no bias), for the
                      cases whose claim is that the answer did not change

    ``build`` is called once per test, so anything expensive or anything that
    has to exist on the device is built there rather than at import.
    ``lane`` overrides the row's default lane for this case. ``equals_baseline``
    and ``differs_from_baseline`` choose the bitwise baseline comparison, and
    ``deviation_floor`` / ``deviation_ceiling`` replace the row's tolerance with
    a bound on ``max|d|`` for the cases that record a vendor deviation rather
    than an agreement.
    """

    tag: str
    clause: str
    build: Callable[[], dict]
    lane: str | None = None
    raises: str | None = None
    grad: bool = False
    equals_baseline: bool = False
    differs_from_baseline: bool = False
    deviation_floor: float | None = None
    deviation_ceiling: float | None = None


class LaneRow(NamedTuple):
    """What was measured on one chip, and how its cases have to be driven.

    ``observe`` says whether the lane can be read off the dispatch at all on
    this vendor -- GCU's fused lane is a leaf the mode sees, while Ascend's
    ACLNN path reached through the composite is asserted by the answer alone.
    ``rules`` is the leaf-suffix-to-lane map used when it can. ``no_grad`` says
    whether the cases run under ``torch.no_grad()``, which is the mode the
    vendor lane was measured in; a case may still ask for grad itself.

    ``revalidated`` is ``False`` for a row whose numbers were inherited from the
    chip that did measure them; the gap is recorded here and in
    ``docs/reference/operator-support.md`` rather than papered over.
    """

    lane: str
    observe: bool
    rules: tuple[tuple[str, str], ...]
    compare: Callable[[torch.Tensor, torch.Tensor], None]
    no_grad: bool
    cases: tuple[Case, ...]
    measured_on: str
    revalidated: bool = True


def run_case(row: LaneRow, case: Case) -> None:
    """Drive one case against one row, asserting everything the case claims."""
    argv = case.build()
    enabled = case.grad or not row.no_grad

    def call() -> torch.Tensor:
        return argv["run"]()

    if case.raises is not None:
        with pytest.raises(RuntimeError, match=case.raises):
            with torch.set_grad_enabled(enabled):
                call()
        return

    recorder = RouteRecorder() if row.observe else None
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.set_grad_enabled(enabled))
        if recorder is not None:
            stack.enter_context(recorder)
        out = call()
    expected_lane = case.lane or row.lane
    if recorder is not None:
        assert recorder.lane(row.rules) == expected_lane, (
            f"{case.tag} ({case.clause}): expected the {expected_lane} lane, got "
            f"{recorder.lane(row.rules)}: {recorder.describe()}"
        )

    reference = argv.get("reference")
    if reference is None:
        # A case whose claim is that the call *completes* -- dropout is
        # stochastic, so there is no host number to compare against. The row's
        # ``compare`` still runs, on the output as its own reference, so the
        # device and shape assertions it carries are kept.
        row.compare(out, out)
    elif case.deviation_floor is not None or case.deviation_ceiling is not None:
        deviation = (out.cpu().float() - reference.float()).abs().max().item()
        if case.deviation_floor is not None:
            assert deviation > case.deviation_floor, (
                f"{case.tag} ({case.clause}): max|d| against the host is "
                f"{deviation:.3e}, under the {case.deviation_floor:.3e} floor "
                f"measured on {row.measured_on}"
            )
        if case.deviation_ceiling is not None:
            assert deviation <= case.deviation_ceiling, (
                f"{case.tag} ({case.clause}): max|d| against the host is "
                f"{deviation:.3e}, over the {case.deviation_ceiling:.3e} "
                f"ceiling measured on {row.measured_on}"
            )
    else:
        row.compare(out, reference)

    if case.equals_baseline or case.differs_from_baseline:
        baseline_run = argv["baseline"]
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.set_grad_enabled(enabled))
            if recorder is not None:
                stack.enter_context(recorder)
            baseline = baseline_run()
        if recorder is not None and row.observe:
            assert recorder.lane(row.rules) == expected_lane, (
                f"{case.tag}: the baseline call took the "
                f"{recorder.lane(row.rules)} lane: {recorder.describe()}"
            )
        same = torch.equal(out.cpu(), baseline.cpu())
        if case.equals_baseline:
            assert same, (
                f"{case.tag} ({case.clause}): the answer moved by "
                f"{(out.cpu().float() - baseline.cpu().float()).abs().max().item():.3e} "
                "where the feature under test is supposed to be the identity"
            )
        else:
            assert not same, (
                f"{case.tag} ({case.clause}): the answer did not move, so the "
                "feature under test was not read"
            )
