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

"""Every external-correlation pop must pass a real address to the vendor API.

A ``DeviceTracer`` pushes the profiler's correlation id onto the vendor's
per-thread external-correlation stack when an ATen op begins and pops it when the
op ends, so that a device record created in between is stamped with the id of the
op that issued it. Both vendor headers that document the pop describe the
out-parameter as optional, and CUPTI's spelling is ``uint64_t *lastId``.

CANN 9.0 does not honour that: ``msptiActivityPopExternalCorrelationId(kind,
nullptr)`` returns ``MSPTI_ERROR_INVALID_PARAMETER`` *without unwinding the
stack*. A tracer that passes a null there and discards the result leaves every
push on the stack for the life of the process, so each activity record is stamped
with the most recently pushed id -- the op that starts after the launch, not the
one that made it. Ascend's tracer did exactly that: ``aten::matmul``'s kernels
carried the ``External id`` of the ``aten::empty`` call that followed them, the
launcher read ``self_device_time_total == 0.0``, and two kernels could collapse
onto a single id. That is issue #425, and it is invisible without hardware --
the pop returns a failure code the call site never read, and every other
assertion in the profiler contract still passed.

So the guard is a text check on the call shape, alongside the behavioural
assertion that now runs for real on the Ascend job
(``test_profiler_device_time_linkage``). It needs no vendor library, no build and
no ``torch_fl``, and is wired into the platform-agnostic CI job.

Run: pytest tests/unit/test_device_tracer_correlation.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILER_DIR = REPO_ROOT / "csrc" / "profiler"

# The vendor entry points, under the two spellings this tree uses: the CUPTI and
# MUPTI shims keep the vendor's own name as a member, and the CANN tracer holds a
# loaded function pointer named for the symbol with the vendor prefix dropped.
# A bare identifier followed by "(" is a call; the same identifier in a
# declaration (`(*ActivityPopExternalCorrelationId)(`), a `dlsym` string, a
# `LOAD_SYM` argument or an initialiser (`pop_external = nullptr`) is not
# followed by an open paren, so it does not match.
_POP_CALL = re.compile(r"\b(?:pop_external|ActivityPopExternalCorrelationId)\s*\(")

# Null literals that would be rejected. `0` is included because it is what a
# C-style call site would reach for; it is written with a negative lookahead so
# that an address-of expression or a named argument cannot match it.
_NULL_ARGUMENT = re.compile(r"^(?:nullptr|NULL|0)$")

# The tracers that delegate their pop to a vendor API. GCU's tracer keeps the
# current correlation in a process-wide global and calls no vendor pop, so it has
# no call site here; a count below this means the scan stopped matching, not that
# a tracer became correct.
_MIN_CALL_SITES = 3


def _call_arguments(source: str, open_paren: int) -> list[str]:
    """Split the argument list whose "(" is at ``open_paren`` at top level."""
    depth = 0
    current: list[str] = []
    arguments: list[str] = []
    for index in range(open_paren, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
            if depth == 1:
                continue
        elif char == ")":
            depth -= 1
            if depth == 0:
                arguments.append("".join(current).strip())
                return arguments
        elif char == "," and depth == 1:
            arguments.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    raise AssertionError(f"unbalanced parentheses from offset {open_paren}")


def _pop_call_sites() -> list[tuple[str, str, list[str]]]:
    """(relative path, line number as text, arguments) for every pop call."""
    sites: list[tuple[str, str, list[str]]] = []
    for path in sorted(PROFILER_DIR.glob("*")):
        if path.suffix not in {".cc", ".h"}:
            continue
        source = path.read_text(encoding="utf-8")
        for match in _POP_CALL.finditer(source):
            arguments = _call_arguments(source, match.end() - 1)
            line = source.count("\n", 0, match.start()) + 1
            sites.append((str(path.relative_to(REPO_ROOT)), f"line {line}", arguments))
    return sites


def test_external_correlation_pop_passes_a_real_address():
    """No tracer may hand a null out-parameter to the vendor pop."""
    offenders = []
    for path, line, arguments in _pop_call_sites():
        if len(arguments) != 2:
            offenders.append(f"{path}:{line}: expected 2 arguments, got {arguments}")
        elif _NULL_ARGUMENT.match(arguments[1]):
            offenders.append(
                f"{path}:{line}: second argument is {arguments[1]!r}; the vendor "
                "does not unwind the stack for a null out-parameter on at least "
                "CANN, which misattributes every device event (issue #425)"
            )
    assert not offenders, "\n".join(offenders)


def test_the_scan_sees_every_vendor_pop():
    """A check that matches nothing is a check that cannot fail."""
    sites = _pop_call_sites()
    assert len(sites) >= _MIN_CALL_SITES, (
        f"expected at least {_MIN_CALL_SITES} vendor pop call sites in "
        f"{PROFILER_DIR.relative_to(REPO_ROOT)}, found {len(sites)}: {sites}"
    )
    platforms = {path.split("/")[-1].split("_")[0] for path, _, _ in sites}
    assert {"cann", "cupti", "musa"} <= platforms, (
        f"a tracer that delegates its pop to the vendor is no longer scanned: {sorted(platforms)}"
    )


@pytest.mark.parametrize("name", ["cann_device_tracer.cc"])
def test_the_cann_tracer_reads_no_null(name: str):
    """Named explicitly so a rename fails here rather than silently skipping."""
    source = (PROFILER_DIR / name).read_text(encoding="utf-8")
    assert "pop_external(" in source
    assert (
        "pop_external(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, nullptr)" not in source
    )
