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

"""Run the routing calls out of process and report where each one dispatched.

The route a FlagGems operator really takes is a line the C++ layer writes to
stderr (``[flagos dispatch] <op> -> <backend>``), and it has to be read from a
*fresh* interpreter: ``PythonOpCache::GetFunc`` memoizes the resolved FlagGems
callable per qualname and the conf is parsed once per process, so a call made
after the one under test -- or a patch installed after it -- is not a clean
observation. ``probe_routes`` therefore spawns one child with
``FLAGOS_LOG=dispatch,fallback`` in its environment, hands it the whole call
table, and reads the result back as JSON. A conf value of ``none`` leaves the
operator unregistered, so it reaches ``[flagos cpu_fallback] <op>`` instead of
a dispatch line; both outcomes are attributed to the same marker block.

One child runs every call rather than one child per call: importing torch_fl and
torch is the expensive part and it is the same cost for all of them. Each call is
bracketed on stderr by a ``### op <name>`` marker, so a dispatch line is
attributed to the call whose block it falls in rather than by scanning the whole
stream -- which matters because most of these calls emit incidental routes
(``lift_fresh``, ``empty_like``, ``clone``) that have nothing to do with the
operator under test.

Not imported at module scope: torch, torch_fl. The child program is a string and
the only module-level import here is ``flaggems_caps``, which is itself
stdlib-only at import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from flaggems_caps import (
    DEVICE,
    FLAGGEMS_EXECUTION_CALLS,
    FLAGGEMS_ROUTING_CALLS,
    PRELUDE,
)

_MARKER_PREFIX = "### op "
_END_PREFIX = "### end "
_DISPATCH_PREFIX = "[flagos dispatch]"
_FALLBACK_PREFIX = "[flagos cpu_fallback]"

# The child program. Self-contained on purpose: it runs under ``python -c``,
# where nothing from this package is on sys.path. torch_fl is imported *before*
# torch, the order the runtime contract requires -- torch_fl's preload phase has
# to put the vendor runtime assets in place before ``import torch`` caches its
# C++ device hooks, and the other order makes the first device operator fail with
# "Cannot initialize CUDA without ATen_cuda library".
_CHILD_SRC = textwrap.dedent(
    """
    import json
    import sys

    import torch_fl  # noqa: F401  -- claiming the flagos device is the point
    import torch

    DEVICE = %(device)r
    torch.manual_seed(0)
    %(prelude)s

    CALLS = %(calls)s

    for op in CALLS["order"]:
        sys.stderr.write("### op " + op + "\\n")
        sys.stderr.flush()
        entry = {"op": op, "error": None}
        try:
            code = compile(CALLS["calls"][op], "<case>", "exec")
            exec(code, globals())  # noqa: S102 -- the table is this package's own
        except Exception as exc:
            entry["error"] = "%%s: %%s" %% (type(exc).__name__, exc)
        sys.stderr.write("### end " + op + "\\n")
        sys.stderr.flush()
        print(json.dumps(entry), flush=True)
    """
)


def _child_source() -> str:
    """The child program with this run's call table baked in."""
    return _child_source_with_table(_call_table())


def _call_table() -> dict:
    """The call bodies and their stable execution order."""
    calls = {**FLAGGEMS_ROUTING_CALLS, **FLAGGEMS_EXECUTION_CALLS}
    return {
        "calls": calls,
        "order": sorted(calls),
    }


def _child_source_with_table(table: dict) -> str:
    """Render the child program around a call table."""
    return _CHILD_SRC % {
        "device": DEVICE,
        "prelude": PRELUDE,
        "calls": json.dumps(table),
    }


def _generated_table(source: str) -> dict:
    """Read back the table from a rendered child, as its interpreter will."""
    prefix, separator, suffix = source.partition("\nCALLS = ")
    if not separator or '\n\nfor op in CALLS["order"]:' not in suffix:
        raise ValueError("child source has no complete CALLS table and loop")
    assignment, _, _ = suffix.partition('\n\nfor op in CALLS["order"]:')
    namespace = {}
    exec(prefix + "\nCALLS = " + assignment, namespace)
    return namespace["CALLS"]


def _tail(text: str, limit: int = 800) -> str:
    """Last ``limit`` characters of ``text``, for a compact error string."""
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _parse_stdout(stdout: str) -> dict[str, dict]:
    """Index the child's one-JSON-object-per-line report by operator."""
    entries: dict[str, dict] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "op" in entry:
            entries[entry["op"]] = entry
    return entries


def attribute_routes(stderr: str, ops: list[str]) -> dict[str, dict]:
    """How each operator left the PrivateUse1 dispatcher, by marked stderr.

    A dispatch line counts only inside the marker block of the operator it names,
    so an operator that internally calls others is not attributed their routes.
    A fallback line is the explicit ``none`` route. The result for each operator
    has ``dispatched_backend`` and ``cpu_fallback``; both being false means no
    route of its own was observed.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in stderr.splitlines():
        if line.startswith(_MARKER_PREFIX):
            current = line[len(_MARKER_PREFIX) :].strip()
            blocks.setdefault(current, [])
            continue
        if line.startswith(_END_PREFIX):
            current = None
            continue
        if current is not None and line.startswith(
            (_DISPATCH_PREFIX, _FALLBACK_PREFIX)
        ):
            blocks[current].append(line)
    routes: dict[str, dict] = {}
    for op in ops:
        backend: str | None = None
        cpu_fallback = False
        for line in blocks.get(op, ()):
            if line.startswith(_FALLBACK_PREFIX):
                cpu_fallback = line[len(_FALLBACK_PREFIX) :].strip() == op
                continue
            body = line.split("]", 1)[1]
            opname, _, name = body.partition("->")
            if opname.strip() == op:
                backend = name.strip()
        routes[op] = {
            "dispatched_backend": backend,
            "cpu_fallback": cpu_fallback,
        }
    return routes


def probe_routes(
    ops: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, dict]:
    """Run the routing calls on device and report ``op -> outcome``.

    Returns one entry per requested operator:
    ``{"dispatched_backend", "cpu_fallback", "error"}``. The backend is the one
    named by the child's ``[flagos dispatch]`` line; ``cpu_fallback`` says the
    child printed the matching fallback line. A child that exits non-zero yields
    one error entry per operator instead of raising, so a caller can report
    every operator it asked for.
    """
    probe_table = {**FLAGGEMS_ROUTING_CALLS, **FLAGGEMS_EXECUTION_CALLS}
    selected = list(probe_table if ops is None else ops)
    env = os.environ.copy()
    env["FLAGOS_LOG"] = "dispatch,fallback"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", _child_source()],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        message = f"probe subprocess exited {proc.returncode}: {_tail(proc.stderr)}"
        return {
            op: {
                "dispatched_backend": None,
                "cpu_fallback": False,
                "error": message,
            }
            for op in selected
        }
    entries = _parse_stdout(proc.stdout)
    routes = attribute_routes(proc.stderr, selected)
    report: dict[str, dict] = {}
    for op in selected:
        entry = entries.get(op, {"error": "no result emitted"})
        report[op] = {
            "dispatched_backend": routes[op]["dispatched_backend"],
            "cpu_fallback": routes[op]["cpu_fallback"],
            "error": entry.get("error"),
        }
    return report
