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

"""Run the flagos device out of process and report what it measured.

The dispatch contract has to observe a line the C++ layer writes to stderr
(``[flagos dispatch] <op> -> <backend>``) and it has to do so for a *fresh*
interpreter: torch_fl claims the device at import time, so a run that already
built tensors is not a clean probe. ``probe_ops`` therefore spawns one child
with ``FLAGOS_LOG=dispatch`` in its environment, has it run each requested op on
``flagos:0`` next to a CPU reference, and reads the result back as JSON.

Each op is bracketed on stderr by a ``### op <name>`` marker, so a dispatch line
is attributed to the op whose block it falls in rather than by scanning the whole
stream. The child compares the device result to a CPU reference with
``torch.allclose`` and reports ``cpu_reference_ok``; the parent fills in the
``dispatched_backend`` it read from stderr. ``run_probe`` and the
device/stream/allocator/transfer helpers expose the same subprocess view to any
module that wants it.

Not imported at module scope: torch, torch_fl. The snippets are strings, and the
only import here is ``runtime_caps`` -- itself stdlib-only at import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from runtime_caps import RUNTIME_OPS

_MARKER_PREFIX = "### op "
_DISPATCH_PREFIX = "[flagos dispatch]"

# The child program. Self-contained on purpose: it runs under `python -c`, where
# nothing from this package is on sys.path. torch_fl is imported *before* torch,
# which is the order the runtime contract requires: torch_fl's preload phase has
# to put the vendor runtime assets in place before ``import torch`` caches its
# C++ device hooks. The other order makes the first device op that reaches those
# hooks fail with "Cannot initialize CUDA without ATen_cuda library" -- the same
# rule the fresh-process snippets under tests/integration/ops/ follow.
_CHILD_SRC = textwrap.dedent(
    """
    import json
    import sys

    import torch_fl  # noqa: F401  -- claiming the flagos device is the point
    import torch

    _DEVICE = "flagos:0"
    _RTOL = 1e-3
    _ATOL = 1e-3


    def _randn(seed, shape):
        torch.manual_seed(seed)
        return torch.randn(*shape)


    def _run_one(op):
        aten = torch.ops.aten
        if op == "add.Tensor":
            a = _randn(0, (4, 4))
            b = _randn(1, (4, 4))
            return (aten.add.Tensor(a.to(_DEVICE), b.to(_DEVICE)), aten.add.Tensor(a, b))
        if op == "mul.Tensor":
            a = _randn(2, (4, 4))
            b = _randn(3, (4, 4))
            return (aten.mul.Tensor(a.to(_DEVICE), b.to(_DEVICE)), aten.mul.Tensor(a, b))
        if op == "mm":
            a = _randn(4, (8, 8))
            b = _randn(5, (8, 8))
            return (aten.mm(a.to(_DEVICE), b.to(_DEVICE)), aten.mm(a, b))
        if op == "bmm":
            a = _randn(6, (4, 8, 8))
            b = _randn(7, (4, 8, 8))
            return (aten.bmm(a.to(_DEVICE), b.to(_DEVICE)), aten.bmm(a, b))
        if op == "cat":
            a = _randn(8, (4, 4))
            b = _randn(9, (4, 4))
            return (
                aten.cat([a.to(_DEVICE), b.to(_DEVICE)], 0),
                aten.cat([a, b], 0),
            )
        if op == "neg":
            a = _randn(10, (4, 4))
            return (aten.neg(a.to(_DEVICE)), aten.neg(a))
        if op == "sum.dim_IntList":
            a = _randn(11, (4, 4))
            return (
                aten.sum.dim_IntList(a.to(_DEVICE), [1], False),
                aten.sum.dim_IntList(a, [1], False),
            )
        if op == "_softmax":
            a = _randn(12, (4, 4))
            return (
                aten._softmax(a.to(_DEVICE), -1, False),
                aten._softmax(a, -1, False),
            )
        if op == "embedding":
            weight = _randn(13, (10, 5))
            index = torch.tensor([1, 3, 4], dtype=torch.long)
            return (
                aten.embedding(weight.to(_DEVICE), index.to(_DEVICE)),
                aten.embedding(weight, index),
            )
        if op == "where.self":
            cond = _randn(14, (4, 4)) > 0.0
            a = _randn(15, (4, 4))
            b = _randn(16, (4, 4))
            return (
                aten.where.self(
                    cond.to(_DEVICE), a.to(_DEVICE), b.to(_DEVICE)
                ),
                aten.where.self(cond, a, b),
            )
        raise ValueError("unknown op: %s" % op)


    def _main():
        report = []
        for op in sys.argv[1:]:
            sys.stderr.write("### op " + op + "\\n")
            sys.stderr.flush()
            entry = {"op": op, "cpu_reference_ok": False, "error": None}
            try:
                out, ref = _run_one(op)
                ok = bool(torch.allclose(out.cpu(), ref, rtol=_RTOL, atol=_ATOL))
                entry["cpu_reference_ok"] = ok
                if not ok:
                    entry["error"] = "device result does not match CPU reference"
            except Exception as exc:
                entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            report.append(entry)
        json.dump(report, sys.stdout)
        sys.stdout.write("\\n")
        sys.stdout.flush()


    _main()
    """
).strip()


def _tail(text: str, limit: int = 800) -> str:
    """Last ``limit`` characters of ``text``, for a compact error string."""
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _error_entry(op: str, message: str) -> dict:
    """A probe entry that records a failure before any numpy work happened."""
    return {
        "op": op,
        "dispatched_backend": None,
        "cpu_reference_ok": False,
        "error": message,
    }


def _parse_stdout(stdout: str) -> dict[str, dict]:
    """Index the child's JSON result list by op, tolerating a trailing newline."""
    text = stdout.strip()
    if not text:
        return {}
    payload = json.loads(text.splitlines()[-1])
    return {entry["op"]: entry for entry in payload}


def _attribute_dispatch(stderr: str, ops: list[str]) -> dict[str, str | None]:
    """Backend each op dispatched to, read from the marker-bracketed stderr.

    A dispatch line is counted only inside the marker block of the op it names,
    which keeps an op that internally calls others from being attributed their
    routes. An op whose block holds no matching line maps to ``None`` -- it did
    not reach a backend (cpu_fallback, or an uncompiled kernel).
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in stderr.splitlines():
        if line.startswith(_MARKER_PREFIX):
            current = line[len(_MARKER_PREFIX) :].strip()
            blocks.setdefault(current, [])
            continue
        if current is not None and line.startswith(_DISPATCH_PREFIX):
            blocks[current].append(line)
    routed: dict[str, str | None] = {}
    for op in ops:
        backend: str | None = None
        for line in blocks.get(op, ()):
            body = line.split("]", 1)[1]
            opname, _, name = body.partition("->")
            if opname.strip() == op:
                backend = name.strip()
        routed[op] = backend
    return routed


def probe_ops(
    ops: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: int = 600,
) -> list[dict]:
    """Run ``ops`` on flagos:0 in a child and report the dispatch outcome.

    Returns one entry per requested op: ``{"op", "dispatched_backend",
    "cpu_reference_ok", "error"}``. ``dispatched_backend`` is the backend named
    by the child's ``[flagos dispatch]`` line, or ``None`` when the op emitted
    none. A child that exits non-zero yields one error entry per op instead of
    raising, so a caller can report every op it asked for.
    """
    selected = list(RUNTIME_OPS if ops is None else ops)
    env = os.environ.copy()
    env["FLAGOS_LOG"] = "dispatch"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SRC, *selected],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        message = f"probe subprocess exited {proc.returncode}: {_tail(proc.stderr)}"
        return [_error_entry(op, message) for op in selected]
    entries = _parse_stdout(proc.stdout)
    dispatched = _attribute_dispatch(proc.stderr, selected)
    report = []
    for op in selected:
        entry = entries.get(
            op, {"op": op, "cpu_reference_ok": False, "error": "no result emitted"}
        )
        report.append(
            {
                "op": op,
                "dispatched_backend": dispatched.get(op),
                "cpu_reference_ok": bool(entry.get("cpu_reference_ok", False)),
                "error": entry.get("error"),
            }
        )
    return report


def run_probe(
    snippet: str, extra_env: dict[str, str] | None = None, timeout: int = 600
) -> dict:
    """Run a self-contained child snippet and return the JSON object it prints.

    The snippet must ``json.dump`` one object to stdout; it runs with
    ``FLAGOS_LOG=dispatch`` set so a caller may also inspect the child's stderr.
    """
    env = os.environ.copy()
    env["FLAGOS_LOG"] = "dispatch"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet).strip()],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"probe snippet exited {proc.returncode}: {_tail(proc.stderr)}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


_DEVICE_SRC = """
import json
import sys

import torch_fl

props = torch_fl.flagos.get_device_properties(0)
json.dump(
    {
        "available": bool(torch_fl.flagos.is_available()),
        "device_count": int(torch_fl.flagos.device_count()),
        "current_device": int(torch_fl.flagos.current_device()),
        "name": str(props.name),
        "total_memory": int(props.total_memory),
    },
    sys.stdout,
)
"""

_STREAM_SRC = """
import json
import sys

import torch_fl  # noqa: F401  -- preloads the vendor runtime before torch is imported
import torch

torch.manual_seed(0)
a = torch.randn(64, 64)
b = torch.randn(64, 64)
ref = torch.mm(a, b)
da, db = a.to("flagos:0"), b.to("flagos:0")
stream = torch_fl.flagos.Stream()
with torch_fl.flagos.stream(stream):
    out = torch.mm(da, db)
torch_fl.flagos.synchronize()
start = torch_fl.flagos.Event(enable_timing=True)
end = torch_fl.flagos.Event(enable_timing=True)
start.record()
torch.mm(da, db)
end.record()
end.synchronize()
json.dump(
    {
        "matches_default": bool(
            torch.allclose(out.cpu(), ref, rtol=1e-3, atol=1e-3)
        ),
        "elapsed_time": float(start.elapsed_time(end)),
    },
    sys.stdout,
)
"""

_ALLOCATOR_SRC = """
import json
import sys

import torch_fl

torch_fl.flagos.empty_cache()
stats = torch_fl.flagos.memory_stats()
json.dump(
    {
        "keys": sorted(stats.keys()),
        "allocated_bytes": int(stats["allocated_bytes"]),
        "reserved_bytes": int(stats["reserved_bytes"]),
        "peak_allocated_bytes": int(stats["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(stats["peak_reserved_bytes"]),
    },
    sys.stdout,
)
"""

_TRANSFER_SRC = """
import json
import sys

import torch_fl  # noqa: F401  -- preloads the vendor runtime before torch is imported
import torch

torch.manual_seed(0)
cpu = torch.randn(8, 8)
dev = cpu.to("flagos:0")
json.dump(
    {
        "device_type": dev.device.type,
        "h2d_ok": bool(torch.allclose(dev.cpu(), cpu)),
        "d2d_ok": bool(torch.allclose(dev.clone().cpu(), cpu)),
    },
    sys.stdout,
)
"""


def probe_device() -> dict:
    """Device registration and properties, read from a fresh child."""
    return run_probe(_DEVICE_SRC)


def probe_stream() -> dict:
    """Non-default-stream matmul vs the CPU reference and event timing."""
    return run_probe(_STREAM_SRC)


def probe_allocator() -> dict:
    """Allocator statistics key set and bytes, read from a fresh child."""
    return run_probe(_ALLOCATOR_SRC)


def probe_transfer() -> dict:
    """Host->device and device->device round-trips, read from a fresh child."""
    return run_probe(_TRANSFER_SRC)
