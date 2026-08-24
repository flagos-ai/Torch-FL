#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
"""Measure active FlagGems routes through their exact ATen overloads.

This is a manual hardware survey, not a pytest test. CI does not invoke files in
``tests/manual``.

The unit of measurement is an active, unique ``flagos_python`` overload from
``backends_flaggems.conf``. Each overload runs in a fresh child process through
``torch.ops.aten.<name>.<overload>``. Inputs are synthesized from the real ATen
schema and are first validated on CPU; device results are then compared with the
same overload on CPU.

Two support levels are reported:

* basic: at least one CPU-valid case executes correctly on ``flagos``;
* strict: every CPU-valid case in the shape/dtype/layout matrix is correct.

An overload with no CPU-valid synthesized case is UNTESTED, never a failure or a
pass. Raw per-case evidence is retained in JSON for later auditing.

Usage:
  python tests/manual/flaggems_overload_survey.py \
      --conf torch_fl/configs/backends_flaggems.conf \
      --out /tmp/flaggems-overloads.json

Resume after interruption by running the same command again. Use ``--rerun`` to
replace existing results. A single overload can be inspected with ``--ops``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


POISON_RE = re.compile(
    r"illegal memory access|device-side assert|unspecified launch failure|"
    r"misaligned address|vmfault|acceleratorerror",
    re.IGNORECASE,
)

# Diverse enough to expose rank, dtype, and output-layout defects without
# turning this availability survey into an exhaustive conformance suite.
PROFILES = (
    {"name": "2d-f32", "shape": [32, 32], "dtype": "float32", "layout": "contiguous"},
    {
        "name": "4d-f32",
        "shape": [2, 3, 16, 16],
        "dtype": "float32",
        "layout": "contiguous",
    },
    {"name": "1d-f32", "shape": [32], "dtype": "float32", "layout": "contiguous"},
    {"name": "2d-f16", "shape": [32, 32], "dtype": "float16", "layout": "contiguous"},
    {"name": "2d-i64", "shape": [32, 32], "dtype": "int64", "layout": "contiguous"},
    {"name": "2d-bool", "shape": [32, 32], "dtype": "bool", "layout": "contiguous"},
    {
        "name": "2d-f32-strided",
        "shape": [32, 32],
        "dtype": "float32",
        "layout": "strided",
    },
)

HARNESS_VERSION = 4


def active_routes(path: Path) -> list[str]:
    routes = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        op, backend = (part.strip() for part in line.split("=", 1))
        if backend == "flagos_python":
            routes.add(op)
    return sorted(routes)


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


CHILD = r'''
import json
import re
import sys
import traceback

import torch
import torch_fl  # noqa: F401

op_name = sys.argv[1]
profiles = json.loads(sys.argv[2])
DEV = "flagos"
NONDETERMINISTIC_BASES = {
    "bernoulli",
    "bernoulli_",
    "exponential_",
    "multinomial",
    "native_dropout",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "uniform_",
}

# Set per profile by run_profile() below; the synthesis helpers read them as
# module globals.
dtype = torch.float32
shape = (32, 32)
layout = "contiguous"


def emit(profile_name, status, **extra):
    print(json.dumps({"profile": profile_name, "status": status, **extra}), flush=True)


def tensor(*, name="", force_dtype=None, force_shape=None):
    dt = force_dtype or dtype
    sh = tuple(force_shape or shape)
    lname = name.lower()
    if any(x in lname for x in ("index", "indices", "target", "label", "offset", "repeats")):
        dt = torch.int64
        sh = (sh[0],) if sh else (1,)
    elif "mask" in lname:
        dt = torch.bool
    if dt == torch.bool:
        value = torch.ones(sh, dtype=dt)
    elif dt.is_floating_point or dt.is_complex:
        value = torch.randn(sh, dtype=dt)
    else:
        value = torch.randint(0, 2, sh, dtype=dt)
    if layout == "strided" and len(sh) >= 2 and sh[-1] == sh[-2]:
        value = value.t()
    return value


def normalize(type_str):
    """Reduce a schema type to (base, optional).

    ``str(arg.type)`` prints ``Optional[List[int]]`` while the schema text uses
    ``int[1]?``; both spellings appear depending on the accessor, so collapse
    them to one form. Getting this wrong reports a synthesis gap as INVALID_CASE
    for every profile, which silently removes the overload from the denominator.
    """
    t = type_str.strip()
    optional = False
    while True:
        if t.endswith("?"):
            t, optional = t[:-1].strip(), True
            continue
        if t.startswith("Optional[") and t.endswith("]"):
            t, optional = t[len("Optional["):-1].strip(), True
            continue
        break
    if t.startswith("List[") and t.endswith("]"):
        t = f"{t[len('List['):-1].strip()}[]"
    t = re.sub(r"\[\d+\]", "[]", t)
    if t in ("SymInt", "int64_t"):
        t = "int"
    if t == "number":
        t = "Scalar"
    if t == "SymInt[]":
        t = "int[]"
    if t in ("TensorList",):
        t = "Tensor[]"
    if t in ("string", "string_view", "c10::string_view"):
        t = "str"
    return t, optional


def default_for(arg):
    base, optional = normalize(str(arg.type))
    name = arg.name.lower()
    # Some PyTorch schema accessors expose ScalarType? as Optional[int]. Avoid
    # passing enum value 1 (Char) to factory ops when dtype should be omitted.
    if optional and name == "dtype":
        return None
    if base == "Device":
        return torch.device("cpu")
    if optional and name not in ("dim", "dtype"):
        return None
    if base == "Tensor":
        return tensor(name=name)
    if base == "Tensor[]":
        return [tensor(name=name), tensor(name=name)]
    if base == "int":
        if name in ("dim", "dim0", "start", "reduction", "axis"):
            return 0
        if name == "dim1":
            return 1 if len(shape) > 1 else 0
        if name in ("end", "size", "length"):
            return shape[0] if shape else 1
        if name == "output_size":
            return max(1, shape[-1] // 2) if shape else 1
        if "group" in name:
            return 1
        return 1
    if base in ("int[]", "SymInt[]"):
        if name in ("dim", "dims", "axis", "axes"):
            return [0]
        if name == "output_size":
            rank = 2 if len(shape) >= 3 else 1
            return [max(1, size // 2) for size in shape[-rank:]]
        if any(x in name for x in ("kernel", "stride", "padding", "dilation")):
            return [1, 1] if len(shape) >= 3 else [1]
        if name in ("normalized_shape",):
            return [shape[-1]]
        if name in ("size", "shape", "self_sizes", "input_size", "input_sizes"):
            return list(shape)
        return list(shape)
    if base in ("float", "Scalar"):
        if name == "start":
            return 0
        if name == "end":
            return max(2, shape[0] if shape else 2)
        if name == "step":
            return 1
        if name in ("alpha", "beta", "value"):
            return 1.0
        if name == "p":
            return 0.5
        if name == "eps":
            return 1e-5
        return 0.5
    if base == "bool":
        return False
    if base == "ScalarType":
        return None if optional else dtype
    if base in ("Layout", "MemoryFormat", "Generator", "str"):
        return None if optional else ({"str": "none"}.get(base))
    if base == "bool[]":
        return [True] * max(1, len(shape))
    raise TypeError(f"unhandled schema type {arg.type} for {arg.name}")


def build_case(schema):
    """Build one CPU argument list; the device run reuses these exact values.

    Synthesizing separately per device would compare two different random
    inputs, which reports every elementwise kernel as numerically wrong.
    """
    positional = []
    kwargs = {}
    tensors = []
    out_slots = []
    for arg in schema.arguments:
        try:
            value = default_for(arg)
        except Exception:
            if arg.has_default_value():
                value = arg.default_value
            else:
                raise
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        if arg.is_out:
            proto = tensors[0] if tensors else tensor(name=arg.name)
            value = torch.empty_like(proto)
            if layout == "strided" and value.ndim >= 2 and value.shape[-1] == value.shape[-2]:
                value = torch.empty_like(value).t()
            out_slots.append(arg.name if arg.kwarg_only else len(positional))
        if arg.kwarg_only:
            kwargs[arg.name] = value
        else:
            positional.append(value)
    return positional, kwargs, out_slots


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device)
    if isinstance(value, torch.device):
        return torch.device(device)
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def flatten(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for item in x for t in flatten(item)]
    return []


def tensor_outputs(value):
    return flatten(value)


def reset_seed():
    torch.manual_seed(1729)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1729)


def compare_random(actual):
    tensors = tensor_outputs(actual)
    if not tensors:
        return False, "operator returned no tensors"
    for got in tensors:
        if got.device.type == "cpu":
            return False, f"result stayed on CPU instead of {DEV}"
        host = got.detach().cpu()
        if host.dtype.is_floating_point or host.dtype.is_complex:
            if not torch.isfinite(host).all():
                return False, "non-finite random output"
    return True, "execution/device check only; values are nondeterministic"


def compare(expected, actual):
    e, a = flatten(expected), flatten(actual)
    if len(e) != len(a):
        return False, f"return arity {len(e)} != {len(a)}"
    for want, got in zip(e, a):
        if got.device.type != DEV:
            return False, f"result device {got.device} != {DEV}"
        got = got.detach().cpu()
        want = want.detach().cpu()
        if got.shape != want.shape:
            return False, f"shape {tuple(got.shape)} != {tuple(want.shape)}"
        if got.dtype != want.dtype:
            return False, f"dtype {got.dtype} != {want.dtype}"
        if got.dtype.is_floating_point or got.dtype.is_complex:
            if not torch.allclose(got, want, atol=2e-2, rtol=2e-2, equal_nan=True):
                diff = (got.float() - want.float()).abs().max().item()
                return False, f"max_diff={diff}"
        elif not torch.equal(got, want):
            return False, "unequal"
    return True, ""


def run_profile(packet, schema, profile):
    global dtype, shape, layout
    dtype = getattr(torch, profile["dtype"])
    shape = tuple(profile["shape"])
    layout = profile["layout"]

    # Validate on CPU first: input the reference itself rejects says nothing
    # about the accelerator, so it is recorded separately from a runtime failure.
    try:
        cpu_args, cpu_kwargs, out_slots = build_case(schema)
        dev_args = [to_device(v, DEV) for v in cpu_args]
        dev_kwargs = {k: to_device(v, DEV) for k, v in cpu_kwargs.items()}
        reset_seed()
        expected = packet(*cpu_args, **cpu_kwargs)
    except Exception as exc:
        emit(profile["name"], "INVALID_CASE", error=f"{type(exc).__name__}: {exc}"[:300])
        return

    try:
        reset_seed()
        actual = packet(*dev_args, **dev_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        actual_tensors = tensor_outputs(actual)
        if not actual_tensors:
            emit(profile["name"], "UNVERIFIABLE", detail="operator returned no tensors")
            return
        random_op = op_name.split(".", 1)[0] in NONDETERMINISTIC_BASES
        if random_op:
            ok, detail = compare_random(actual)
        else:
            ok, detail = compare(expected, actual)
        if ok and out_slots and not random_op:
            written_cpu = [
                cpu_kwargs[s] if isinstance(s, str) else cpu_args[s] for s in out_slots
            ]
            written_dev = [
                dev_kwargs[s] if isinstance(s, str) else dev_args[s] for s in out_slots
            ]
            ok, detail = compare(written_cpu, written_dev)
            if not ok:
                detail = f"out buffer: {detail}"
        emit(profile["name"], "PASS" if ok else "WRONG", detail=detail)
    except Exception as exc:
        emit(
            profile["name"],
            "ERROR",
            error=f"{type(exc).__name__}: {exc}"[:300],
            traceback=traceback.format_exc()[-600:],
        )


base, dot, overload = op_name.partition(".")
overload = overload if dot else "default"
try:
    packet = getattr(getattr(torch.ops.aten, base), overload)
    schema = packet._schema
except Exception as exc:
    for p in profiles:
        emit(p["name"], "NO_SCHEMA", error=f"{type(exc).__name__}: {exc}"[:300])
    sys.exit(0)

for p in profiles:
    run_profile(packet, schema, p)
'''


def run_overload(child: Path, op: str, env: dict, timeout: int) -> list[dict]:
    """Run every profile for one overload in a single child process.

    One process per overload rather than per case: importing torch dominates
    runtime otherwise. Each profile still reports its own line, so a process that
    dies mid-way is attributed to the exact profile that killed it instead of
    discarding the overload.
    """
    pending = [p["name"] for p in PROFILES]
    try:
        proc = subprocess.run(
            [sys.executable, str(child), op, json.dumps(list(PROFILES))],
            capture_output=True,
            text=True,
            env=env,
            cwd="/tmp",
            timeout=timeout,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        rc, timed_out = None, True

    cases = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = " ".join(str(rec.get(k, "")) for k in ("error", "detail", "traceback"))
        if POISON_RE.search(text):
            rec["context_poison"] = True
        cases.append(rec)
        if rec["profile"] in pending:
            pending.remove(rec["profile"])

    # Profiles with no line never reported: the process aborted (a faulting
    # kernel can kill it instead of raising) or ran out of time.
    for name in pending:
        cases.append(
            {
                "profile": name,
                "status": "TIMEOUT" if timed_out else "CRASH",
                "error": (stderr or stdout)[-300:],
                "returncode": rc,
            }
        )
    return cases


def summarize(routes: list[str], results: dict[str, dict]) -> dict:
    counts: dict[str, int] = {}
    basic = strict = tested = 0
    for op in routes:
        rec = results.get(op)
        if not rec:
            verdict = "PENDING"
        else:
            valid = [
                case
                for case in rec["cases"]
                if case["status"] not in ("INVALID_CASE", "UNVERIFIABLE")
            ]
            passed = [case for case in valid if case["status"] == "PASS"]
            tested += bool(valid)
            basic += bool(passed)
            strict += bool(valid) and len(passed) == len(valid)
            if not valid:
                verdict = "UNTESTED"
            elif len(passed) == len(valid):
                verdict = "STRICT"
            elif passed:
                verdict = "BASIC_ONLY"
            else:
                verdict = "FAILED"
        counts[verdict] = counts.get(verdict, 0) + 1
    strided_fail = sorted(
        op
        for op in routes
        if any(
            c.get("profile") == "2d-f32-strided"
            and c["status"] in ("WRONG", "ERROR", "CRASH")
            for c in results.get(op, {}).get("cases", [])
        )
    )
    return {
        "registered": len(routes),
        "tested": tested,
        "basic_executable": basic,
        "strict_support": strict,
        "verdicts": counts,
        "strided_failures": strided_fail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ops", help="comma-separated exact overload names")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    conf_bytes = args.conf.read_bytes()
    conf_sha256 = hashlib.sha256(conf_bytes).hexdigest()
    routes = active_routes(args.conf)
    full_routes = routes
    if args.ops:
        selected = set(args.ops.split(","))
        unknown = selected.difference(full_routes)
        if unknown:
            parser.error(
                f"routes not active in {args.conf}: {', '.join(sorted(unknown))}"
            )
        routes = [op for op in routes if op in selected]
    state = {"meta": {}, "results": {}}
    if args.out.exists() and not args.rerun:
        state = json.loads(args.out.read_text())
        old_version = state.get("meta", {}).get("harness_version")
        if old_version != HARNESS_VERSION:
            parser.error(
                f"{args.out} was written by harness version {old_version!r}; "
                "use --rerun to replace it"
            )
        old_hash = state.get("meta", {}).get("conf_sha256")
        if old_hash != conf_sha256:
            parser.error(
                f"{args.out} used config SHA256 {old_hash!r}, not {conf_sha256}; "
                "use --rerun to replace it"
            )
    state["meta"].update(
        {
            "conf": str(args.conf.resolve()),
            "conf_sha256": conf_sha256,
            "harness_version": HARNESS_VERSION,
            "registered": len(routes),
            "routes": routes,
            "profiles": list(PROFILES),
            "unit": "active unique ATen overload",
        }
    )

    tag = os.getpid()
    child = Path(f"/tmp/flaggems_overload_child_{tag}.py")
    child.write_text(CHILD)
    env = dict(os.environ)
    env["FLAGOS_BACKEND_CONFIG"] = str(args.conf.resolve())
    env["TRITON_CACHE_DIR"] = f"/tmp/triton_flaggems_survey_{tag}"
    try:
        pending = [op for op in routes if args.rerun or op not in state["results"]]
        for index, op in enumerate(pending, 1):
            cases = run_overload(child, op, env, args.timeout)
            state["results"][op] = {"cases": cases}
            state["summary"] = summarize(routes, state["results"])
            atomic_write(args.out, state)
            verdicts = state["summary"]["verdicts"]
            print(
                f"[{index}/{len(pending)}] {op:44s} "
                f"strict={verdicts.get('STRICT', 0)} "
                f"basic_only={verdicts.get('BASIC_ONLY', 0)} "
                f"failed={verdicts.get('FAILED', 0)} "
                f"untested={verdicts.get('UNTESTED', 0)}",
                flush=True,
            )
    finally:
        child.unlink(missing_ok=True)

    state["summary"] = summarize(routes, state["results"])
    state["meta"]["completed_at"] = time.time()
    atomic_write(args.out, state)
    print(json.dumps(state["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
