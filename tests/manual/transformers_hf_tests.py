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
"""Run HuggingFace's own tests for one model architecture on an accelerator.

This is a manual hardware measurement, not a pytest test: CI does not invoke
files in ``tests/manual``. Where ``transformers_model_probe.py`` builds a tiny
model and probes four layers of its own design, this runner executes the
assertions HuggingFace maintains for the architecture --- everything under
``tests/models/<module>/`` in the transformers source tree that matches the
installed wheel.

The unit of work is one architecture per invocation, in its own subprocess. A
sweep is an external loop. Accelerator faults are not contained: one illegal
memory access poisons the device context, after which every later test in the
same process fails with the same symptom, so a poisoned run is reported as one
finding for the model rather than as a list of independent failures.

Usage:
  python tests/manual/transformers_hf_tests.py --model qwen3
  python tests/manual/transformers_hf_tests.py --model qwen3 --out /tmp/qwen3.json
  python tests/manual/transformers_hf_tests.py --model blip-2 --collect-only
  python tests/manual/transformers_hf_tests.py --all --out /tmp/transformers-all.json
  python tests/manual/transformers_hf_tests.py --list-models

``--model`` runs every official test under one architecture directory. ``--all``
iterates all architecture keys exposed by the installed Transformers version,
with one isolated subprocess per architecture; it does not enumerate Hub
checkpoints or download model weights.

Issue filing is deliberately not part of this runner. It writes an auditable
JSON result that a later reporter consumes, so that executing tests and writing
to a shared tracker stay separable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers_hf_source import (  # noqa: E402 - local manual-test helper
    SourceError,
    atomic_write,
    resolve_version,
    use_source,
)

HARNESS_VERSION = 1
SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_SPEC = Path(__file__).resolve().parent / "hf_device_spec.py"
# Filename the spec is staged under inside the pinned Transformers source tree.
# Prefixed so it cannot collide with anything HF ships or later adds.
STAGED_SPEC_NAME = "torch_fl_hf_device_spec.py"

# One illegal access poisons the device context for the rest of the process.
# Same pattern set as the model probe, so both harnesses call a fault a fault.
POISON_RE = re.compile(
    r"illegal memory access|device-side assert|unspecified launch failure|"
    r"misaligned address|vmfault|acceleratorerror",
    re.IGNORECASE,
)

# ``require_torch_gpu`` and its relatives compare against ``cuda`` literally, so
# they skip on any other accelerator. Those cases are CUDA-specific and are not
# coverage gaps for this platform; keeping them separate from ordinary skips is
# what stops a report from inflating its own denominator.
CUDA_ONLY_RE = re.compile(
    r"requires? (a )?(cuda|gpu|rocm)|cuda gpu|rocm|nvidia|"
    r"test requires multiple cuda|bitsandbytes|flash.?attn",
    re.IGNORECASE,
)

# A missing optional dependency is an environment gap, not a platform defect:
# the official suite pulls accelerate, datasets, librosa, peft and more, and a
# box without them never ran the assertion at all.
ENVIRONMENT_RE = re.compile(
    r"No module named|ModuleNotFoundError|ImportError|"
    r"is not installed|requires the .{0,40} library|cannot import name",
    re.IGNORECASE,
)

STATUSES = (
    "PASS",
    "FAIL",
    "ERROR",
    "XFAIL",
    "XPASS",
    "SKIP_CUDA_ONLY",
    "SKIP_OTHER",
    "ENVIRONMENT_ERROR",
    "COLLECT_ERROR",
)

MARK = "@@HFTEST@@"


# The plugin runs inside the pytest subprocess. It is written to a temporary
# file and loaded with ``-p`` rather than added as a dependency: a new runtime
# dependency for a manual harness would have to be installed on every vendor box
# before that box could be measured at all.
PLUGIN = r"""
import json
import os


def _text(value):
    if value is None:
        return None
    return str(value)[:8000]


class Recorder:
    def __init__(self, path):
        self.path = path
        self.records = []

    def add(self, record):
        self.records.append(record)
        # Append as we go: a fault that kills the interpreter still leaves the
        # tests that already finished on disk.
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome == "failed" or report.skipped:
            self.add(
                {
                    "kind": "test",
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "duration": getattr(report, "duration", None),
                    "wasxfail": getattr(report, "wasxfail", None) is not None,
                    "longrepr": _text(report.longrepr),
                    "sections": [
                        [name, _text(content)] for name, content in report.sections[:6]
                    ],
                }
            )

    def pytest_collectreport(self, report):
        if report.failed:
            self.add(
                {
                    "kind": "collect",
                    "nodeid": report.nodeid,
                    "outcome": "failed",
                    "longrepr": _text(report.longrepr),
                }
            )

    def pytest_collection_finish(self, session):
        self.add({"kind": "collected", "count": len(session.items)})


def pytest_configure(config):
    path = os.environ["HF_TEST_REPORT"]
    config.pluginmanager.register(Recorder(path), "torch_fl_hf_recorder")
"""


def classify_skip(reason: str | None) -> str:
    """Split a skip into CUDA-only and everything else."""
    if reason and CUDA_ONLY_RE.search(reason):
        return "SKIP_CUDA_ONLY"
    return "SKIP_OTHER"


def classify_failure(detail: str | None) -> str:
    """Decide whether a failure is a platform result or environment noise."""
    if detail and ENVIRONMENT_RE.search(detail):
        return "ENVIRONMENT_ERROR"
    return "FAIL"


def _is_xfail(record: dict) -> bool:
    """Interpret the plugin's boolean xfail field defensively."""
    return bool(record.get("wasxfail"))


def reduce_records(records: list[dict]) -> dict:
    """Fold raw pytest reports into one status per test node.

    A node reports up to three times (setup, call, teardown). The worst outcome
    decides, and a setup or teardown failure is an error rather than a failed
    assertion: the test body never ran, so calling it a failure would blame the
    model's behaviour for a harness problem.
    """
    # Keep this explicit rather than relying on the order of ``STATUSES``:
    # SKIP_CUDA_ONLY and XFAIL are informational and must never hide a FAIL.
    severity = {
        "PASS": 0,
        "SKIP_CUDA_ONLY": 0,
        "SKIP_OTHER": 0,
        "XFAIL": 0,
        "XPASS": 1,
        "FAIL": 3,
        "ERROR": 4,
        "ENVIRONMENT_ERROR": 5,
        "COLLECT_ERROR": 6,
    }

    tests: dict[str, dict] = {}
    collect_errors: list[dict] = []
    collected = None

    for record in records:
        kind = record.get("kind")
        if kind == "collected":
            collected = record.get("count")
            continue
        if kind == "collect":
            detail = record.get("longrepr")
            collect_errors.append(
                {
                    "nodeid": record.get("nodeid"),
                    "status": "ENVIRONMENT_ERROR"
                    if detail and ENVIRONMENT_RE.search(detail)
                    else "COLLECT_ERROR",
                    "detail": detail,
                }
            )
            continue
        if kind != "test":
            continue

        nodeid = record.get("nodeid")
        outcome = record.get("outcome")
        when = record.get("when")
        detail = record.get("longrepr")
        entry = tests.setdefault(
            nodeid,
            {
                "nodeid": nodeid,
                "status": "PASS",
                "duration_s": 0.0,
                "detail": None,
                "output": None,
            },
        )
        duration = record.get("duration")
        if isinstance(duration, (int, float)):
            entry["duration_s"] = round(entry["duration_s"] + duration, 3)

        if outcome == "skipped":
            status = "XFAIL" if _is_xfail(record) else classify_skip(detail)
        elif outcome == "failed":
            if _is_xfail(record):
                status = "XFAIL"
            elif when == "call":
                status = classify_failure(detail)
            else:
                status = (
                    "ENVIRONMENT_ERROR"
                    if (detail and ENVIRONMENT_RE.search(detail))
                    else "ERROR"
                )
        elif outcome == "passed" and _is_xfail(record):
            status = "XPASS"
        else:
            status = "PASS"

        # Worst status wins, so a passing call cannot hide a teardown error.
        old_status = entry["status"]
        if severity.get(status, 0) > severity.get(old_status, 0) or (
            old_status == "PASS" and status != "PASS"
        ):
            entry["status"] = status
            entry["detail"] = detail
            sections = record.get("sections") or []
            entry["output"] = (
                "\n".join(
                    f"--- {name}\n{content}" for name, content in sections if content
                )
                or None
            )

    return {
        "tests": sorted(tests.values(), key=lambda t: t["nodeid"]),
        "collect_errors": collect_errors,
        "collected": collected,
    }


def summarize_statuses(tests: list[dict], collect_errors: list[dict]) -> dict:
    counts = {name: 0 for name in STATUSES}
    for test in tests:
        counts[test["status"]] += 1
    for error in collect_errors:
        counts[error["status"]] += 1
    return {name: value for name, value in counts.items() if value}


def verdict(result: dict) -> str:
    """Reduce one model's run to a single verdict.

    Ordered by how much each outcome invalidates the measurement. A run that
    never executed the tests is not a coverage result and must not be counted as
    support or as a defect.
    """
    run = result["run"]
    # An architecture the pinned version does not ship is neither support nor a
    # defect, and a sweep must be able to tell it apart from a run that started
    # and executed nothing.
    if run.get("status") == "NOT_IN_VERSION":
        return "NOT_IN_VERSION"
    if run.get("timed_out"):
        return "TIMEOUT"
    if run.get("context_poison"):
        return "CRASH"
    counts = result["summary"]
    if run.get("crashed"):
        return "CRASH"
    if counts.get("COLLECT_ERROR"):
        return "COLLECT_ERROR"
    if counts.get("FAIL") or counts.get("ERROR"):
        return "FAIL"
    executed = counts.get("PASS", 0) + counts.get("XFAIL", 0) + counts.get("XPASS", 0)
    if not executed:
        if run.get("collect_only") and run.get("collected"):
            return "PASS"
        if counts.get("ENVIRONMENT_ERROR"):
            return "ENVIRONMENT_ERROR"
        return "NO_TESTS_RUN"
    return "PASS"


def pytest_process_crashed(returncode: int | None) -> bool:
    """Recognize interpreter crashes without misclassifying pytest outcomes."""
    return returncode is not None and returncode not in (0, 1, 2, 3, 4, 5)


def read_report(path: Path) -> list[dict]:
    """Parse the plugin's JSON-lines report, ignoring a truncated last line.

    A crash can cut the file mid-write. Dropping only the damaged line keeps the
    tests that did finish, which is exactly the evidence a crash needs.
    """
    records = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def environment(device: str) -> dict:
    """Collect the versions a result cannot be interpreted without."""
    import torch
    import transformers

    commit = None
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=REPO_ROOT,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_fl_commit": commit,
        "device": device,
        "harness_version": HARNESS_VERSION,
    }


def module_name(model: str) -> str:
    """Map an architecture key to its test directory name.

    Keys and directories disagree for a substantial minority of architectures
    (``blip-2`` -> ``blip_2``), so the mapping has to come from transformers
    itself rather than from a string transform of the key.
    """
    from transformers.models.auto.configuration_auto import model_type_to_module_name

    return model_type_to_module_name(model)


def known_models() -> list[str]:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return sorted(CONFIG_MAPPING_NAMES)


def sweep_models() -> list[str]:
    """Pick one architecture key per test directory for a full sweep.

    Several registry keys share a directory: transformers 5.16.1 exposes 709
    keys over 492 test directories, because component configs such as
    ``blip_text_model`` and ``blip_vision_model`` map to ``tests/models/blip``
    alongside ``blip`` itself. Sweeping the raw key list would run those
    directories two or three times and count one failure as several.
    """
    seen: dict[str, str] = {}
    for model in known_models():
        seen.setdefault(module_name(model), model)
    return [seen[module] for module in sorted(seen)]


def test_dir(source: Path, model: str) -> Path:
    return source / "tests" / "models" / module_name(model)


def child_env(
    source: Path, device: str, report: Path, offline: bool, workdir: Path
) -> dict:
    """Build the environment HuggingFace's device injection contract needs."""
    env = dict(os.environ)
    # Custom PrivateUse1 names are registered by the spec. Transformers validates
    # TRANSFORMERS_TEST_DEVICE before importing that spec, so setting it to
    # ``flagos`` would fail at torch.device() validation.
    env.pop("TRANSFORMERS_TEST_DEVICE", None)
    # Transformers 5.16.1 rejects this value unless it names an existing file
    # relative to the child's cwd, then strips ``.py`` and imports the remainder.
    # A repository-relative path satisfies neither step, so pass the staged name.
    env["TRANSFORMERS_TEST_DEVICE_SPEC"] = STAGED_SPEC_NAME
    env["HF_TEST_REPORT"] = str(report)
    # This repository also has a top-level ``tests`` package. If it stays
    # importable, HF's ``tests.models...`` imports resolve into the wrong tree
    # and every model test errors on import.
    path_entries = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != REPO_ROOT
    ]
    # HF's ``tests`` package must be importable by name: its model tests use
    # relative imports such as ``from ...causal_lm_tester import ...``.
    env["PYTHONPATH"] = os.pathsep.join([str(source), str(workdir), *path_entries])
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    env["_HF_TESTS_SOURCE"] = str(source)
    return env


def pytest_command(
    target: Path, marks: str, extra: list[str], collect_only: bool
) -> list[str]:
    """Build the pytest invocation.

    Deliberately minimal: no marker filtering and no plugin disabling by default,
    so what runs is what ``pytest tests/models/<module>/`` runs upstream. Making
    a platform look better by deselecting upstream tests would defeat the point
    of measuring coverage.
    """
    command = [sys.executable, "-m", "pytest"]
    if marks:
        command += ["-m", marks]
    if collect_only:
        command.append("--collect-only")
    command += extra
    command.append(str(target))
    return command


def run_tests(model: str, source: Path, args: argparse.Namespace) -> dict:
    """Run one architecture's official tests in an isolated subprocess."""
    target = test_dir(source, model)
    if not target.is_dir():
        return {
            "run": {"status": "NOT_IN_VERSION", "target": str(target)},
            "tests": [],
            "collect_errors": [],
            "summary": {},
        }

    # A private directory holds the per-run report and the recording plugin, so
    # concurrent runs never share either and the source cache keeps no artifacts.
    workdir = Path(tempfile.mkdtemp(prefix=f"hf-tests-{model.replace('/', '_')}-"))
    report = workdir / "report.jsonl"
    report.touch()
    (workdir / "hf_report_plugin.py").write_text(PLUGIN)
    # Transformers imports the spec as a module name, so it must sit on the
    # child's sys.path. The child runs from the source root (see below), so the
    # spec is staged there under a torch_fl-prefixed name and removed afterwards.
    source_spec = source / STAGED_SPEC_NAME
    shutil.copyfile(DEVICE_SPEC, source_spec)
    # Transformers 5.16.1's utils/ is a namespace package (no __init__.py). Some
    # test files do sys.path.append(".") then import from utils, which fails when
    # utils remains a namespace package. Convert it to a regular package.
    utils_init = source / "utils" / "__init__.py"
    utils_init_existed = utils_init.exists()
    if not utils_init_existed:
        utils_init.touch()

    env = child_env(source, args.device, report, args.offline, workdir)
    command = pytest_command(
        target,
        args.marks,
        [
            # HF's conftest.py and its marker declarations live in the source
            # tree; the child runs elsewhere, so point pytest at them explicitly.
            "-c",
            str(source / "pyproject.toml"),
            "--rootdir",
            str(source),
            # pytest requires a module name here, not a path. ``workdir`` is on
            # the child PYTHONPATH so the staged plugin resolves by name.
            "-p",
            "hf_report_plugin",
            *args.pytest_arg,
        ],
        args.collect_only,
    )

    started = time.time()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            # Several 5.16.1 test modules do sys.path.append(".") and then import
            # from the source tree's top-level ``utils`` package. Running anywhere
            # else turns those modules into collection errors unrelated to the
            # device under test.
            cwd=str(source),
            timeout=args.timeout,
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
    finally:
        source_spec.unlink(missing_ok=True)
        if not utils_init_existed:
            utils_init.unlink(missing_ok=True)

    reduced = reduce_records(read_report(report))
    combined = "\n".join(
        [
            stderr,
            stdout,
            *(item.get("detail") or "" for item in reduced["tests"]),
            *(item.get("detail") or "" for item in reduced["collect_errors"]),
        ]
    )
    poisoned = bool(POISON_RE.search(combined))
    # pytest returns 0-5 for test outcomes; anything else (or a signal) means the
    # interpreter died, which the per-test records cannot describe on their own.
    crashed = pytest_process_crashed(rc)

    result = {
        "run": {
            "status": "RAN",
            "target": str(target),
            "command": command,
            "returncode": rc,
            "timed_out": timed_out,
            "collect_only": args.collect_only,
            "crashed": crashed,
            "context_poison": poisoned,
            "duration_s": round(time.time() - started, 1),
            "collected": reduced["collected"],
            "report": str(report),
            "workdir": str(workdir),
            "stdout_tail": stdout.strip()[-4000:] or None,
            "stderr_tail": stderr.strip()[-4000:] or None,
        },
        "tests": reduced["tests"],
        "collect_errors": reduced["collect_errors"],
    }
    result["summary"] = summarize_statuses(reduced["tests"], reduced["collect_errors"])
    return result


def fingerprint(model: str, device: str, test: dict) -> str:
    """Fingerprint the cause, not the occurrence.

    Addresses, shapes, durations, and temporary paths are stripped so that a
    differing pointer value does not read as a different failure. The later
    reporter uses this for dedup; nothing here writes to a tracker.
    """
    detail = test.get("detail") or ""
    normalized = re.sub(r"0x[0-9a-f]+", "0xADDR", detail, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d+\.\d+s\b", "TIMEs", normalized)
    normalized = re.sub(r"/tmp/[^\s'\"]+", "/tmp/PATH", normalized)
    normalized = re.sub(r"\b\d+\b", "N", normalized)
    payload = f"{test['status']}|{model}|{device}|{test['nodeid']}|{normalized[-2000:]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def prepare_result(
    model: str, source: Path, args: argparse.Namespace, environment_data: dict
) -> dict:
    """Run one model and attach the stable top-level result metadata."""
    result = run_tests(model, source, args)
    result["schema_version"] = SCHEMA_VERSION
    result["model"] = {"requested": model, "module": module_name(model)}
    result["environment"] = dict(environment_data)
    result["verdict"] = verdict(result)
    for test in result["tests"]:
        if test["status"] in ("FAIL", "ERROR", "ENVIRONMENT_ERROR"):
            test["fingerprint"] = fingerprint(model, args.device, test)
    return result


def all_verdict(summary: dict, expected: int, completed: int) -> str:
    """Return the aggregate verdict for an all-architecture run.

    Ordered like the per-model verdict: the outcome that most invalidates the
    sweep wins. ``NOT_IN_VERSION`` is not a failure --- the registry lists
    architecture keys that the pinned version ships no test directory for, and
    counting those as defects would inflate every sweep.
    """
    if completed < expected:
        return "INCOMPLETE"
    for status in (
        "TIMEOUT",
        "CRASH",
        "COLLECT_ERROR",
        "ENVIRONMENT_ERROR",
        "FAIL",
        "NO_TESTS_RUN",
    ):
        if summary.get(status):
            return status
    if any(status not in ("PASS", "NOT_IN_VERSION") for status in summary):
        return "INCOMPLETE"
    return "PASS"


def aggregate_results(results: list[dict], architectures: list[str]) -> dict:
    """Build the stable all-architecture aggregate from completed model results.

    ``attempted`` is the number of architectures whose tests actually ran, which
    is the only denominator a coverage rate may use.
    """
    verdicts: dict[str, int] = {}
    for result in results:
        model_verdict = result["verdict"]
        verdicts[model_verdict] = verdicts.get(model_verdict, 0) + 1
    absent = verdicts.get("NOT_IN_VERSION", 0)
    return {
        "expected": len(architectures),
        "completed": len(results),
        "attempted": len(results) - absent,
        "not_in_version": absent,
        "architectures": architectures,
        "verdicts": verdicts,
    }


def all_result(
    results: list[dict], architectures: list[str], environment_data: dict
) -> dict:
    """Assemble the all-architecture result, writable while the sweep is running."""
    aggregate = aggregate_results(results, architectures)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "all",
        "environment": dict(environment_data),
        "models": results,
        "aggregate": aggregate,
        "verdict": all_verdict(
            aggregate["verdicts"], aggregate["expected"], aggregate["completed"]
        ),
    }


def summarize_all(result: dict) -> str:
    """Render a compact aggregate summary without hiding per-model JSON."""
    aggregate = result["aggregate"]
    lines = [
        f"mode       {result['mode']}",
        f"device     {result['environment']['device']}",
        f"verdict    {result['verdict']}",
        f"completed  {aggregate['completed']}/{aggregate['expected']}",
        f"attempted  {aggregate['attempted']}"
        f"  (not in version {aggregate['not_in_version']})",
    ]
    counts = aggregate["verdicts"]
    if counts:
        lines.append("")
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    interesting = [
        model
        for model in result["models"]
        if model["verdict"] not in ("PASS", "NOT_IN_VERSION")
    ]
    if interesting:
        lines.append("")
        for model in interesting:
            lines.append(f"  {model['verdict']:<18} {model['model']['requested']}")
    return "\n".join(lines)


def summarize(result: dict) -> str:
    run = result["run"]
    lines = [
        f"model      {result['model']['requested']} "
        f"(tests/models/{result['model']['module']})",
        f"device     {result['environment']['device']}",
        f"verdict    {result['verdict']}",
        f"duration   {run.get('duration_s')}s",
    ]
    if run["status"] == "NOT_IN_VERSION":
        lines.append(f"note       no test directory at {run['target']}")
        return "\n".join(lines)
    if run.get("context_poison"):
        lines.append("WARNING    device context was poisoned; later tests are void")
    if run.get("timed_out"):
        lines.append(f"WARNING    timed out; partial results from {run['report']}")
    counts = result["summary"]
    lines.append("")
    lines.append(
        "  "
        + "  ".join(f"{name}={value}" for name, value in sorted(counts.items()))
        + (f"  (collected {run.get('collected')})" if run.get("collected") else "")
    )
    interesting = [
        test
        for test in result["tests"]
        if test["status"] in ("FAIL", "ERROR", "ENVIRONMENT_ERROR")
    ]
    if interesting:
        lines.append("")
        for test in interesting[:20]:
            first = (test.get("detail") or "").strip().splitlines()
            lines.append(f"  {test['status']:<18} {test['nodeid']}")
            if first:
                lines.append(f"    {first[-1][:160]}")
        if len(interesting) > 20:
            lines.append(f"  ... {len(interesting) - 20} more in the JSON result")
    for error in result["collect_errors"]:
        lines.append(f"  {error['status']:<18} {error['nodeid']} (collection)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HuggingFace's official Transformers tests on an accelerator"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model", help="run every official test for one architecture, e.g. qwen3"
    )
    selection.add_argument(
        "--all",
        dest="all_models",
        action="store_true",
        help="run every architecture in the installed Transformers registry",
    )
    selection.add_argument(
        "--list-models", action="store_true", help="list available architectures"
    )
    parser.add_argument("--device", default="flagos")
    parser.add_argument(
        "--transformers-version",
        default="latest",
        help="'latest' records the installed release; an explicit version must match it",
    )
    parser.add_argument("--source-dir", help="prepared transformers source tree")
    parser.add_argument("--cache-dir", help="where version-matched sources are cached")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never download: require a cached source tree and set HF offline mode",
    )
    parser.add_argument(
        "--marks",
        default="",
        help="optional pytest -m expression (disabled by default)",
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", type=Path, help="write the JSON result here")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="extra argument passed through to pytest (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for name in known_models():
            print(name)
        return 0
    if not args.model and not args.all_models:
        parser.error("choose exactly one of --model, --all, or --list-models")

    try:
        resolution = resolve_version(args.transformers_version, args.offline)
        source = use_source(
            resolution["version"],
            args.source_dir,
            args.cache_dir,
            args.offline,
        )
    except SourceError as exc:
        print(f"environment error: {exc}", file=sys.stderr)
        return 2

    env = environment(args.device)
    env["transformers_requested"] = resolution["requested"]
    env["transformers_latest"] = resolution["latest"]
    env["source_path"] = source["path"]
    env["source_version"] = source["version"]

    source_path = Path(source["path"])
    print(
        f"transformers {env['transformers']}  torch {env['torch']}"
        f"  torch_fl {env['torch_fl_commit']}"
    )
    print(f"source       {source['path']}")
    print()

    if args.all_models:
        models = sweep_models()
        results: list[dict] = []
        print(f"sweeping {len(models)} architectures, one subprocess each\n")
        for index, model in enumerate(models, start=1):
            print(f"[{index}/{len(models)}] {model}")
            results.append(prepare_result(model, source_path, args, env))
            print(summarize(results[-1]))
            print()
            # A full sweep is long enough that an interrupted run must still
            # leave the architectures already measured on disk.
            if args.out:
                atomic_write(args.out, all_result(results, models, env))
        result = all_result(results, models, env)
        print(summarize_all(result))
    else:
        result = prepare_result(args.model, source_path, args, env)
        print(summarize(result))

    if args.out:
        atomic_write(args.out, result)
        print(f"\nJSON written to {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
