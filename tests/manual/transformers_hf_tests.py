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

Exit codes separate "what was measured" from "whether it worked":
0 the measurement is clean, 1 something was measured and something failed,
2 nothing was measured because the environment, the source tree, or the
preflight check in :func:`preflight` failed. A caller must not read 2 as a
coverage result.

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
    device_name,
    resolve_version,
    use_source,
)

HARNESS_VERSION = 1
SCHEMA_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_SPEC = Path(__file__).resolve().parent / "hf_device_spec.py"


def spec_device() -> str:
    """The accelerator the official tests execute on.

    The device spec is the contract HuggingFace reads, so its ``DEVICE_NAME``
    is the only name this runner may report.  A caller-supplied device string
    cannot be verified against the spec and can therefore be silently wrong.
    """
    return device_name(DEVICE_SPEC)


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
    "BATCH_CRASHED",
)

MARK = "@@HFTEST@@"


# The plugin runs inside the pytest subprocess. It is written to a temporary
# file and loaded with ``-p`` rather than added as a dependency: a new runtime
# dependency for a manual harness would have to be installed on every vendor box
# before that box could be measured at all.
PLUGIN = r"""
import json
import os
import re


FALLBACK_RE = re.compile(r"\[flagos cpu_fallback\]\s+(\S+)")


def _text(value):
    if value is None:
        return None
    text = str(value)
    if len(text) <= 8000:
        return text
    return text[:4000] + "\n... <truncated> ...\n" + text[-4000:]


def _fallback_ops(sections):
    return sorted(
        {
            op
            for _, content in sections
            for op in FALLBACK_RE.findall(str(content))
        }
    )


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
            sections = list(report.sections[:6])
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
                        [name, _text(content)] for name, content in sections
                    ],
                    "cpu_fallback_ops": _fallback_ops(sections),
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
    import torch_fl  # noqa: F401 - register the custom device before collection

    path = os.environ["HF_TEST_REPORT"]
    config.pluginmanager.register(Recorder(path), "torch_fl_hf_recorder")


def pytest_collection_modifyitems(config, items):
    import pytest

    if os.environ.get("HF_TEST_SKIP_FLEX_ATTENTION") == "1":
        skip = pytest.mark.skip(reason="flex attention requires a CUDA Triton backend")
        for item in items:
            if "flex_attention" in item.nodeid:
                item.add_marker(skip)
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
                "cpu_fallback_ops": [],
            },
        )
        entry["cpu_fallback_ops"] = sorted(
            set(entry["cpu_fallback_ops"]) | set(record.get("cpu_fallback_ops") or [])
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


def canonicalize_nodeids(tests: list[dict], selected: list[str]) -> None:
    """Restore the file part pytest drops from a selected nodeid.

    A nodeid passed on the command line comes back from the plugin as
    ``::BertModelTest::test_x``: pytest builds the reported nodeid from the
    directory it was pointed at rather than from the file that holds the test,
    and the file part is lost. Every recorded nodeid that reaches triage, the
    verifier, and the isolation rerun has to be selectable again, so each is
    restored from the nodeid that was actually selected, matched on its
    ``::Class::test`` tail. A tail shared by two selected nodeids is left alone:
    guessing which file was meant would be worse than the reported form.
    """
    tails: dict[str, list[str]] = {}
    for nodeid in selected:
        tail = nodeid[nodeid.find("::") :] if "::" in nodeid else ""
        if tail:
            tails.setdefault(tail, []).append(nodeid)

    for test in tests:
        nodeid = test.get("nodeid") or ""
        tail = nodeid[nodeid.find("::") :] if "::" in nodeid else ""
        candidates = tails.get(tail) or []
        if len(candidates) == 1:
            test["nodeid"] = candidates[0]


def fallback_ops(tests: list[dict]) -> list[str]:
    """Return the unique operators that executed through CPU fallback."""
    return sorted({op for test in tests for op in test.get("cpu_fallback_ops", [])})


def summarize_statuses(tests: list[dict], collect_errors: list[dict]) -> dict:
    counts = {name: 0 for name in STATUSES}
    for test in tests:
        status = test["status"]
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    for error in collect_errors:
        status = error["status"]
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
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
    if counts.get("BATCH_CRASHED"):
        return "CRASH"
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
    # The device spec imports torch_fl after torch. Disable unrelated torch
    # backend entry points first so a broken optional plugin cannot abort startup.
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
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


def child_env(source: Path, report: Path | None = None, offline: bool = False) -> dict:
    """Build the environment HuggingFace's device injection contract needs."""
    env = dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["FLAGOS_LOG"] = "fallback"
    # Custom PrivateUse1 names are registered by the spec. Transformers validates
    # TRANSFORMERS_TEST_DEVICE before importing that spec, so setting it to
    # ``flagos`` would fail at torch.device() validation.
    env.pop("TRANSFORMERS_TEST_DEVICE", None)
    env["TRANSFORMERS_TEST_DEVICE_SPEC"] = "hf_device_spec.py"
    if report is not None:
        env["HF_TEST_REPORT"] = str(report)
    # This repository also has a top-level ``tests`` package. If it stays ahead
    # of the source tree, HF's ``tests.models...`` imports resolve into the wrong
    # tree and every model test errors on import. The source tree therefore comes
    # first, and a caller-supplied repository root is moved to the end instead of
    # being dropped: ``hf_device_spec.py`` imports ``torch_fl``, which a checkout
    # that was never installed cannot provide from a private work directory, and
    # discarding the caller's own PYTHONPATH turned a configured interpreter into
    # "the child died while starting up".  A caller that wants the installed
    # build simply leaves the repository off PYTHONPATH.
    entries = [
        str(Path(entry).resolve())
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    repo_entries = [entry for entry in entries if entry == str(REPO_ROOT)]
    path_entries = [entry for entry in entries if entry != str(REPO_ROOT)]
    # HF's ``tests`` package must be importable by name: its model tests use
    # relative imports such as ``from ...causal_lm_tester import ...``.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), str(source / "utils"), *path_entries, *repo_entries]
    )
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    env["_HF_TESTS_SOURCE"] = str(source)
    return env


# The preflight runs in the child environment, because the failures worth
# catching here --- a hook the spec does not expose, a device name that does not
# match the registration, an accelerator the driver cannot see --- appear only
# once the device spec is imported in the process that will execute the tests.
PREFLIGHT = r"""
import importlib.util
import json
import sys

spec_path, expected_device, expected_version, report_path = sys.argv[1:5]
report = {"problems": []}
problems = report["problems"]


def check_hook(name):
    value = getattr(module, name, None)
    report.setdefault("hooks", {})[name] = callable(value)
    if not callable(value):
        problems.append(f"the device spec exposes no callable {name}")
    return value


try:
    import torch_fl

    report["torch_fl"] = getattr(torch_fl, "__file__", "unknown")
except Exception as exc:
    report["torch_fl"] = None
    problems.append(f"torch_fl did not import: {exc!r}")

try:
    import torch

    report["registered_device"] = torch._C._get_privateuse1_backend_name()
except Exception as exc:
    report["registered_device"] = None
    problems.append(f"could not read the registered PrivateUse1 name: {exc!r}")

module = None
count_fn = None
try:
    loader = importlib.util.spec_from_file_location("hf_device_spec_preflight", spec_path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    report["spec_device"] = getattr(module, "DEVICE_NAME", None)
    count_fn = check_hook("DEVICE_COUNT_FN")
    check_hook("MANUAL_SEED_FN")
    check_hook("EMPTY_CACHE_FN")
except Exception as exc:
    problems.append(f"the device spec did not import: {exc!r}")

report["expected_device"] = expected_device
# Every comparison below decides whether a run is declared invalid, so what it
# learns is the caller's only diagnosis of one that is.  None of it may abort
# this child: a check that raised here --- reading a key the spec never
# published, say --- used to end the process before it wrote anything, and the
# caller was left with "the preflight published no verdict" and no reason for it.
try:
    if module is not None and report.get("spec_device") != expected_device:
        problems.append(
            f"the device spec names {report.get('spec_device')!r},"
            f" not {expected_device!r}"
        )
    registered = report.get("registered_device")
    if registered is not None and registered != expected_device:
        problems.append(
            f"torch_fl registers {registered!r} but the tests run on {expected_device!r}"
        )

    if callable(count_fn):
        try:
            report["device_count"] = count_fn()
        except Exception as exc:
            report["device_count"] = None
            problems.append(f"enumerating accelerator devices failed: {exc!r}")
        else:
            count = report["device_count"]
            if not isinstance(count, int) or count < 1:
                problems.append(
                    "no accelerator is visible: device_count() reported "
                    f"{count!r}"
                )

    try:
        import transformers

        report["transformers"] = transformers.__version__
    except Exception as exc:
        report["transformers"] = None
        problems.append(f"transformers did not import: {exc!r}")
    if report.get("transformers") not in (None, expected_version):
        problems.append(
            f"transformers {report['transformers']} is installed but the source tree "
            f"is {expected_version}"
        )
except Exception as exc:
    problems.append(f"the environment checks could not run: {exc!r}")

with open(report_path, "w") as file:
    json.dump(report, file, indent=1, sort_keys=True)
"""


def read_preflight(path: Path) -> dict | None:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return report if isinstance(report, dict) else None


def preflight(source: Path, version: str, offline: bool, timeout: int = 600) -> dict:
    """Prove the child environment can execute the official tests at all.

    The check runs in a subprocess built by :func:`child_env` --- the same
    environment the tests get --- because a parent that imports ``torch``
    successfully says nothing about a child that must import the device spec,
    claim the accelerator and find its hooks.  A failure here invalidates the
    whole measurement, so it is reported before the first test runs rather than
    inferred from tens of thousands of errors afterwards.
    """
    device = spec_device()
    command: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="hf-preflight-"))
    try:
        spec = workdir / DEVICE_SPEC.name
        script = workdir / "hf_preflight.py"
        report_path = workdir / "preflight.json"
        shutil.copyfile(DEVICE_SPEC, spec)
        script.write_text(PREFLIGHT)

        command = [
            sys.executable,
            str(script),
            str(spec),
            device,
            version,
            str(report_path),
        ]
        started = time.time()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=child_env(source, offline=offline),
                cwd=str(workdir),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "ran": True,
                "device": device,
                "duration_s": round(time.time() - started, 1),
                "command": " ".join(command),
                "problems": [f"the preflight did not finish within {timeout}s"],
            }

        report = read_preflight(report_path)
        if report is None:
            return {
                "ran": True,
                "device": device,
                "duration_s": round(time.time() - started, 1),
                "command": " ".join(command),
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr.strip()[-2000:] or None,
                "problems": [
                    "the preflight published no verdict "
                    f"(exit {proc.returncode}); the child died while starting up"
                ],
            }
        report.update(
            {
                "ran": True,
                "duration_s": round(time.time() - started, 1),
                "command": " ".join(command),
                "returncode": proc.returncode,
            }
        )
        report.setdefault("problems", [])
        return report
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def pytest_command(
    target: Path,
    marks: str,
    extra: list[str],
    collect_only: bool,
    include_target: bool = True,
) -> list[str]:
    """Build the pytest invocation.

    Deliberately minimal: no marker filtering and no plugin disabling by default,
    so what runs is what ``pytest tests/models/<module>/`` runs upstream. Making
    a platform look better by deselecting upstream tests would defeat the point
    of measuring coverage.

    ``include_target`` must be ``False`` when ``extra`` already carries specific
    nodeids (e.g. batch/isolation runs): pytest treats a bare directory arg as
    "collect everything under here" regardless of any nodeids also passed, so
    appending both silently reruns the whole suite instead of the selection.
    """
    command = [sys.executable, "-m", "pytest"]
    if marks:
        command += ["-m", marks]
    if collect_only:
        command.append("--collect-only")
    command += extra
    if include_target:
        command.append(str(target))
    return command


def collect_all_tests(model: str, source: Path, args: argparse.Namespace) -> dict:
    """Collect all test nodeids without running them."""
    target = test_dir(source, model)
    if not target.is_dir():
        return {"tests": [], "collected": 0, "error": "target not found"}

    workdir = Path(tempfile.mkdtemp(prefix=f"hf-collect-{model.replace('/', '_')}-"))
    try:
        (workdir / "hf_report_plugin.py").write_text(PLUGIN)
        shutil.copyfile(DEVICE_SPEC, workdir / DEVICE_SPEC.name)
        (workdir / "tests").symlink_to(source / "tests", target_is_directory=True)
        (workdir / "src").symlink_to(source / "src", target_is_directory=True)

        env = child_env(source, workdir / "dummy.jsonl", args.offline)
        env["PYTHONPATH"] = os.pathsep.join([str(workdir), env["PYTHONPATH"]])

        command = pytest_command(
            target,
            args.marks,
            [
                "-c",
                str(source / "pyproject.toml"),
                "--rootdir",
                str(source),
                "--collect-only",
                "-q",
            ],
            collect_only=True,
        )

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(workdir),
            timeout=120,
        )

        # Parse pytest --collect-only output
        test_nodeids = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if "::" in line and not line.startswith("="):
                # Clean up pytest's output format
                nodeid = line.split()[0] if " " in line else line
                if nodeid and nodeid.endswith("py"):
                    continue  # Skip file-level entries
                test_nodeids.append(nodeid)

        return {"tests": test_nodeids, "collected": len(test_nodeids)}

    except Exception as e:
        return {"tests": [], "collected": 0, "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_test_batch(
    model: str,
    source: Path,
    batch_nodeids: list[str],
    args: argparse.Namespace,
    batch_timeout: int,
) -> dict:
    """Run a batch of specific tests."""
    target = test_dir(source, model)
    workdir = Path(tempfile.mkdtemp(prefix=f"hf-batch-{model.replace('/', '_')}-"))

    try:
        report = workdir / "report.jsonl"
        report.touch()
        (workdir / "hf_report_plugin.py").write_text(PLUGIN)
        shutil.copyfile(DEVICE_SPEC, workdir / DEVICE_SPEC.name)
        (workdir / "tests").symlink_to(source / "tests", target_is_directory=True)
        (workdir / "src").symlink_to(source / "src", target_is_directory=True)

        env = child_env(source, report, args.offline)
        env["HF_TEST_SKIP_FLEX_ATTENTION"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(workdir), env["PYTHONPATH"]])

        # Build command with specific test nodeids
        command = pytest_command(
            target,
            args.marks,
            [
                "-c",
                str(source / "pyproject.toml"),
                "--rootdir",
                str(source),
                "-p",
                "hf_report_plugin",
                *args.pytest_arg,
            ]
            + batch_nodeids,  # Add nodeids to select specific tests
            collect_only=False,
            include_target=False,
        )

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(workdir),
            timeout=batch_timeout,
        )

        reduced = reduce_records(read_report(report))
        canonicalize_nodeids(reduced["tests"], batch_nodeids)
        crashed = pytest_process_crashed(proc.returncode)

        result = {
            "tests": reduced["tests"],
            "collect_errors": reduced["collect_errors"],
            "collected": reduced["collected"],
            "crashed": crashed,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.strip()[-2000:] or None,
            "stderr_tail": proc.stderr.strip()[-2000:] or None,
        }
        result["cpu_fallback_ops"] = fallback_ops(result["tests"])
        return result

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")

        # Try to read whatever got written before timeout
        reduced = reduce_records(read_report(report))

        result = {
            "tests": reduced["tests"],
            "collect_errors": reduced["collect_errors"],
            "collected": reduced["collected"],
            "crashed": True,
            "timed_out": True,
            "returncode": None,
            "stdout_tail": stdout.strip()[-2000:] or None,
            "stderr_tail": stderr.strip()[-2000:] or None,
        }
        result["cpu_fallback_ops"] = fallback_ops(result["tests"])
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def reset_device_context(device: str):
    """Attempt to reset device state after a crash."""
    try:
        import torch

        if device == "flagos":
            torch.flagos.empty_cache()
            torch.flagos.synchronize()
    except Exception:
        pass  # Best effort, ignore failures


def run_tests_resilient(model: str, source: Path, args: argparse.Namespace) -> dict:
    """Resilient mode: run tests in batches, continue on crash."""
    device = spec_device()
    target = test_dir(source, model)
    if not target.is_dir():
        return {
            "run": {"status": "NOT_IN_VERSION", "target": str(target)},
            "tests": [],
            "collect_errors": [],
            "summary": {},
        }

    print(f"[Resilient mode] Collecting tests for {model}...")
    collected = collect_all_tests(model, source, args)

    if collected["collected"] == 0:
        return {
            "run": {
                "status": "NO_TESTS_COLLECTED",
                "error": collected.get("error"),
            },
            "tests": [],
            "collect_errors": [],
            "summary": {},
        }

    all_test_nodeids = collected["tests"]
    print(f"[Resilient mode] Collected {len(all_test_nodeids)} tests")

    batch_size = args.batch_size
    batch_timeout = args.batch_timeout
    all_tests = []
    all_collect_errors = []
    crashed_batches = []
    total_batches = (len(all_test_nodeids) + batch_size - 1) // batch_size

    started = time.time()

    for batch_idx, start in enumerate(range(0, len(all_test_nodeids), batch_size)):
        batch = all_test_nodeids[start : start + batch_size]
        batch_name = f"{batch_idx + 1}/{total_batches}"

        print(f"[Batch {batch_name}] Running {len(batch)} tests...")

        try:
            batch_result = run_test_batch(model, source, batch, args, batch_timeout)

            if batch_result.get("crashed") or batch_result.get("timed_out"):
                print(
                    f"[Batch {batch_name}] ✗ CRASHED (rc={batch_result.get('returncode')})"
                )
                crashed_batches.append(
                    {
                        "batch_idx": batch_idx,
                        "test_count": len(batch),
                        "first_nodeid": batch[0] if batch else None,
                        "crashed": batch_result.get("crashed", False),
                        "timed_out": batch_result.get("timed_out", False),
                    }
                )

                reported_nodeids = {test["nodeid"] for test in batch_result["tests"]}
                all_tests.extend(batch_result["tests"])
                all_collect_errors.extend(batch_result["collect_errors"])

                # Keep completed per-test evidence from a crashed batch. Only
                # nodeids with no report are unknown; marking the whole batch as
                # crashed would overwrite real PASS/FAIL results.
                for nodeid in batch:
                    if nodeid not in reported_nodeids:
                        all_tests.append(
                            {
                                "nodeid": nodeid,
                                "status": "BATCH_CRASHED",
                                "detail": f"Batch {batch_name} crashed or timed out before reporting this test",
                                "cpu_fallback_ops": [],
                            }
                        )

                # Try to release cached allocations. A poisoned accelerator
                # context may require a process/driver reset; this is best effort.
                reset_device_context(device)

            else:
                print(f"[Batch {batch_name}] ✓ {len(batch_result['tests'])} results")
                all_tests.extend(batch_result["tests"])
                all_collect_errors.extend(batch_result["collect_errors"])

        except Exception as e:
            print(f"[Batch {batch_name}] ✗ Exception: {e}")
            crashed_batches.append(
                {
                    "batch_idx": batch_idx,
                    "test_count": len(batch),
                    "first_nodeid": batch[0] if batch else None,
                    "exception": str(e),
                }
            )

            for nodeid in batch:
                all_tests.append(
                    {
                        "nodeid": nodeid,
                        "status": "BATCH_CRASHED",
                        "detail": f"Batch {batch_name} exception: {e}",
                        "cpu_fallback_ops": [],
                    }
                )

            reset_device_context(device)

    duration = round(time.time() - started, 1)

    # Detect poison in all test details
    combined = "\n".join(item.get("detail") or "" for item in all_tests)
    poisoned = bool(POISON_RE.search(combined))

    result = {
        "run": {
            "status": "COMPLETED_RESILIENT",
            "target": str(target),
            "duration_s": duration,
            "collected": len(all_test_nodeids),
            "crashed_batches": crashed_batches,
            "total_batches": total_batches,
            "context_poison": poisoned,
            "batch_size": batch_size,
            "batch_timeout": batch_timeout,
            "cpu_fallback_ops": fallback_ops(all_tests),
        },
        "tests": all_tests,
        "collect_errors": all_collect_errors,
    }

    result["summary"] = summarize_statuses(all_tests, all_collect_errors)
    return result


def run_tests(model: str, source: Path, args: argparse.Namespace) -> dict:
    """Run one architecture's official tests in an isolated subprocess."""
    # Check if resilient mode is enabled. getattr keeps helper-level callers
    # compatible with a minimal argparse namespace in unit tests and scripts.
    if getattr(args, "resilient", False):
        return run_tests_resilient(model, source, args)

    target = test_dir(source, model)
    if not target.is_dir():
        return {
            "run": {"status": "NOT_IN_VERSION", "target": str(target)},
            "tests": [],
            "collect_errors": [],
            "summary": {},
        }

    # Transformers resolves TRANSFORMERS_TEST_DEVICE_SPEC by importing the value
    # as a module name after appending its directory to sys.path, so the spec has
    # to be importable from the child's working directory. Running from a private
    # directory keeps the shared source cache unmodified.
    workdir = Path(tempfile.mkdtemp(prefix=f"hf-tests-{model.replace('/', '_')}-"))
    report = workdir / "report.jsonl"
    report.touch()
    (workdir / "hf_report_plugin.py").write_text(PLUGIN)
    shutil.copyfile(DEVICE_SPEC, workdir / DEVICE_SPEC.name)
    (workdir / "tests").symlink_to(source / "tests", target_is_directory=True)
    (workdir / "src").symlink_to(source / "src", target_is_directory=True)
    env = child_env(source, report, args.offline)
    env["HF_TEST_SKIP_FLEX_ATTENTION"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(workdir), env["PYTHONPATH"]])
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
            cwd=str(workdir),
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
            "cpu_fallback_ops": fallback_ops(reduced["tests"]),
        },
        "tests": reduced["tests"],
        "collect_errors": reduced["collect_errors"],
    }
    result["summary"] = summarize_statuses(reduced["tests"], reduced["collect_errors"])
    return result


def occurrence_fingerprint(model: str, device: str, test: dict) -> str:
    """Name one failing occurrence, not the cause behind it.

    The nodeid is part of the payload, so two tests that fail for the same
    reason get two fingerprints. Causes are fingerprinted later, by the
    reporter, from the class, component, subject and mechanism; keeping the two
    distinct is what stops a sweep from counting one defect 400 times.
    """
    detail = test.get("detail") or ""
    normalized = re.sub(r"0x[0-9a-f]+", "<ADDR>", detail, flags=re.IGNORECASE)
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
    device = environment_data.get("device") or spec_device()
    result["schema_version"] = SCHEMA_VERSION
    result["model"] = {"requested": model, "module": module_name(model)}
    result["environment"] = dict(environment_data)
    result["verdict"] = verdict(result)
    for test in result["tests"]:
        if test["status"] in ("FAIL", "ERROR", "ENVIRONMENT_ERROR"):
            test["occurrence_fingerprint"] = occurrence_fingerprint(model, device, test)
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
        lines.append(f"WARNING    timed out; partial results from {run.get('report')}")
    if run.get("cpu_fallback_ops"):
        lines.append(
            "WARNING    CPU fallback operators: " + ", ".join(run["cpu_fallback_ops"])
        )
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


def invalid_result(environment_data: dict, report: dict) -> dict:
    """A result that says no measurement happened, and why.

    Written even when the run aborts, so a pipeline that only reads the JSON
    cannot mistake "the environment was broken" for "nothing failed".
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "invalid",
        "environment": dict(environment_data) | {"preflight": report},
        "error": "; ".join(report["problems"]),
        "verdict": "ENVIRONMENT_ERROR",
    }


def measured_anything(result: dict) -> bool:
    """True when at least one test actually produced a result."""
    models = result.get("models")
    if models is None:
        models = [result]
    return any(
        test.get("status") == "PASS"
        for model in models
        for test in model.get("tests", [])
    )


def exit_code(result: dict) -> int:
    """Map a finished run to its process exit code.

    0  the measurement is clean
    1  something was measured and something failed
    2  nothing was measured: the environment, the source, or the preflight failed

    The distinction matters to the caller: 1 is a report, 2 is a pipeline that
    must not be summarized as a coverage result.
    """
    if result.get("error"):
        return 2
    verdict = result["verdict"]
    if verdict == "PASS":
        return 0
    # An environment error means the assertions never ran. Only when it is the
    # whole outcome is the measurement itself invalid; alongside real results
    # it is one more finding to report.
    if verdict == "ENVIRONMENT_ERROR" and not measured_anything(result):
        return 2
    return 1


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
    parser.add_argument(
        "--resilient",
        action="store_true",
        help="resilient mode: run tests in batches, continue on crash",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="tests per batch in resilient mode (default: 20)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=900,
        help="timeout per batch in seconds (default: 900 = 15 minutes)",
    )
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
        device = spec_device()
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

    source_path = Path(source["path"])
    report = preflight(source_path, resolution["version"], args.offline)
    if report["problems"]:
        print(
            f"environment error: {len(report['problems'])} problem(s)",
            file=sys.stderr,
        )
        for problem in report["problems"]:
            print(f"  - {problem}", file=sys.stderr)
        if args.out:
            atomic_write(
                args.out,
                invalid_result(
                    {"device": device, "source_path": source["path"]}, report
                ),
            )
        return 2

    env = environment(device)
    env["transformers_requested"] = resolution["requested"]
    env["transformers_latest"] = resolution["latest"]
    env["source_path"] = source["path"]
    env["source_version"] = source["version"]
    env["preflight"] = report

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
    return exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
