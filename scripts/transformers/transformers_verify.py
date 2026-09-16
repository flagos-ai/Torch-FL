#!/usr/bin/env python3
"""Verify Transformers findings in fresh pytest subprocesses."""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_SPEC = REPO_ROOT / "tests" / "manual" / "hf_device_spec.py"
SOURCE_HELPER = REPO_ROOT / "tests" / "manual" / "transformers_hf_source.py"

# pytest's summary line is the only place the isolated run states how many tests
# it ran; ``in 10.30s`` identifies that line.
SUMMARY_LINE_RE = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")
OUTCOME_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)\b"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers_triage import generate_fingerprint  # noqa: E402 - sibling tool


def load_source_helper():
    """Load the runner's cache helper by path.

    The manual test tree is not an importable package, and its cache root is
    the only place a version-matched source tree is guaranteed to exist.
    """
    spec = importlib.util.spec_from_file_location(
        "transformers_hf_source", SOURCE_HELPER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recorded_version(findings_json: Dict) -> Optional[str]:
    """The transformers version the findings were measured against.

    Read from the run's own environment instead of from the cache directory
    listing: a cache holding several versions must not verify a finding
    against a source tree that did not produce it.
    """
    environment = findings_json.get("environment") or {}
    return environment.get("source_version") or environment.get("transformers")


def isolated_env(test_source_dir: Path, workdir: Path) -> dict[str, str]:
    """Build the same PrivateUse1 test environment as the official runner."""
    env = dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["FLAGOS_LOG_FALLBACK"] = "1"
    env.pop("TRANSFORMERS_TEST_DEVICE", None)
    env["TRANSFORMERS_TEST_DEVICE_SPEC"] = "hf_device_spec.py"
    # This mirrors the runner's ``child_env``. This repository also has a
    # top-level ``tests`` package, so the source tree stays ahead of the
    # repository root; the repository root is moved to the end rather than
    # dropped, because ``hf_device_spec.py`` imports ``torch_fl`` and a checkout
    # that was never installed can only provide it through the caller's
    # PYTHONPATH. Dropping it made every isolated run die of
    # ``ModuleNotFoundError: No module named 'torch_fl'`` before collecting a
    # test, which the verifier could only record as ``ERROR``.
    entries = [
        str(Path(entry).resolve())
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    repo_entries = [entry for entry in entries if entry == str(REPO_ROOT)]
    path_entries = [entry for entry in entries if entry != str(REPO_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(workdir),
            str(test_source_dir),
            str(test_source_dir / "utils"),
            *path_entries,
            *repo_entries,
        ]
    )
    return env


def isolated_outcomes(output: str) -> tuple[int, set[str]]:
    """The test count and outcome names pytest reported for one isolated run.

    "An isolation result is valid only when pytest collected exactly one test"
    is the documented rule, and this is the only place that can enforce it: a
    nodeid pytest cannot select exits with a usage error, which used to be
    recorded as an ordinary ``ERROR`` beside real per-test evidence. Reading the
    last line that carries a duration keeps a traceback mentioning "2 failed"
    from being counted as a result.
    """
    lines = [line for line in output.splitlines() if SUMMARY_LINE_RE.search(line)]
    if not lines:
        return 0, set()
    matches = OUTCOME_COUNT_RE.findall(lines[-1])
    return sum(int(count) for count, _ in matches), {
        name.rstrip("s") for _, name in matches
    }


def run_isolated_test(
    nodeid: str,
    test_source_dir: Path,
    timeout: int = 120,
) -> Dict:
    """Run exactly one nodeid in a fresh subprocess."""
    normalized_nodeid = nodeid.removeprefix(str(test_source_dir) + os.sep)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(test_source_dir / "pyproject.toml"),
        "--rootdir",
        str(test_source_dir),
        normalized_nodeid,
        "-q",
        "-p",
        "no:warnings",
        "--tb=short",
    ]
    command_str = " ".join(cmd)
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="hf-verify-") as tmp:
        workdir = Path(tmp)
        try:
            (workdir / DEVICE_SPEC.name).write_text(DEVICE_SPEC.read_text())
            (workdir / "tests").symlink_to(
                test_source_dir / "tests", target_is_directory=True
            )
            (workdir / "src").symlink_to(
                test_source_dir / "src", target_is_directory=True
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env=isolated_env(test_source_dir, workdir),
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - started
            return {
                "status": "TIMEOUT",
                "detail": f"Test exceeded {timeout}s timeout",
                "duration_s": round(duration, 1),
                "command": command_str,
                "returncode": None,
            }
        except (OSError, ValueError) as exc:
            duration = time.time() - started
            return {
                "status": "ERROR",
                "detail": f"Could not run isolated test: {exc}",
                "duration_s": round(duration, 1),
                "command": command_str,
                "returncode": None,
            }

    duration = time.time() - started
    combined = result.stdout + result.stderr
    reported, outcomes = isolated_outcomes(combined)
    if reported != 1:
        # A nodeid that cannot be selected, or a run that collected the whole
        # directory, says nothing about this finding. Recording it as anything
        # but ERROR would turn an empty rerun into evidence.
        status = "ERROR"
        detail = (
            f"the isolated run reported {reported} tests instead of one; the "
            "nodeid did not select a single test\n" + combined[-8000:]
        )
    elif "error" in outcomes:
        # A setup or teardown failure is not a per-test device defect.
        status = "ERROR"
        detail = combined[-8000:]
    elif result.returncode == 0:
        status = "SKIP" if "skipped" in outcomes else "PASS"
        detail = combined[-8000:]
    elif result.returncode == 1:
        status = "FAIL"
        detail = combined[-8000:]
    else:
        status = "ERROR"
        detail = combined[-8000:]

    return {
        "status": status,
        "detail": detail,
        "duration_s": round(duration, 1),
        "command": command_str,
        "returncode": result.returncode,
        "reported_tests": reported,
    }


def determine_verdict(isolation_status: str, original_class: str) -> str:
    """Map an isolation outcome to a filing verdict.

    ``original_class`` is not used to decide the verdict, but it is part of the
    signature because the verdict only means something for the class it was
    measured on: :func:`apply_isolation` owns the class refinement.
    """
    del original_class
    if isolation_status in ("FAIL", "TIMEOUT"):
        return "CONFIRMED"
    if isolation_status in ("PASS", "SKIP"):
        return "COLLATERAL"
    return "INCONCLUSIVE"


def apply_isolation(finding: Dict, isolation_result: Dict) -> None:
    """Record what an isolated re-run showed about one finding.

    A test that hangs alone is a crash-shaped defect. The batch classification
    saw it alongside hundreds of other failures of the same run, so isolation is
    the better evidence for the class, and the fingerprint is recomputed to stay
    the hash of the class it now carries.
    """
    status = isolation_result["status"]
    finding["isolation_status"] = status
    finding["isolation_detail"] = isolation_result["detail"]
    finding["isolation_duration_s"] = isolation_result["duration_s"]
    finding["isolation_command"] = isolation_result["command"]
    finding["isolation_reported_tests"] = isolation_result.get("reported_tests")
    if status == "TIMEOUT" and finding["class"] != "CRASH":
        finding["isolation_note"] = (
            f"the isolated run timed out; reclassified from {finding['class']} to CRASH"
        )
        finding["class"] = "CRASH"
        finding["fingerprint"] = generate_fingerprint(
            "CRASH",
            finding.get("component", "unknown"),
            finding["subject"],
            finding["mechanism"],
        )
    finding["verdict"] = determine_verdict(status, finding["class"])


def verify_findings(
    findings_json: Dict,
    test_source_dir: Path,
    timeout: int,
    max_workers: Optional[int],
) -> Dict:
    """Verify findings serially unless parallelism was explicitly requested."""
    findings = findings_json["findings"]
    pending = [f for f in findings if f.get("verification_required", True)]
    if not pending:
        findings_json["summary"]["verified"] = {
            "CONFIRMED": sum(f.get("verdict") == "CONFIRMED" for f in findings)
        }
        return findings_json
    workers = max_workers or 1
    if workers != 1:
        print(
            "Error: parallel accelerator verification is not supported because "
            "subprocesses may share one device context. Use --workers 1."
        )
        for finding in pending:
            finding["isolation_status"] = "ERROR"
            finding["isolation_detail"] = "parallel verification rejected"
            finding["isolation_duration_s"] = 0
            finding["isolation_command"] = ""
            finding["verdict"] = "INCONCLUSIVE"
        findings_json["summary"]["verified"] = {
            "INCONCLUSIVE": len(pending),
            "CONFIRMED": sum(f.get("verdict") == "CONFIRMED" for f in findings),
        }
        return findings_json
    print(f"Verifying {len(pending)} findings serially")

    # Multiple pytest subprocesses can still share one accelerator context and
    # memory pool, so verification remains serial until per-worker device pinning
    # exists.
    for index, finding in enumerate(pending, start=1):
        isolation_result = run_isolated_test(
            finding["representative_nodeid"], test_source_dir, timeout
        )
        apply_isolation(finding, isolation_result)
        print(
            f"  [{index}/{len(pending)}] {finding['class']} {finding['subject']}: "
            f"{isolation_result['status']} → {finding['verdict']}"
        )

    verdict_counts = {}
    for finding in findings:
        verdict = finding["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    findings_json["summary"]["verified"] = verdict_counts
    return findings_json


def version_key(path: Path) -> tuple[int, ...]:
    """Return a numeric key for a transformers-X.Y.Z source directory."""
    suffix = path.name.removeprefix("transformers-")
    return tuple(int(part) for part in suffix.split(".") if part.isdigit())


def resolve_test_source(root: Path, version: Optional[str]) -> Path:
    """Resolve either an exact source tree or a versioned cache root."""
    if (root / "tests" / "models").is_dir():
        return root
    if version:
        selected = root / f"transformers-{version}"
        if not selected.is_dir():
            raise FileNotFoundError(
                f"Transformers {version} source not found: {selected}"
            )
        return selected
    version_dirs = [p for p in root.glob("transformers-*") if p.is_dir()]
    if not version_dirs:
        raise FileNotFoundError(f"No transformers-X.Y.Z directory found in {root}")
    return max(version_dirs, key=version_key)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify transformers test findings in isolation"
    )
    parser.add_argument(
        "input", type=Path, help="Findings JSON from transformers_triage.py"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output verified findings JSON"
    )
    parser.add_argument(
        "--test-source-dir",
        type=Path,
        help="Exact Transformers source tree or its versioned cache root "
        "(default: the cache the official runner writes to, honouring "
        "HF_COVERAGE_CACHE)",
    )
    parser.add_argument(
        "--transformers-version",
        help="Select an exact transformers-X.Y.Z cache directory "
        "(default: the version recorded in the findings JSON)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Per-test timeout in seconds"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for isolated multi-device hosts; verification remains serial",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    with open(args.input) as file:
        findings_json = json.load(file)

    test_source_root = args.test_source_dir
    if test_source_root is None:
        # ``cache_root()`` rather than the module constant, so that the
        # documented HF_COVERAGE_CACHE override resolves here exactly as it does
        # in the runner that wrote the cache.
        test_source_root = Path(load_source_helper().cache_root()).expanduser()
    if not test_source_root.exists():
        raise FileNotFoundError(
            f"Test source directory not found: {test_source_root}\n"
            "Run transformers_hf_tests.py first to cache the official source, or "
            "pass --test-source-dir."
        )
    version = args.transformers_version or recorded_version(findings_json)
    if version is None:
        print(
            "Warning: the findings JSON records no transformers version; falling "
            "back to the newest cached source tree"
        )

    test_source_dir = resolve_test_source(test_source_root, version)
    print(f"Using test source: {test_source_dir}")

    result = verify_findings(findings_json, test_source_dir, args.timeout, args.workers)

    print("\nVerification summary:")
    for verdict, count in result["summary"].get("verified", {}).items():
        print(f"  {verdict}: {count}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as file:
        json.dump(result, file, indent=2)
    print(f"\nWriting {args.out}")

    # An isolation that selected no test measured nothing. That is a defect in
    # the harness, not a result, and reporting it as an ordinary run would let
    # "0 new findings" stand for "0 findings checked".
    verified = [f for f in result["findings"] if f.get("verification_required", True)]
    if verified and all(f.get("isolation_reported_tests") == 0 for f in verified):
        print(
            f"error: none of the {len(verified)} isolated runs selected a single "
            "test; nothing was measured, so no finding is confirmed",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
