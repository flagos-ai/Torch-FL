#!/usr/bin/env python3
"""Verify Transformers findings in fresh pytest subprocesses."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_SPEC = REPO_ROOT / "tests" / "manual" / "hf_device_spec.py"


def isolated_env(test_source_dir: Path, workdir: Path) -> dict[str, str]:
    """Build the same PrivateUse1 test environment as the official runner."""
    env = dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["FLAGOS_LOG_FALLBACK"] = "1"
    env.pop("TRANSFORMERS_TEST_DEVICE", None)
    env["TRANSFORMERS_TEST_DEVICE_SPEC"] = "hf_device_spec.py"
    entries = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != REPO_ROOT
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(workdir), str(test_source_dir), str(test_source_dir / "utils"), *entries]
    )
    return env


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
    if result.returncode == 0:
        status = "SKIP" if " skipped" in combined.lower() else "PASS"
    elif result.returncode == 1:
        status = "FAIL"
    else:
        status = "ERROR"

    return {
        "status": status,
        "detail": combined[-8000:],
        "duration_s": round(duration, 1),
        "command": command_str,
        "returncode": result.returncode,
    }


def determine_verdict(isolation_status: str, original_class: str) -> str:
    """Map an isolation outcome to a filing verdict."""
    del original_class
    if isolation_status in ("FAIL", "TIMEOUT"):
        return "CONFIRMED"
    if isolation_status in ("PASS", "SKIP"):
        return "COLLATERAL"
    return "INCONCLUSIVE"


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
        finding["isolation_status"] = isolation_result["status"]
        finding["isolation_detail"] = isolation_result["detail"]
        finding["isolation_duration_s"] = isolation_result["duration_s"]
        finding["isolation_command"] = isolation_result["command"]
        finding["verdict"] = determine_verdict(
            isolation_result["status"], finding["class"]
        )
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
        default=Path("/root/.cache/torch_fl/hf-tests"),
        help="Exact Transformers source tree or its versioned cache root",
    )
    parser.add_argument(
        "--transformers-version",
        help="Select an exact transformers-X.Y.Z cache directory",
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
    if not args.test_source_dir.exists():
        raise FileNotFoundError(
            f"Test source directory not found: {args.test_source_dir}\n"
            "Run transformers_hf_tests.py first to cache the official source."
        )

    test_source_dir = resolve_test_source(
        args.test_source_dir, args.transformers_version
    )
    print(f"Using test source: {test_source_dir}")

    with open(args.input) as file:
        findings_json = json.load(file)
    result = verify_findings(findings_json, test_source_dir, args.timeout, args.workers)

    print("\nVerification summary:")
    for verdict, count in result["summary"].get("verified", {}).items():
        print(f"  {verdict}: {count}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as file:
        json.dump(result, file, indent=2)
    print(f"\nWriting {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
