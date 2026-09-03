#!/usr/bin/env python3
"""
Transformers Test Verification Tool

Isolate and rerun each finding's representative test in a fresh subprocess
to distinguish real failures from collateral damage due to device poisoning.

Usage:
    python scripts/transformers_verify.py /tmp/qwen3-findings.json --out /tmp/qwen3-verified.json

Features:
- Runs tests in parallel (default: CPU count)
- Fresh subprocess per test (no contamination)
- Timeout protection
- Records isolation outcome

Output schema adds to each finding:
    {
      "isolation_status": "FAIL",  # FAIL/PASS/SKIP/TIMEOUT/ERROR
      "isolation_detail": "...",
      "isolation_duration_s": 3.2,
      "isolation_command": "pytest ...",
      "verdict": "CONFIRMED"  # CONFIRMED/COLLATERAL
    }
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional


def run_isolated_test(
    nodeid: str,
    test_source_dir: Path,
    timeout: int = 120,
) -> Dict:
    """
    Run a single test in isolation.

    Args:
        nodeid: pytest nodeid (e.g., "tests/models/qwen3/test_modeling_qwen3.py::Qwen3ModelTest::test_save_load")
        test_source_dir: root directory containing the test
        timeout: test timeout in seconds

    Returns:
        {
            "status": "FAIL"|"PASS"|"SKIP"|"TIMEOUT"|"ERROR",
            "detail": "...",
            "duration_s": 3.2,
            "command": "pytest ...",
            "returncode": 0,
        }
    """
    # Build pytest command
    cmd = [
        "pytest",
        str(test_source_dir / nodeid),
        "-xvs",  # stop on first failure, verbose, no capture
        f"--timeout={timeout}",
        "--tb=short",  # short traceback
    ]

    command_str = " ".join(cmd)
    started = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
            + 10,  # subprocess timeout slightly longer than pytest timeout
            cwd=test_source_dir,
        )
        duration = time.time() - started

        # Parse pytest outcome from output
        combined = result.stdout + result.stderr

        if "PASSED" in combined or result.returncode == 0:
            status = "PASS"
        elif "SKIPPED" in combined:
            status = "SKIP"
        elif "FAILED" in combined or result.returncode != 0:
            status = "FAIL"
        else:
            status = "ERROR"

        return {
            "status": status,
            "detail": combined[-2000:],  # last 2000 chars
            "duration_s": round(duration, 1),
            "command": command_str,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - started
        return {
            "status": "TIMEOUT",
            "detail": f"Test exceeded {timeout}s timeout",
            "duration_s": round(duration, 1),
            "command": command_str,
            "returncode": -1,
        }

    except Exception as e:
        duration = time.time() - started
        return {
            "status": "ERROR",
            "detail": f"Exception running test: {e}",
            "duration_s": round(duration, 1),
            "command": command_str,
            "returncode": -1,
        }


def determine_verdict(isolation_status: str, original_class: str) -> str:
    """
    Determine if finding is CONFIRMED or COLLATERAL based on isolation outcome.

    Rules:
    - FAIL in isolation → CONFIRMED (real defect)
    - TIMEOUT in isolation → CONFIRMED (timeout is a defect)
    - PASS in isolation → COLLATERAL (suite failure but isolated pass = device poisoning side effect)
    - SKIP in isolation → COLLATERAL (not actually exercising the code path)
    - ERROR in isolation → CONFIRMED (assume real until proven otherwise)
    """
    if isolation_status in ("FAIL", "TIMEOUT", "ERROR"):
        return "CONFIRMED"
    elif isolation_status in ("PASS", "SKIP"):
        return "COLLATERAL"
    else:
        return "UNKNOWN"


def verify_findings(
    findings_json: Dict,
    test_source_dir: Path,
    timeout: int,
    max_workers: Optional[int],
) -> Dict:
    """
    Verify all findings by running isolated tests in parallel.

    Args:
        findings_json: output from transformers_triage.py
        test_source_dir: root directory containing transformers tests
        timeout: per-test timeout
        max_workers: parallelism (None = CPU count)

    Returns:
        findings_json with isolation results added
    """
    findings = findings_json["findings"]
    print(f"Verifying {len(findings)} findings with {max_workers or 'auto'} workers")

    # Prepare tasks
    tasks = []
    for i, finding in enumerate(findings):
        nodeid = finding["representative_nodeid"]
        tasks.append((i, nodeid, finding))

    # Run in parallel
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_isolated_test,
                nodeid,
                test_source_dir,
                timeout,
            ): (i, finding)
            for i, nodeid, finding in tasks
        }

        for future in as_completed(futures):
            i, finding = futures[future]
            try:
                isolation_result = future.result()
                results[i] = isolation_result

                # Print progress
                verdict = determine_verdict(
                    isolation_result["status"],
                    finding["class"],
                )
                print(
                    f"  [{i + 1}/{len(findings)}] {finding['class']} {finding['subject']}: "
                    f"{isolation_result['status']} → {verdict}"
                )

            except Exception as e:
                print(f"  [{i + 1}/{len(findings)}] ERROR: {e}")
                results[i] = {
                    "status": "ERROR",
                    "detail": str(e),
                    "duration_s": 0,
                    "command": "",
                    "returncode": -1,
                }

    # Merge results back into findings
    for finding, isolation_result in zip(findings, results):
        finding["isolation_status"] = isolation_result["status"]
        finding["isolation_detail"] = isolation_result["detail"]
        finding["isolation_duration_s"] = isolation_result["duration_s"]
        finding["isolation_command"] = isolation_result["command"]
        finding["verdict"] = determine_verdict(
            isolation_result["status"],
            finding["class"],
        )

    # Update summary
    verdict_counts = {}
    for finding in findings:
        verdict = finding["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    findings_json["summary"]["verified"] = verdict_counts

    return findings_json


def main():
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
        help="Root directory containing transformers test sources (default: HF cache)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-test timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    # Find the test source directory
    # transformers_hf_tests.py caches sources under /root/.cache/torch_fl/hf-tests/transformers-X.Y.Z/
    # We need to find the specific version directory
    if not args.test_source_dir.exists():
        raise FileNotFoundError(
            f"Test source directory not found: {args.test_source_dir}\n"
            f"Make sure transformers_hf_tests.py has been run and cached the test sources."
        )

    # Find the transformers version directory
    version_dirs = list(args.test_source_dir.glob("transformers-*"))
    if not version_dirs:
        raise FileNotFoundError(
            f"No transformers-X.Y.Z directory found in {args.test_source_dir}"
        )

    # Use the most recent (highest version)
    test_source_dir = sorted(version_dirs)[-1]
    print(f"Using test source: {test_source_dir}")

    print(f"Reading {args.input}")
    with open(args.input) as f:
        findings_json = json.load(f)

    print(f"\nVerifying {len(findings_json['findings'])} findings...")
    result = verify_findings(
        findings_json,
        test_source_dir,
        args.timeout,
        args.workers,
    )

    print("\nVerification summary:")
    for verdict, count in result["summary"].get("verified", {}).items():
        print(f"  {verdict}: {count}")

    confirmed = [f for f in result["findings"] if f["verdict"] == "CONFIRMED"]
    collateral = [f for f in result["findings"] if f["verdict"] == "COLLATERAL"]
    print(f"\nConfirmed: {len(confirmed)}")
    print(f"Collateral: {len(collateral)}")

    print(f"\nWriting {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
