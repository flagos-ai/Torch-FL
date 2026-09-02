#!/usr/bin/env python3
"""
Transformers Test Triage Tool

Automatically classify test failures from transformers_hf_tests.py JSON output.

Usage:
    python scripts/transformers_triage.py /tmp/qwen3.json --out /tmp/qwen3-findings.json

Output schema:
    {
      "findings": [
        {
          "fingerprint": "a1b2c3d4e5f6",
          "class": "OP_UNSUPPORTED",
          "subject": "aten::index_copy_.out",
          "mechanism": "NotImplementedError: backend not registered",
          "nodeids": ["test_...::test_save_load", ...],
          "models": ["qwen3"],
          "representative_nodeid": "test_...::test_save_load",
          "representative_detail": "full error text",
          "count": 1
        }
      ],
      "summary": {
        "total_failures": 20,
        "op_unsupported": 5,
        "precision": 3,
        "crash": 1,
        "feature_unsupported": 2,
        "precision_known_issue": 8,
        "unknown": 1
      }
    }
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def extract_op_name(detail: str) -> str:
    """Extract aten operator name from error message."""
    # Pattern: "could not run 'aten::add.Tensor'"
    match = re.search(r"'(aten::[^']+)'", detail)
    if match:
        return match.group(1)

    # Pattern: NotImplementedError in traceback with op name
    match = re.search(r"aten::(\w+(?:\.\w+)?)", detail)
    if match:
        return f"aten::{match.group(1)}"

    return "unknown_op"


def extract_feature(detail: str) -> str:
    """Extract feature name from error message."""
    # AttributeError: 'Foo' object has no attribute 'bar'
    match = re.search(r"no attribute '(\w+)'", detail)
    if match:
        return match.group(1)

    # "X is not supported"
    match = re.search(r"'?(\w+)'? (?:is )?not supported", detail, re.I)
    if match:
        return match.group(1)

    return "unknown_feature"


def normalize_error(text: str) -> str:
    """Normalize error text for fingerprinting."""
    # Remove addresses
    t = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)

    # Remove temp paths
    t = re.sub(r"/tmp/[^\s\'\"]+", "/tmp/PATH", t)

    # Normalize site-packages paths
    t = re.sub(r"(/[^\s\'\"]*)?/(site-packages|torch_fl|tests)/", r"/PATH/\2/", t)

    # Remove timing info
    t = re.sub(r"\b\d+\.\d+s\b", "TIMEs", t)

    # Collapse tensor shapes
    t = re.sub(r"\[[\d,\s]+\]", "[SHAPE]", t)

    # Keep diagnostic codes, collapse other numbers
    t = re.sub(r"(?<!err )(?<!code )(?<!errno )\b\d+\b", "N", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Take last 200 chars (most specific part)
    return t[-200:]


def detect_crash(test_record: Dict, run_info: Dict) -> Tuple[bool, str]:
    """
    Detect crash patterns using platform-agnostic signals.

    Returns: (is_crash, crash_type)
    """
    detail = test_record.get("detail", "")

    # 1. Run-level device poisoning
    if run_info.get("context_poison"):
        return True, "device_context_poisoned"

    # 2. Segmentation fault (universal)
    if "segmentation fault" in detail.lower() or "sigsegv" in detail.lower():
        return True, "segfault"

    # 3. Core dump
    if "core dumped" in detail.lower():
        return True, "core_dump"

    # 4. Test timeout
    if test_record.get("timed_out"):
        return True, "timeout"

    # 5. Fatal Python error
    if "fatal python error" in detail.lower():
        return True, "fatal_python_error"

    # 6. Generic runtime errors (platform-agnostic patterns)
    runtime_patterns = [
        r"runtime error",
        r"device error",
        r"kernel.*(?:error|failed)",
        r"launch.*(?:error|failed)",
        r"memory.*(?:error|access)",
        r"invalid.*(?:device|kernel)",
    ]
    for pattern in runtime_patterns:
        if re.search(pattern, detail, re.I):
            return True, "runtime_error"

    # 7. Process crash (no detail but failed)
    if test_record["status"] == "FAIL" and not detail.strip():
        return True, "empty_failure_likely_crash"

    return False, ""


def classify_failure(test_record: Dict, run_info: Dict) -> Tuple[str, str]:
    """
    Classify a test failure.

    Returns: (failure_class, subject)

    Classes:
    - OP_UNSUPPORTED: missing operator
    - PRECISION: numerical mismatch
    - CRASH: segfault, timeout, device poisoning
    - FEATURE_UNSUPPORTED: missing feature/API
    - PRECISION_KNOWN_ISSUE: SDPA tolerance (not filed)
    - UNKNOWN: unclassified
    """
    detail = test_record.get("detail", "")
    nodeid = test_record["nodeid"]

    # Check crash first
    is_crash, crash_type = detect_crash(test_record, run_info)
    if is_crash:
        return "CRASH", crash_type

    # OP_UNSUPPORTED patterns
    op_patterns = [
        r"NotImplementedError",
        r"backend not registered",
        r"could not run 'aten::",
        r"No kernel found for",
        r"operator.*not implemented",
    ]
    for pattern in op_patterns:
        if re.search(pattern, detail, re.I):
            op_name = extract_op_name(detail)
            return "OP_UNSUPPORTED", op_name

    # PRECISION patterns
    if "AssertionError" in detail:
        # Check for numerical comparison
        if re.search(r"\d+\.?\d*\s*[><]=?\s*\d+\.?\d*", detail):
            # Exclude known SDPA tolerance issues
            if "eager_matches_sdpa" in nodeid or "sdpa_inference" in nodeid:
                return "PRECISION_KNOWN_ISSUE", "sdpa_tolerance_unrecognized_device"
            return "PRECISION", "numerical_mismatch"

    # FEATURE_UNSUPPORTED patterns
    feature_patterns = [
        r"AttributeError",
        r"not supported",
        r"requires.*not available",
        r"No module named",
    ]
    for pattern in feature_patterns:
        if re.search(pattern, detail, re.I):
            feature = extract_feature(detail)
            return "FEATURE_UNSUPPORTED", feature

    # Unknown
    return "UNKNOWN", "unclassified"


def compute_fingerprint(failure_class: str, subject: str, mechanism: str) -> str:
    """
    Compute cause fingerprint for deduplication.

    Does NOT include model name or nodeid, so same issue across models merges.
    """
    payload = f"{failure_class}|{subject}|{mechanism}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def extract_model_from_nodeid(nodeid: str) -> str:
    """Extract model name from test nodeid."""
    # Pattern: tests/models/qwen3/test_modeling_qwen3.py::...
    match = re.search(r"tests/models/(\w+)/", nodeid)
    if match:
        return match.group(1)

    # Pattern: test_modeling_qwen3.py
    match = re.search(r"test_modeling_(\w+)\.py", nodeid)
    if match:
        return match.group(1)

    return "unknown"


def triage_failures(test_json: Dict) -> Dict:
    """
    Triage all test failures and group by cause fingerprint.

    Returns findings dict with fingerprinted failures.
    """
    run_info = test_json.get("run", {})
    tests = test_json.get("tests", [])

    # Collect failures
    failures = [t for t in tests if t["status"] == "FAIL"]

    # Group by fingerprint
    fingerprint_map: Dict[str, List[Dict]] = defaultdict(list)
    class_counts = defaultdict(int)

    for test in failures:
        failure_class, subject = classify_failure(test, run_info)
        class_counts[failure_class.lower().replace("_", "")] += 1

        mechanism = normalize_error(test.get("detail", ""))
        fingerprint = compute_fingerprint(failure_class, subject, mechanism)

        fingerprint_map[fingerprint].append(
            {
                "nodeid": test["nodeid"],
                "detail": test.get("detail", ""),
                "class": failure_class,
                "subject": subject,
                "mechanism": mechanism,
                "model": extract_model_from_nodeid(test["nodeid"]),
            }
        )

    # Build findings list
    findings = []
    for fingerprint, records in fingerprint_map.items():
        # Pick representative (first occurrence)
        rep = records[0]

        # Aggregate models and nodeids
        models = sorted(set(r["model"] for r in records))
        nodeids = [r["nodeid"] for r in records]

        findings.append(
            {
                "fingerprint": fingerprint,
                "class": rep["class"],
                "subject": rep["subject"],
                "mechanism": rep["mechanism"],
                "nodeids": nodeids,
                "models": models,
                "representative_nodeid": rep["nodeid"],
                "representative_detail": rep["detail"],
                "count": len(records),
            }
        )

    # Sort by class priority: CRASH > OP_UNSUPPORTED > PRECISION > FEATURE > UNKNOWN
    priority = {
        "CRASH": 0,
        "OP_UNSUPPORTED": 1,
        "PRECISION": 2,
        "FEATURE_UNSUPPORTED": 3,
        "PRECISION_KNOWN_ISSUE": 4,
        "UNKNOWN": 5,
    }
    findings.sort(key=lambda f: (priority.get(f["class"], 99), f["subject"]))

    return {
        "findings": findings,
        "summary": {
            "total_failures": len(failures),
            **dict(class_counts),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Triage transformers test failures")
    parser.add_argument(
        "input", type=Path, help="JSON output from transformers_hf_tests.py"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output findings JSON")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    print(f"Reading {args.input}")
    with open(args.input) as f:
        test_json = json.load(f)

    print("Triaging failures...")
    result = triage_failures(test_json)

    print("\nSummary:")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")

    print(f"\nFindings: {len(result['findings'])} unique causes")
    for finding in result["findings"]:
        print(
            f"  [{finding['class']}] {finding['subject']} "
            f"({finding['count']} occurrences across {len(finding['models'])} models)"
        )

    print(f"\nWriting {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
