#!/usr/bin/env python3
"""
Transformers Test Triage Tool

Automatically classify test failures from transformers_hf_tests.py JSON output.

Usage:
    python scripts/transformers/transformers_triage.py /tmp/qwen3.json --out /tmp/qwen3-findings.json

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

CPU_FALLBACK_CLASS = "OP_CPU_FALLBACK"


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

    # "X is not supported" or "unsupported device type X"
    match = re.search(r"'?(\w+)'? (?:is )?not supported", detail, re.I)
    if match:
        return match.group(1)

    match = re.search(
        r"unsupported ([\w ]+?)(?: type)? ['\"]?(\w+)['\"]?(?:\s|$)", detail, re.I
    )
    if match:
        return f"{match.group(1).strip()}_{match.group(2)}"

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


def detect_crash(test_record: Dict) -> Tuple[bool, str]:
    """
    Detect crash patterns using platform-agnostic signals.

    Returns: (is_crash, crash_type)
    """
    detail = test_record.get("detail", "")

    # A run-level poison marker invalidates later tests, but it does not prove
    # that every failed test caused the poison. Classify only per-test evidence.

    # 1. Segmentation fault (universal)
    if "segmentation fault" in detail.lower() or "sigsegv" in detail.lower():
        return True, "segfault"

    # 2. Core dump
    if "core dumped" in detail.lower():
        return True, "core_dump"

    # 3. Test timeout
    if test_record.get("timed_out"):
        return True, "timeout"

    # 4. Fatal Python error
    if "fatal python error" in detail.lower():
        return True, "fatal_python_error"

    # 5. Explicit device-side crash signatures. A normal RuntimeError is not a
    # crash: unsupported operators and feature gaps often use that exception.
    crash_patterns = [
        r"illegal memory access",
        r"device-side assert",
        r"unspecified launch failure",
        r"misaligned address",
        r"vmfault",
        r"acceleratorerror",
    ]
    for pattern in crash_patterns:
        if re.search(pattern, detail, re.I):
            return True, "device_runtime_crash"

    # 6. Process crash (no detail but failed)
    if test_record["status"] == "FAIL" and not detail.strip():
        return True, "empty_failure_likely_crash"

    return False, ""


def classify_failure(test_record: Dict) -> Tuple[str, str]:
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
    is_crash, crash_type = detect_crash(test_record)
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
        r"unsupported",
        r"requires.*not available",
        r"No module named",
    ]
    for pattern in feature_patterns:
        if re.search(pattern, detail, re.I):
            feature = extract_feature(detail)
            return "FEATURE_UNSUPPORTED", feature

    # Unknown
    return "UNKNOWN", "unclassified"


def compute_fingerprint(
    failure_class: str,
    component: str,
    subject: str,
    mechanism: str,
) -> str:
    """Compute a cause fingerprint without model or nodeid occurrence data."""
    payload = "|".join((failure_class, component, subject, mechanism))
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


def extract_component(test_json: Dict) -> str:
    """Identify the measured backend for cross-platform cause deduplication."""
    environment = test_json.get("environment", {})
    device = environment.get("device") or test_json.get("run", {}).get("device")
    return str(device or "unknown")


def fallback_findings(test_json: Dict, component: str) -> list[Dict]:
    """Turn measured CPU fallbacks into operator implementation findings."""
    occurrences: dict[str, list[dict]] = defaultdict(list)
    for test in test_json.get("tests", []):
        for op in test.get("cpu_fallback_ops", []):
            occurrences[op].append(test)

    findings = []
    for op, tests in sorted(occurrences.items()):
        mechanism = f"[flagos cpu_fallback] {op}"
        findings.append(
            {
                "fingerprint": compute_fingerprint(
                    CPU_FALLBACK_CLASS, component, op, mechanism
                ),
                "class": CPU_FALLBACK_CLASS,
                "component": component,
                "subject": op,
                "mechanism": mechanism,
                "nodeids": [test["nodeid"] for test in tests],
                "models": sorted(
                    {extract_model_from_nodeid(test["nodeid"]) for test in tests}
                ),
                "representative_nodeid": tests[0]["nodeid"],
                "representative_detail": (
                    f"{op} executed through torch_fl's CPU fallback while the "
                    "test otherwise continued."
                ),
                "count": len(tests),
                "verification_required": False,
                "verdict": "CONFIRMED",
            }
        )
    return findings


def triage_failures(test_json: Dict) -> Dict:
    """
    Triage all test failures and group by cause fingerprint.

    Returns findings dict with fingerprinted failures.
    """
    tests = test_json.get("tests", [])
    component = extract_component(test_json)

    # Collect failures
    failures = [t for t in tests if t["status"] == "FAIL"]

    # Group by fingerprint
    fingerprint_map: Dict[str, List[Dict]] = defaultdict(list)
    class_counts = defaultdict(int)

    for test in failures:
        failure_class, subject = classify_failure(test)
        class_counts[failure_class.lower().replace("_", "")] += 1

        mechanism = normalize_error(test.get("detail", ""))
        fingerprint = compute_fingerprint(failure_class, component, subject, mechanism)

        fingerprint_map[fingerprint].append(
            {
                "nodeid": test["nodeid"],
                "detail": test.get("detail", ""),
                "class": failure_class,
                "component": component,
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
                "component": rep["component"],
                "subject": rep["subject"],
                "mechanism": rep["mechanism"],
                "nodeids": nodeids,
                "models": models,
                "representative_nodeid": rep["nodeid"],
                "representative_detail": rep["detail"],
                "count": len(records),
            }
        )

    # CPU fallback is a correctness-success but an accelerator coverage failure.
    fallback_items = fallback_findings(test_json, component)
    findings.extend(fallback_items)
    if fallback_items:
        class_counts[CPU_FALLBACK_CLASS.lower().replace("_", "")] += len(fallback_items)

    # Sort by class priority: CRASH > unsupported/fallback > precision > feature.
    priority = {
        "CRASH": 0,
        "OP_UNSUPPORTED": 1,
        CPU_FALLBACK_CLASS: 1,
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
