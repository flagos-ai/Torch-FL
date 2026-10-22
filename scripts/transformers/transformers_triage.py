#!/usr/bin/env python3
"""
Transformers Test Triage Tool

Automatically classify test failures from transformers_hf_tests.py JSON output.

Usage:
    python scripts/transformers/transformers_triage.py /tmp/qwen3.json --out /tmp/qwen3-findings.json

Both runner output shapes are accepted: one model at the top level, and the
``--all`` aggregate, which keeps each model in ``models[]``.

Output schema:
    {
      "findings": [
        {
          "fingerprint": "a1b2c3d4e5f6",
          "class": "OP_UNSUPPORTED",
          "component": "flagos",
          "subject": "aten::index_copy_.out",
          "mechanism": "NotImplementedError: Could not run 'aten::index_copy_.out' ...",
          "nodeids": ["tests/models/qwen3/...::test_save_load", ...],
          "models": ["qwen3"],
          "representative_nodeid": "tests/models/qwen3/...::test_save_load",
          "representative_detail": "full error text",
          "count": 1,
          "actionable": true,
          "verification_required": true
        }
      ],
      "summary": {
        "total_failures": 20,
        "actionable": 8,
        "environment_error": 3,
        "statuses": {"FAIL": 17, "ENVIRONMENT_ERROR": 3},
        "classes": {"CRASH": 1, "OP_UNSUPPORTED": 5, "ENVIRONMENT_ERROR": 3}
      }
    }
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

CPU_FALLBACK_CLASS = "OP_CPU_FALLBACK"
TEST_ERROR_CLASS = "TEST_ERROR"
ENVIRONMENT_ERROR_CLASS = "ENVIRONMENT_ERROR"

# Statuses the runner writes for tests that are not a pass. ``XPASS`` is
# deliberately absent: an unexpected pass is a pass, and counting it as a defect
# would turn the suite's own expectations into platform findings.
DEFECT_STATUSES = ("FAIL", "ERROR")
ENVIRONMENT_STATUSES = ("ENVIRONMENT_ERROR",)
COLLECTION_STATUSES = ("COLLECT_ERROR", "BATCH_CRASHED")
FAILING_STATUSES = DEFECT_STATUSES + ENVIRONMENT_STATUSES + COLLECTION_STATUSES

# Classes that never claim a platform defect. They stay in the report --- an
# environment that never reached the accelerator has to be visible --- but no
# isolated re-run can turn them into a backend problem, so the filing path must
# skip them. ``PRECISION_KNOWN_ISSUE`` is on this list because the classifier
# has always documented it as not filed.
NON_ACTIONABLE_CLASSES = (
    TEST_ERROR_CLASS,
    ENVIRONMENT_ERROR_CLASS,
    "PRECISION_KNOWN_ISSUE",
)


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
    """Replace everything that varies between two runs of one failure.

    Every substitution is idempotent, and each writes a token that its own
    pattern cannot match again. That is the whole point: the previous spelling
    replaced addresses with ``0xADDR``, and since ``0xADD`` is itself valid hex,
    re-normalizing the result produced ``0xADDRR`` and then ``0xADDRRR``, so the
    same failure fingerprinted differently at every stage of the pipeline.
    """
    # Addresses. The legacy ``0xADDR`` token is folded in first: text that was
    # normalized by an older version of this function still circulates in saved
    # results and cached baselines.
    t = text.replace("0xADDR", "<ADDR>")
    t = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", t)

    # Temp paths and library paths become one token, without the matched words,
    # so the pattern cannot match a second time.
    t = re.sub(r"/tmp/[^\s'\"]+", "<TMP>", t)
    t = re.sub(r"(?:/[^\s'\"]*)?/(?:site-packages|torch_fl|tests)/", "<PATH>/", t)

    # Timing info
    t = re.sub(r"\b\d+\.\d+s\b", "<TIME>", t)

    # Collapse tensor shapes
    t = re.sub(r"\[[\d,\s]+\]", "<SHAPE>", t)

    # Keep diagnostic codes, collapse other numbers
    t = re.sub(r"(?<!err )(?<!code )(?<!errno )\b\d+\b", "N", t)

    # Normalize whitespace
    return re.sub(r"\s+", " ", t).strip()


def shorten(text: str, limit: int = 200) -> str:
    """Bound a statement without cutting a word in half.

    Both ends are kept, as the runner does for its tracebacks: the exception
    type sits at the front and the most specific detail at the back.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    marker = " ... "
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    front = text[:head].rsplit(" ", 1)[0]
    back = text[-tail:].split(" ", 1)[-1]
    return f"{front}{marker}{back}"


def exception_line(detail: str) -> str:
    """The line of a traceback that states the failure itself.

    pytest's short format ends with the exception, which is the most specific
    statement the output contains. A caret marker (``~~~^~~~``) and pytest's
    bare ``E`` continuation carry no information and are skipped.
    """
    for line in reversed((detail or "").splitlines()):
        stripped = re.sub(r"^E\s+", "", line.strip()).strip()
        if not stripped or set(stripped) <= set("~^"):
            continue
        return stripped
    return ""


def mechanism_from(detail: str) -> str:
    """The shortest statement that still identifies one failure.

    A fixed window of the raw traceback was neither stable nor readable: it cut
    mid-word, and anything inserted above the exception shifted the window and
    therefore the fingerprint of an unchanged defect.
    """
    line = exception_line(detail)
    if not line:
        return "no output captured"
    return shorten(normalize_error(line))


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
    if test_record.get("status") == "FAIL" and not detail.strip():
        return True, "empty_failure_likely_crash"

    return False, ""


def missing_module(detail: str) -> str | None:
    """Name the import a broken environment failed on, when it says so."""
    match = re.search(r"No module named '?([\w.]+)'?", detail)
    return match.group(1) if match else None


def classify_failure(test_record: Dict) -> Tuple[str, str]:
    """
    Classify a test record that did not pass.

    Returns: (failure_class, subject)

    Classes:
    - OP_UNSUPPORTED: missing operator
    - OP_CPU_FALLBACK: operator ran through the host fallback
    - PRECISION: numerical mismatch
    - CRASH: segfault, timeout, device poisoning
    - FEATURE_UNSUPPORTED: missing feature/API
    - PRECISION_KNOWN_ISSUE: SDPA tolerance (not filed)
    - TEST_ERROR: the test body never ran, or never reported
    - ENVIRONMENT_ERROR: the environment, not the platform, failed
    - UNKNOWN: unclassified

    The status decides first. A setup or teardown error, a collection failure
    and a lost batch all mean the assertion never executed, so running the
    platform-defect patterns over their output would invent a finding from text
    that no test produced.
    """
    detail = test_record.get("detail") or ""
    nodeid = test_record.get("nodeid") or ""
    status = test_record.get("status") or "FAIL"

    if status in ENVIRONMENT_STATUSES:
        return ENVIRONMENT_ERROR_CLASS, missing_module(detail) or "environment"
    if status == "ERROR":
        return TEST_ERROR_CLASS, "setup_or_teardown_error"
    if status in COLLECTION_STATUSES:
        return TEST_ERROR_CLASS, status.lower()

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


def generate_fingerprint(
    failure_class: str,
    component: str,
    subject: str,
    mechanism: str,
) -> str:
    """Hash one cause, without any model or nodeid occurrence data.

    This is the only fingerprint the pipeline files against. Two tests that fail
    for the same reason on the same component share it, so a defect that breaks
    400 tests is one finding; the same defect on a different backend does not,
    so a fix verified on one platform is not assumed to hold on another.
    """
    payload = "|".join((failure_class, component, subject, mechanism))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# The old name, kept because the verifier and the unit tests import it.
compute_fingerprint = generate_fingerprint


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


def component_of(block: Dict, fallback: Dict) -> str:
    """The backend a model's tests were measured on.

    Read from the environment the runner recorded, and never from a value a
    caller passed in: the runner derives ``device`` from the device spec, so
    this is the one spelling guaranteed to name the accelerator the tests
    actually used. A device string supplied on the command line could disagree
    with the spec and silently mis-attribute every fingerprint computed from it.
    """
    environment = block.get("environment") or fallback or {}
    return str(environment.get("device") or "unknown")


def run_units(test_json: Dict) -> Iterator[Dict]:
    """Yield each measured model as a self-contained unit.

    The runner writes two shapes. One model is the whole document; ``--all``
    keeps every model under ``models[]`` with no top-level ``tests`` key.
    Reading only the flat shape is why a full sweep used to triage to zero
    findings, so both are folded into the same unit here.
    """
    fallback_environment = test_json.get("environment") or {}
    blocks = test_json.get("models") if test_json.get("mode") == "all" else None
    for block in blocks if blocks is not None else [test_json]:
        model = (block.get("model") or {}).get("requested")
        yield {
            "tests": block.get("tests") or [],
            "collect_errors": block.get("collect_errors") or [],
            "component": component_of(block, fallback_environment),
            "model": model,
        }


def failing_records(unit: Dict) -> Iterator[Dict]:
    """Yield every record that did not pass, including collection failures.

    Only ``FAIL`` used to be read, so a run in which every test errored out on a
    missing hook was summarized as "all tests passed".
    """
    for record in unit["tests"]:
        if record.get("status") in FAILING_STATUSES:
            yield record
    for record in unit["collect_errors"]:
        if record.get("status") in FAILING_STATUSES:
            yield record


def models_of(nodeids: List[str], model: str | None) -> List[str]:
    """The architectures a finding was observed in."""
    if model:
        return [model]
    return sorted({extract_model_from_nodeid(nodeid) for nodeid in nodeids})


def fallback_findings(unit: Dict) -> list[Dict]:
    """Turn measured CPU fallbacks into operator implementation findings."""
    occurrences: dict[str, list[dict]] = defaultdict(list)
    for test in unit["tests"]:
        for op in test.get("cpu_fallback_ops") or []:
            occurrences[op].append(test)

    findings = []
    for op, tests in sorted(occurrences.items()):
        mechanism = f"[flagos cpu_fallback] {op}"
        nodeids = [test["nodeid"] for test in tests]
        component = unit["component"]
        findings.append(
            {
                "fingerprint": generate_fingerprint(
                    CPU_FALLBACK_CLASS, component, op, mechanism
                ),
                "class": CPU_FALLBACK_CLASS,
                "component": component,
                "subject": op,
                "mechanism": mechanism,
                "nodeids": nodeids,
                "models": models_of(nodeids, unit["model"]),
                "representative_nodeid": tests[0]["nodeid"],
                "representative_detail": (
                    f"{op} executed through torch_fl's CPU fallback while the "
                    "test otherwise continued."
                ),
                "count": len(tests),
                "actionable": True,
                "verification_required": False,
                "verdict": "CONFIRMED",
            }
        )
    return findings


# Most severe first. TEST_ERROR and ENVIRONMENT_ERROR sit last: they are
# reported so a broken environment stays visible, but they claim no defect.
PRIORITY = {
    "CRASH": 0,
    "OP_UNSUPPORTED": 1,
    CPU_FALLBACK_CLASS: 1,
    "PRECISION": 2,
    "FEATURE_UNSUPPORTED": 3,
    "PRECISION_KNOWN_ISSUE": 4,
    TEST_ERROR_CLASS: 5,
    ENVIRONMENT_ERROR_CLASS: 5,
    "UNKNOWN": 6,
}


def measured_environment(test_json: Dict) -> Dict:
    """The environment the measurement ran in.

    Triage used to return findings and a summary and drop everything else, so
    the verifier could not learn which ``transformers`` version produced them
    and fell back to the newest cached source tree --- a 5.12.1 run was verified
    against 5.14.1. The run's own environment is carried through instead. In
    ``--all`` mode the environment lives on each model block; one interpreter
    measured them all, so the first block that has one answers for the run.
    """
    if test_json.get("environment"):
        return test_json["environment"]
    for block in test_json.get("models") or []:
        if block.get("environment"):
            return block["environment"]
    return {}


def triage_failures(test_json: Dict) -> Dict:
    """
    Triage every non-passing test across every measured model.

    Returns findings grouped by cause fingerprint.
    """
    fingerprint_map: Dict[str, List[Dict]] = defaultdict(list)
    fallbacks: list[Dict] = []
    class_counts: Dict[str, int] = defaultdict(int)
    status_counts: Dict[str, int] = defaultdict(int)
    total = 0

    for unit in run_units(test_json):
        for record in failing_records(unit):
            total += 1
            status_counts[record.get("status") or "FAIL"] += 1
            failure_class, subject = classify_failure(record)
            cause = {
                "nodeid": record.get("nodeid") or "<unknown>",
                "detail": record.get("detail") or "",
                "class": failure_class,
                "component": unit["component"],
                "subject": subject,
                "mechanism": mechanism_from(record.get("detail") or ""),
                "model": unit["model"]
                or extract_model_from_nodeid(record.get("nodeid") or ""),
            }
            fingerprint = generate_fingerprint(
                failure_class, unit["component"], subject, cause["mechanism"]
            )
            fingerprint_map[fingerprint].append(cause)

        # CPU fallback is a correctness success but an accelerator coverage
        # failure, so it is a finding even in a run where nothing failed.
        fallbacks.extend(fallback_findings(unit))

    findings = []
    for fingerprint, records in fingerprint_map.items():
        rep = records[0]
        nodeids = [record["nodeid"] for record in records]
        findings.append(
            {
                "fingerprint": fingerprint,
                "class": rep["class"],
                "component": rep["component"],
                "subject": rep["subject"],
                "mechanism": rep["mechanism"],
                "nodeids": nodeids,
                "models": models_of(nodeids, rep["model"]),
                "representative_nodeid": rep["nodeid"],
                "representative_detail": rep["detail"],
                "count": len(records),
                "actionable": rep["class"] not in NON_ACTIONABLE_CLASSES,
                "verification_required": rep["class"] not in NON_ACTIONABLE_CLASSES,
                "verdict": (
                    "INCONCLUSIVE" if rep["class"] in NON_ACTIONABLE_CLASSES else None
                ),
            }
        )
    findings.extend(fallbacks)

    for finding in findings:
        class_counts[finding["class"]] += 1

    findings.sort(key=lambda f: (PRIORITY.get(f["class"], 99), f["subject"]))

    return {
        "findings": findings,
        "environment": measured_environment(test_json),
        "summary": {
            "total_failures": total,
            "actionable": sum(1 for f in findings if f["actionable"]),
            "environment_error": sum(
                1 for f in findings if f["class"] == ENVIRONMENT_ERROR_CLASS
            ),
            "statuses": dict(sorted(status_counts.items())),
            "classes": dict(sorted(class_counts.items())),
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
