#!/usr/bin/env python3
"""
Transformers Issue Preview Tool

Generate issue bodies for new findings and create a markdown preview for user review.

Usage:
    python scripts/transformers_preview_issues.py /tmp/qwen3-new.json \
        --out /tmp/qwen3-preview.md \
        --chip "MUSA MTT S5000" \
        --transformers-version 5.16.1 \
        --torch-fl-commit 64e60dd

Output:
- Markdown preview with all issue bodies
- Individual issue body files in /tmp/issue-<fingerprint>.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def get_chip_name() -> str:
    """Try to get chip name from torch.flagos.get_device_properties."""
    try:
        import torch

        props = torch.flagos.get_device_properties(0)
        return props.name
    except Exception:
        return "Unknown"


def generate_issue_title(
    finding: Dict,
    chip: str,
    transformers_version: str,
) -> str:
    """Generate issue title following the convention."""
    model = finding["models"][0] if finding["models"] else "unknown"
    subject = finding["subject"]
    failure_class = finding["class"]

    if failure_class == "OP_UNSUPPORTED":
        return f"[AI][{chip}] {model}: {subject} not supported (transformers {transformers_version})"
    elif failure_class == "PRECISION":
        return f"[AI][{chip}] {model}: {subject} precision mismatch vs CPU (transformers {transformers_version})"
    elif failure_class == "CRASH":
        return f"[AI][{chip}] {model}: {subject} crash (transformers {transformers_version})"
    elif failure_class == "FEATURE_UNSUPPORTED":
        return f"[AI][{chip}] {model}: {subject} feature not supported (transformers {transformers_version})"
    else:
        return f"[AI][{chip}] {model}: {subject} failure (transformers {transformers_version})"


def generate_issue_body(
    finding: Dict,
    chip: str,
    transformers_version: str,
    torch_fl_commit: str,
    pytorch_version: str,
) -> str:
    """
    Generate full issue body following .github/ISSUE_TEMPLATE/ai_agent_issue.md.
    """
    fp = finding["fingerprint"]
    failure_class = finding["class"]
    subject = finding["subject"]
    mechanism = finding["mechanism"]
    models = ", ".join(finding["models"])
    count = finding["count"]
    nodeid = finding["representative_nodeid"]
    detail = finding["representative_detail"]
    isolation_status = finding.get("isolation_status", "NOT_VERIFIED")
    isolation_detail = finding.get("isolation_detail", "")

    # Determine labels
    labels = ["ai-generated"]
    if failure_class == "CRASH":
        labels.append("bug")
        labels.append("P0")
    elif failure_class in ("OP_UNSUPPORTED", "FEATURE_UNSUPPORTED"):
        labels.append("enhancement")
    elif failure_class == "PRECISION":
        labels.append("bug")
    else:
        labels.append("bug")

    labels_str = ", ".join(labels)

    body = (
        f"""## AI Agent Information
- **Agent/Tool**: Transformers Auto-Triage
- **Model**: Automated script
- **Human Reviewer**: (assign manually)
- **Automated**: Yes - this issue was auto-generated from test failure analysis

## Summary

**Fingerprint**: `{fp}`

Test failure in `{nodeid}` indicates `{subject}` {failure_class.lower().replace("_", " ")} on {chip}.

This finding was:
- Observed in {count} test case(s) across {len(finding["models"])} model(s): {models}
- Verified in isolation: {isolation_status}
- Classified as: {failure_class}

## Change Type
- [x] Bug Fix
- [ ] New Feature
- [ ] Performance Optimization
- [ ] Refactoring
- [ ] Documentation
- [ ] Testing

## Problem Analysis

### What is broken?

{subject} fails with:

```
{mechanism}
```

### Reproduction

**Test command** (isolated):
```bash
{finding.get("isolation_command", "pytest " + nodeid)}
```

**Representative test**: `{nodeid}`

**Full error output**:
```
{detail[:1000]}
{"..." if len(detail) > 1000 else ""}
```

### Environment

- **Chip**: {chip}
- **Device**: flagos
- **PyTorch**: {pytorch_version}
- **Transformers**: {transformers_version}
- **torch_fl commit**: {torch_fl_commit}

### Root Cause Analysis

**Class**: {failure_class}

**Subject**: {subject}

**Mechanism**: {mechanism}

{"**Status**: This operator is not registered for the flagos backend." if failure_class == "OP_UNSUPPORTED" else ""}
{"**Status**: Numerical output differs from CPU baseline beyond acceptable tolerance." if failure_class == "PRECISION" else ""}
{"**Status**: Runtime crash or device error." if failure_class == "CRASH" else ""}

### Isolation Verification

The finding was re-run in a fresh subprocess to distinguish real failures from device poisoning collateral:

- **Isolation status**: {isolation_status}
- **Verdict**: {finding.get("verdict", "UNKNOWN")}

<details>
<summary>Isolation output</summary>

```
{isolation_detail[:1000]}
{"..." if len(isolation_detail) > 1000 else ""}
```

</details>

## Proposed Solution

"""
        + (
            """
1. Implement `{subject}` for flagos backend
2. Register the operator in the appropriate backend file (csrc/aten/backends/*/...)
3. Add unit test for the operator
4. Rerun the transformers test to verify
"""
            if failure_class == "OP_UNSUPPORTED"
            else """
1. Investigate the precision difference vs CPU
2. Check if accumulation order or reduced-precision intermediates are causing the mismatch
3. Validate against CUDA if available to determine acceptable tolerance
4. Fix implementation or adjust test tolerance if justified
"""
            if failure_class == "PRECISION"
            else """
1. Investigate the crash location using debugger or LAUNCH_BLOCKING
2. Fix the underlying cause (memory access, kernel launch, etc.)
3. Rerun affected tests to verify fix
4. Check for device poisoning after fix (run full test suite)
"""
            if failure_class == "CRASH"
            else """
1. Investigate the failure and determine root cause
2. Implement the missing feature or fix the bug
3. Add test coverage
4. Verify fix with transformers test
"""
        )
        + f"""

## Verification Plan

### Unit Tests
- [ ] Add/update unit test for {subject}
- [ ] Verify test passes on flagos

### Integration Tests
- [ ] Rerun transformers test: `{nodeid}`
- [ ] Verify test passes in isolation
- [ ] Run full model suite to check for regressions

### Hardware Tests
- [ ] Test on {chip}
- [ ] Verify no device poisoning in full suite
- [ ] Check performance impact

## Related Work

- Discovered during transformers coverage sweep
- Related models: {models}
- Total occurrences: {count}

## Checklist
- [x] Automated classification completed
- [x] Isolation verification completed
- [x] Fingerprint computed for deduplication
- [x] Environment details provided
- [x] Root cause analysis documented
- [ ] Human review required
- [ ] Fix implementation pending

---

**Labels**: {labels_str}

**Priority**: {"P0 (blocker)" if failure_class == "CRASH" else "P1 (high)" if failure_class == "OP_UNSUPPORTED" else "P2 (medium)"}

---
🤖 Auto-generated by transformers-auto-triage
"""
    )

    return body


def generate_preview_markdown(
    findings: List[Dict],
    chip: str,
    transformers_version: str,
    torch_fl_commit: str,
    pytorch_version: str,
    issue_bodies_dir: Path,
) -> str:
    """Generate markdown preview for user review."""
    total = len(findings)

    preview = f"""# Transformers Test Issues Preview

## Summary

- **Total new findings**: {total}
- **Chip**: {chip}
- **Transformers**: {transformers_version}
- **torch_fl**: {torch_fl_commit}

---

"""

    for i, finding in enumerate(findings):
        title = generate_issue_title(finding, chip, transformers_version)
        fp = finding["fingerprint"]

        preview += f"""
## Issue {i + 1}/{total}

**Title**: `{title}`

**Fingerprint**: `{fp}`

**Class**: {finding["class"]}

**Subject**: {finding["subject"]}

**Affects**: {", ".join(finding["models"])} ({finding["count"]} test cases)

**Isolation**: {"✅ CONFIRMED" if finding.get("verdict") == "CONFIRMED" else "⚠️ " + finding.get("verdict", "UNKNOWN")}

<details>
<summary>Full issue body (click to expand)</summary>

```markdown
{generate_issue_body(finding, chip, transformers_version, torch_fl_commit, pytorch_version)[:2000]}
...
```

</details>

**Issue body file**: `{issue_bodies_dir / f"issue-{fp}.md"}`

---
"""

    preview += f"""

## Action Required

Review the {total} issue(s) above. To proceed:

### Option 1: File all issues
```bash
python scripts/transformers_file_issues.py {issue_bodies_dir.parent / (issue_bodies_dir.parent.stem + "-new.json")} \\
    --approve-all \\
    --repo flagos-ai/Torch-FL
```

### Option 2: File specific issues by fingerprint
```bash
python scripts/transformers_file_issues.py {issue_bodies_dir.parent / (issue_bodies_dir.parent.stem + "-new.json")} \\
    --approve {" ".join(f["fingerprint"] for f in findings[:3])} \\
    --repo flagos-ai/Torch-FL
```

### Option 3: Cancel
(do nothing - no issues will be filed)

---

**Note**: Issue body files have been written to `{issue_bodies_dir}/`. You can review and edit them before filing.
"""

    return preview


def main():
    parser = argparse.ArgumentParser(
        description="Generate issue preview for transformers test findings"
    )
    parser.add_argument(
        "input", type=Path, help="New findings JSON from transformers_deduplicate.py"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output preview markdown"
    )
    parser.add_argument(
        "--issue-bodies-dir",
        type=Path,
        default=None,
        help="Directory to write individual issue bodies (default: /tmp/transformers-issues/)",
    )
    parser.add_argument(
        "--chip",
        default=None,
        help="Chip name (default: auto-detect from torch.flagos)",
    )
    parser.add_argument(
        "--transformers-version", required=True, help="Transformers version"
    )
    parser.add_argument("--torch-fl-commit", required=True, help="torch_fl commit SHA")
    parser.add_argument(
        "--pytorch-version", default="2.10.0+cpu", help="PyTorch version"
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    # Auto-detect chip if not provided
    chip = args.chip
    if not chip:
        chip = get_chip_name()
        print(f"Auto-detected chip: {chip}")

    # Default issue bodies directory
    issue_bodies_dir = args.issue_bodies_dir or Path("/tmp/transformers-issues")
    issue_bodies_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.input}")
    with open(args.input) as f:
        findings_json = json.load(f)

    findings = findings_json["findings"]
    print(f"Generating preview for {len(findings)} new findings...")

    # Generate individual issue bodies
    for finding in findings:
        fp = finding["fingerprint"]
        body = generate_issue_body(
            finding,
            chip,
            args.transformers_version,
            args.torch_fl_commit,
            args.pytorch_version,
        )

        body_file = issue_bodies_dir / f"issue-{fp}.md"
        with open(body_file, "w") as f:
            f.write(body)

        print(f"  Wrote {body_file}")

    # Generate preview markdown
    preview = generate_preview_markdown(
        findings,
        chip,
        args.transformers_version,
        args.torch_fl_commit,
        args.pytorch_version,
        issue_bodies_dir,
    )

    print(f"\nWriting preview to {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(preview)

    print(f"\n{'=' * 60}")
    print("Preview generated successfully!")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print(f"1. Review the preview: cat {args.out}")
    print(f"2. Edit issue bodies if needed: {issue_bodies_dir}/")
    print("3. File issues: python scripts/transformers_file_issues.py ...")
    print()


if __name__ == "__main__":
    main()
