#!/usr/bin/env python3
"""
Transformers Issue Preview Tool

Generate issue bodies for new findings and create a markdown preview for user review.

Usage:
    python scripts/transformers/transformers_preview_issues.py /tmp/qwen3-new.json \
        --out /tmp/qwen3-preview.md \
        --chip "MUSA MTT S5000" \
        --issue-bodies-dir /tmp/qwen3-issues

Output:
- Markdown preview with all issue bodies
- Individual issue body files: <issue-bodies-dir>/issue-<fingerprint>.md
- A metadata sidecar per body: <issue-bodies-dir>/issue-<fingerprint>.json

The sidecar is the only place a title and a label set are decided. The filing
tool reads it instead of rebuilding a title from the body text, so what the
reviewer approves is exactly what gets submitted.

Versions default to the environment recorded by the run that produced the
findings. Deriving them here from the current interpreter would describe the
box that happens to be reviewing the draft, not the box that measured it.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

ISSUE_TYPE_NAMES = (
    "Bug Report",
    "Feature Request",
    "Operator Implementation",
    "Platform Support",
    "Performance Issue",
    "Documentation",
)

# One table decides both the Issue Type checkbox and the GitHub labels, so the
# two can never disagree. No priority label is assigned automatically: the
# classifier cannot tell a P0 from a P2, and a wrong priority on a filed issue
# is worse than none.
ISSUE_CLASSES = {
    "OP_UNSUPPORTED": {
        "type": "Operator Implementation",
        "labels": ["ai-generated", "enhancement"],
    },
    "OP_CPU_FALLBACK": {
        "type": "Operator Implementation",
        "labels": ["ai-generated", "enhancement"],
    },
    "FEATURE_UNSUPPORTED": {
        "type": "Feature Request",
        "labels": ["ai-generated", "enhancement"],
    },
    "PRECISION": {"type": "Bug Report", "labels": ["ai-generated", "bug"]},
    "CRASH": {"type": "Bug Report", "labels": ["ai-generated", "bug"]},
    # These two are reported, never filed; they carry no defect claim.
    "TEST_ERROR": {"type": "Bug Report", "labels": ["ai-generated"]},
    "ENVIRONMENT_ERROR": {"type": "Bug Report", "labels": ["ai-generated"]},
}
DEFAULT_ISSUE_CLASS = {"type": "Bug Report", "labels": ["ai-generated", "bug"]}

# The filing tool refuses any body that still carries one of these. Prose
# placeholders cannot be checked mechanically and were silently publishable.
UNFILLED = "<!-- UNFILLED: {} -->"


def issue_class(failure_class: str) -> Dict:
    return ISSUE_CLASSES.get(failure_class, DEFAULT_ISSUE_CLASS)


def get_chip_name() -> str:
    """Detect the chip from the accelerator extension itself.

    ``torch.flagos`` only exists after ``torch_fl`` has registered the backend,
    so the import is not optional: without it this returned ``"Unknown"`` and
    every title silently said so.
    """
    import torch_fl  # noqa: F401 - registers the ``flagos`` device module
    import torch

    if not hasattr(torch, "flagos"):
        raise RuntimeError(
            "torch_fl did not register a 'flagos' module on torch; "
            "pass --chip explicitly"
        )
    props = torch.flagos.get_device_properties(0)
    name = getattr(props, "name", None)
    if not name:
        raise RuntimeError(
            "torch.flagos.get_device_properties(0) reported no name; "
            "pass --chip explicitly"
        )
    return name


def environment_defaults(findings_json: Dict) -> Dict[str, str]:
    """Read the versions a draft needs from the run that produced it."""
    environment = findings_json.get("environment") or {}
    return {
        "transformers_version": environment.get("transformers")
        or environment.get("source_version")
        or "unknown",
        "torch_fl_commit": environment.get("torch_fl_commit") or "unknown",
        "pytorch_version": environment.get("torch") or "unknown",
        "python_version": environment.get("python") or "unknown",
    }


def head_and_tail(text: str, limit: int = 8000) -> List[str]:
    """Keep both ends of a long traceback, as the runner does.

    The exception and its message live at the tail, the test's own frames at the
    head; a single-sided truncation drops one of them.
    """
    if len(text) <= limit:
        return [text]
    half = limit // 2
    return [text[:half], "... <truncated> ...", text[-half:]]


def generate_issue_title(
    finding: Dict,
    chip: str,
    transformers_version: str,
) -> str:
    """Generate issue title following the convention."""
    model = finding["models"][0] if finding["models"] else "unknown"
    subject = finding["subject"]
    failure_class = finding["class"]

    if failure_class in ("OP_UNSUPPORTED", "OP_CPU_FALLBACK"):
        action = (
            "uses CPU fallback"
            if failure_class == "OP_CPU_FALLBACK"
            else "not supported"
        )
        return f"[AI][{chip}] {model}: {subject} {action} (transformers {transformers_version})"
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
    context: Dict[str, str],
) -> str:
    """
    Generate full issue body following .github/ISSUE_TEMPLATE/ai_agent_issue.md.

    Sections follow the template in its order, and every value this tool cannot
    know becomes an ``UNFILLED`` marker rather than prose. A reviewer has to
    fill those in; the filing tool will not publish a draft that still has one.
    """
    fp = finding["fingerprint"]
    failure_class = finding["class"]
    subject = finding["subject"]
    mechanism = finding["mechanism"]
    models = ", ".join(finding["models"]) or "unknown"
    count = finding["count"]
    nodeid = finding["representative_nodeid"]
    detail = finding["representative_detail"]
    isolation_status = finding.get("isolation_status", "NOT_VERIFIED")
    isolation_detail = finding.get("isolation_detail", "")
    wanted_type = issue_class(failure_class)["type"]
    class_status = {
        "OP_UNSUPPORTED": "This operator is not registered for the flagos backend.",
        "OP_CPU_FALLBACK": "This operator reaches the host through the CPU fallback.",
        "PRECISION": "Numerical output differs from the CPU baseline beyond tolerance.",
        "CRASH": "Runtime crash or device error.",
    }.get(failure_class)

    type_lines = "\n".join(
        f"- [{'x' if name == wanted_type else ' '}] {name}" for name in ISSUE_TYPE_NAMES
    )

    body = f"""## Issue Type
{type_lines}

## AI Agent Information
- **Agent**: Claude Code CLI with Transformers Auto-Triage evidence
- **Model**: Claude Opus 5
- **Session Context**: Transformers coverage measurement on {chip}; human root-cause review is required before publication.

## Summary

Fingerprint: `{fp}`

Test failure in `{nodeid}` indicates `{subject}` {failure_class.lower().replace("_", " ")} on {chip}.

This finding was:
- Observed in {count} test case(s) across {len(finding["models"])} model(s): {models}
- Verified in isolation: {isolation_status}
- Classified as: {failure_class}

## Environment (for bug reports)
<details>
<summary>Click to expand environment details</summary>

- **Platform**: {chip}
- **Python**: {context["python_version"]}
- **PyTorch**: {context["pytorch_version"]}
- **torch_fl**: commit `{context["torch_fl_commit"]}`
- **Transformers**: {context["transformers_version"]}
- **Hardware**: {chip}; {UNFILLED.format("driver and SDK versions")}

**Build config:**
```bash
{UNFILLED.format("build config: the command that produced this torch_fl")}
```

**Runtime config:**
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0
TRANSFORMERS_TEST_DEVICE_SPEC=hf_device_spec.py
FLAGOS_LOG_FALLBACK=1
```
</details>

## Reproduction

The isolated upstream test is the smallest case available today; replace it with
a validated minimal reproducer whenever one exists.

**Minimal reproducer:**
```python
{UNFILLED.format("minimal self-contained reproducer")}
```

**Isolated test command:**
```bash
{finding.get("isolation_command") or "pytest " + nodeid}
```

**Representative test**: `{nodeid}`

## Expected vs Actual Behavior

**Expected:**
The measured operation or feature should execute on {chip} without an unsupported
path, host fallback, numerical defect, or crash.

**Actual:**
```
{mechanism}
```

## Root Cause Analysis

- **Mechanism**: {mechanism}
- **Class**: {failure_class}
- **Subject**: {subject}
- **Backend component**: {finding.get("component", "unknown")}
- **Responsible code location**: {UNFILLED.format("file:line for the first failing call path")}

{f"**Status**: {class_status}" if class_status else ""}

Full captured error output:

```
{chr(10).join(head_and_tail(detail))}
```

## Proposed Solution

"""
    if failure_class == "OP_CPU_FALLBACK":
        body += f"""
1. Add a device implementation for `{subject}` on {chip}
2. Route the operator through the platform's existing code generator and commit regenerated artifacts
3. Remove the CPU round-trip for the measured dtype and shape
4. Add an operator test and rerun the Transformers nodeid
"""
    elif failure_class == "OP_UNSUPPORTED":
        body += f"""
1. Implement `{subject}` for the flagos backend responsible for `{finding.get("component", "unknown")}`
2. Use that platform's existing code generator and commit regenerated artifacts when the platform is not CUDA-compatible
3. Add an operator-level regression test for the measured dtype and shape
4. Rerun the isolated Transformers test and the affected architecture suite
"""
    elif failure_class == "PRECISION":
        body += """
1. Investigate the precision difference vs CPU using the same dtype and seed
2. Identify the responsible operator and compare the backend path against eager CPU
3. Fix the implementation, or propose an upstream tolerance only with measured justification
4. Rerun the isolated nodeid and the affected architecture suite
"""
    else:
        body += """
1. Reduce the failure to the first operation that reproduces it in a fresh process
2. Identify the responsible torch_fl, vendor, or upstream component before filing a fix
3. Add unit and integration regression coverage
4. Rerun the isolated nodeid and then the full architecture suite
"""

    body += f"""
## Verification Plan

- [ ] Unit test: add or update coverage for `{subject}`
- [ ] Integration test: rerun `{nodeid}` in isolation
- [ ] Manual verification: rerun the affected architecture suite and confirm no device poisoning

### Isolation Verification

The finding was re-run in a fresh subprocess to distinguish real failures from
device-poisoning collateral:

- **Isolation status**: {isolation_status}
- **Verdict**: {finding.get("verdict", "UNKNOWN")}
{
        "" + chr(10) + "- **Reclassification**: " + finding["isolation_note"]
        if finding.get("isolation_note")
        else ""
    }

<details>
<summary>Isolation output</summary>

```
{isolation_detail}
```
</details>

## Context & Investigation

- Discovered during a Transformers coverage sweep
- Related models: {models}
- Total occurrences: {count}
- Cause fingerprint checked before filing: `{fp}`
- Isolation outcome: `{finding.get("verdict", "UNKNOWN")}`

## Related Code Locations

- {
        UNFILLED.format(
            "responsible torch_fl, generated backend, vendor, or upstream file:line"
        )
    }

## Checklist - AI Agents MUST Complete All
- [ ] I have provided complete environment information
- [ ] I have included a minimal, self-contained reproducer, or explained why the exact isolated test is the smallest available case
- [x] I have included captured error output
- [ ] I have analyzed the root cause (not just symptoms)
- [ ] I have proposed a specific solution with implementation approach
- [ ] I have identified affected code locations with line numbers
- [x] I have described how to verify the fix
- [x] I have checked for duplicate issues
- [x] All text is in **English**
- [ ] Human reviewer has completed the draft before publication

---

**Suggested labels**: {", ".join(issue_class(failure_class)["labels"])}

---
🤖 Draft generated by transformers-auto-triage; publication requires human review and explicit authorization
"""

    return body


def write_sidecar(
    issue_bodies_dir: Path,
    finding: Dict,
    title: str,
    chips_labels: List[str],
) -> Path:
    """Record the title and labels that go with one body file."""
    sidecar = issue_bodies_dir / f"issue-{finding['fingerprint']}.json"
    sidecar.write_text(
        json.dumps(
            {
                "fingerprint": finding["fingerprint"],
                "title": title,
                "labels": chips_labels,
                "class": finding["class"],
                "subject": finding["subject"],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    return sidecar


def generate_preview_markdown(
    findings: List[Dict],
    chip: str,
    context: Dict[str, str],
    issue_bodies_dir: Path,
    input_path: Path,
) -> str:
    """Generate markdown preview for user review."""
    total = len(findings)

    preview = f"""# Transformers Test Issues Preview

## Summary

- **Total new findings**: {total}
- **Chip**: {chip}
- **Transformers**: {context["transformers_version"]}
- **torch_fl**: {context["torch_fl_commit"]}

---

"""

    for i, finding in enumerate(findings):
        title = generate_issue_title(finding, chip, context["transformers_version"])
        fp = finding["fingerprint"]
        body = generate_issue_body(finding, chip, context)
        shown = body[:4000]
        truncated = (
            f"... <{len(body) - 4000} more characters; read the file>"
            if len(body) > 4000
            else ""
        )
        unfilled = body.count("<!-- UNFILLED:") + body.count(
            "- [ ] Human reviewer has completed"
        )

        preview += f"""
## Issue {i + 1}/{total}

**Title**: `{title}`

Fingerprint: `{fp}`

**Class**: {finding["class"]}

**Subject**: {finding["subject"]}

**Affects**: {", ".join(finding["models"])} ({finding["count"]} test cases)

**Isolation**: {"✅ CONFIRMED" if finding.get("verdict") == "CONFIRMED" else "⚠️ " + finding.get("verdict", "UNKNOWN")}

**Draft state**: {unfilled} field(s) a human must fill before this can be filed

<details>
<summary>Full issue body (click to expand)</summary>

```markdown
{shown}
{truncated}
```

</details>

**Issue body file**: `{issue_bodies_dir / f"issue-{fp}.md"}`

---

"""

    approved = " ".join(f["fingerprint"] for f in findings)
    preview += f"""

## Action Required

Review the {total} issue(s) above. File only the fingerprints that the user
explicitly approved after reviewing each root cause and issue body:

### File explicitly approved issues
```bash
python scripts/transformers/transformers_file_issues.py {input_path} \\
    --approve {approved} \\
    --repo flagos-ai/Torch-FL
```

Every fingerprint in this run is listed above; remove the ones you are not
approving. Approval for one finding does not authorize any other finding or
tracker action.

Each draft still contains fields marked `<!-- UNFILLED: ... -->`. The filing tool
refuses a draft that has any left, so the reviewer must complete them first.

---

**Note**: Issue body files have been written to `{issue_bodies_dir}/`. Review and
edit them before filing.
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
        help="Chip name (default: detect it from torch.flagos)",
    )
    parser.add_argument(
        "--transformers-version",
        default=None,
        help="Transformers version (default: the one recorded in the findings JSON)",
    )
    parser.add_argument(
        "--torch-fl-commit",
        default=None,
        help="torch_fl commit SHA (default: the one recorded in the findings JSON)",
    )
    parser.add_argument(
        "--pytorch-version",
        default=None,
        help="PyTorch version (default: the one recorded in the findings JSON)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    print(f"Reading {args.input}")
    with open(args.input) as f:
        findings_json = json.load(f)

    context = environment_defaults(findings_json)
    context["transformers_version"] = (
        args.transformers_version or context["transformers_version"]
    )
    context["torch_fl_commit"] = args.torch_fl_commit or context["torch_fl_commit"]
    context["pytorch_version"] = args.pytorch_version or context["pytorch_version"]
    for key, value in context.items():
        if value == "unknown":
            print(f"Warning: {key} is not recorded in {args.input}")

    chip = args.chip or get_chip_name()
    if not args.chip:
        print(f"Auto-detected chip: {chip}")

    issue_bodies_dir = args.issue_bodies_dir or Path("/tmp/transformers-issues")
    issue_bodies_dir.mkdir(parents=True, exist_ok=True)

    findings = findings_json["findings"]
    print(f"Generating preview for {len(findings)} new findings...")

    for finding in findings:
        fp = finding["fingerprint"]
        title = generate_issue_title(finding, chip, context["transformers_version"])
        body = generate_issue_body(finding, chip, context)
        labels = issue_class(finding["class"])["labels"]

        body_file = issue_bodies_dir / f"issue-{fp}.md"
        body_file.write_text(body)
        print(f"  Wrote {body_file}")

        sidecar = write_sidecar(issue_bodies_dir, finding, title, labels)
        print(f"  Wrote {sidecar}")

    preview = generate_preview_markdown(
        findings,
        chip,
        context,
        issue_bodies_dir,
        args.input,
    )

    print(f"\nWriting preview to {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(preview)

    print(f"\n{'=' * 60}")
    print("Preview generated successfully!")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print(f"1. Review the preview: cat {args.out}")
    print(f"2. Fill in the UNFILLED fields in {issue_bodies_dir}/issue-<fp>.md")
    print("3. File approved fingerprints with transformers_file_issues.py")
    print()


if __name__ == "__main__":
    main()
