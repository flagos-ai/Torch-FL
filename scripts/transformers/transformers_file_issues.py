#!/usr/bin/env python3
"""
Transformers Issue Filing Tool

File GitHub issues for approved findings.

Usage:
    # File specific issues after the user approves their fingerprints
    python scripts/transformers/transformers_file_issues.py /tmp/qwen3-new.json \
        --approve a1b2c3d4e5f6 b2c3d4e5f6a7 \
        --repo flagos-ai/Torch-FL

    # Dry run (don't actually file)
    python scripts/transformers/transformers_file_issues.py /tmp/qwen3-new.json \
        --approve a1b2c3d4e5f6 --dry-run
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


def file_github_issue(
    title: str,
    body_file: Path,
    labels: List[str],
    repo: str,
    dry_run: bool,
) -> Optional[int]:
    """
    File a GitHub issue using gh CLI.

    Args:
        title: issue title
        body_file: path to markdown file with issue body
        labels: list of labels to apply
        repo: "owner/repo"
        dry_run: if True, print command but don't execute

    Returns:
        Issue number if created, None on error
    """
    if not body_file.exists():
        raise FileNotFoundError(f"Issue body file not found: {body_file}")

    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--body-file",
        str(body_file),
    ]

    # Add labels
    for label in labels:
        cmd.extend(["--label", label])

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        # Parse issue number from output (gh prints URL)
        # Example: https://github.com/owner/repo/issues/123
        output = result.stdout.strip()
        if "/issues/" in output:
            issue_num = int(output.split("/issues/")[-1])
            return issue_num
        else:
            print(f"Warning: Could not parse issue number from: {output}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error creating issue: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def extract_body_metadata(body_text: str, key: str) -> Optional[str]:
    """Extract one Markdown list item such as ``- **Platform**: MUSA``."""
    match = re.search(
        rf"^- \*\*{re.escape(key)}\*\*: (.+)$",
        body_text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def get_issue_labels(finding: Dict) -> List[str]:
    """Determine appropriate labels for finding."""
    labels = ["ai-generated"]

    failure_class = finding["class"]
    if failure_class == "CRASH":
        labels.extend(["bug", "P0"])
    elif failure_class in (
        "OP_UNSUPPORTED",
        "OP_CPU_FALLBACK",
        "FEATURE_UNSUPPORTED",
    ):
        labels.append("enhancement")
    elif failure_class == "PRECISION":
        labels.append("bug")
    else:
        labels.append("bug")

    return labels


def update_baseline_with_issues(
    findings_json: Dict,
    filed_issues: Dict[str, int],
    coverage_file: Path,
) -> None:
    """
    Update docs/reference/hf-coverage.md with filed issue numbers.

    This appends a findings table to the most recent baseline entry.
    """
    if not filed_issues:
        print("No issues filed, skipping baseline update")
        return

    if not coverage_file.exists():
        print(f"Warning: Coverage file not found: {coverage_file}")
        return

    # Generate findings table
    table_lines = [
        "",
        "| Fingerprint | Class | Subject | Issue |",
        "| --- | --- | --- | --- |",
    ]

    for finding in findings_json["findings"]:
        fp = finding["fingerprint"]
        if fp in filed_issues:
            issue_num = filed_issues[fp]
            table_lines.append(
                f"| `{fp}` | {finding['class']} | {finding['subject']} | "
                f"[#{issue_num}](https://github.com/flagos-ai/Torch-FL/issues/{issue_num}) |"
            )

    table = "\n".join(table_lines)

    # Append to coverage file
    with open(coverage_file, "a") as f:
        f.write("\n")
        f.write(table)
        f.write("\n")

    print(f"Updated {coverage_file} with {len(filed_issues)} issue references")


def main():
    parser = argparse.ArgumentParser(
        description="File GitHub issues for transformers test findings"
    )
    parser.add_argument(
        "input", type=Path, help="New findings JSON from transformers_deduplicate.py"
    )
    parser.add_argument(
        "--issue-bodies-dir",
        type=Path,
        default=Path("/tmp/transformers-issues"),
        help="Directory containing issue body files",
    )
    parser.add_argument(
        "--repo",
        default="flagos-ai/Torch-FL",
        help="GitHub repo (default: flagos-ai/Torch-FL)",
    )

    # Approval options
    approval = parser.add_mutually_exclusive_group(required=True)
    approval.add_argument("--approve", nargs="+", help="File specific fingerprints")

    # Other options
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands but do not file"
    )
    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path("docs/reference/hf-coverage.md"),
        help="Coverage baseline to update (default: docs/reference/hf-coverage.md)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between issue creations in seconds (default: 2.0)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    print(f"Reading {args.input}")
    with open(args.input) as f:
        findings_json = json.load(f)

    findings = findings_json["findings"]

    # Determine which findings to file
    approved_fps = set(args.approve)
    to_file = [f for f in findings if f["fingerprint"] in approved_fps]
    print(f"Filing {len(to_file)} explicitly approved findings")

    # Warn about unknown fingerprints
    found_fps = {f["fingerprint"] for f in to_file}
    unknown = approved_fps - found_fps
    if unknown:
        print(f"Warning: Unknown fingerprints: {unknown}")

    non_confirmed = [f for f in to_file if f.get("verdict") != "CONFIRMED"]
    if non_confirmed:
        subjects = ", ".join(f["subject"] for f in non_confirmed)
        raise ValueError("Only CONFIRMED findings may be filed; blocked: " + subjects)

    incomplete_bodies = []
    for finding in to_file:
        body_file = args.issue_bodies_dir / f"issue-{finding['fingerprint']}.md"
        if not body_file.exists():
            continue
        body_text = body_file.read_text()
        if (
            "Fill in before filing" in body_text
            or "- [ ] Human reviewer has completed" in body_text
        ):
            incomplete_bodies.append(str(body_file))
    if incomplete_bodies:
        raise ValueError(
            "Issue drafts still contain mandatory review placeholders: "
            + ", ".join(incomplete_bodies)
        )

    if not to_file:
        print("No findings to file.")
        return

    # File issues
    filed_issues = {}  # {fingerprint: issue_number}
    failed_issues = []

    for i, finding in enumerate(to_file):
        fp = finding["fingerprint"]
        subject = finding["subject"]

        print(f"\n[{i + 1}/{len(to_file)}] Filing {subject} ({fp})...")

        # Load title from issue body (first line after ## header)
        body_file = args.issue_bodies_dir / f"issue-{fp}.md"
        if not body_file.exists():
            print(f"  Error: Issue body file not found: {body_file}")
            failed_issues.append((fp, "body file not found"))
            continue

        # Extract title from first heading in body
        with open(body_file) as f:
            lines = f.readlines()
            title = None
            for line in lines:
                if line.startswith("## "):
                    # Use the subject as title instead
                    # We'll construct it from finding metadata
                    break

        # Get platform and version metadata from the generated body.
        with open(body_file) as f:
            body_text = f.read()
            chip_match = extract_body_metadata(
                body_text, "Platform"
            ) or extract_body_metadata(body_text, "Chip")
            tf_version = extract_body_metadata(body_text, "Transformers")
            chip_match = chip_match or "Unknown"
            tf_version = tf_version or "unknown"

        # Reconstruct title
        model = finding["models"][0] if finding["models"] else "unknown"
        title = f"[AI][{chip_match}] {model}: {subject}"
        if finding["class"] in ("OP_UNSUPPORTED", "OP_CPU_FALLBACK"):
            action = (
                " uses CPU fallback"
                if finding["class"] == "OP_CPU_FALLBACK"
                else " not supported"
            )
            title += f"{action} (transformers {tf_version})"
        elif finding["class"] == "CRASH":
            title += f" crash (transformers {tf_version})"
        else:
            title += f" failure (transformers {tf_version})"

        labels = get_issue_labels(finding)

        issue_num = file_github_issue(
            title,
            body_file,
            labels,
            args.repo,
            args.dry_run,
        )

        if issue_num:
            filed_issues[fp] = issue_num
            print(f"  ✅ Created issue #{issue_num}")
        else:
            failed_issues.append((fp, "gh command failed"))
            print("  ❌ Failed")

        # Rate limiting delay
        if i < len(to_file) - 1:
            time.sleep(args.delay)

    # Summary
    print(f"\n{'=' * 60}")
    print("Filing complete")
    print(f"{'=' * 60}")
    print(f"Successfully filed: {len(filed_issues)}")
    print(f"Failed: {len(failed_issues)}")

    if filed_issues:
        print("\nFiled issues:")
        for fp, issue_num in filed_issues.items():
            print(f"  #{issue_num}: {fp}")

    if failed_issues:
        print("\nFailed issues:")
        for fp, reason in failed_issues:
            print(f"  {fp}: {reason}")

    # Update baseline
    if filed_issues and not args.dry_run:
        print("\nUpdating baseline...")
        update_baseline_with_issues(
            findings_json,
            filed_issues,
            args.coverage_file,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
