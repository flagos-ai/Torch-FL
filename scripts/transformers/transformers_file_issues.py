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

The title and the labels of every issue come from the sidecar written by
``transformers_preview_issues.py``, and a draft that still carries an
``<!-- UNFILLED: ... -->`` marker, placeholder prose, or an unticked human
review box is refused. Both rules exist for the same reason: what a reviewer
approved is what gets submitted, and a draft that is not finished does not
reach the tracker.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# A draft is filable only when none of these remain. The first is the sentinel
# every unfilled field carries; the second catches the free-text placeholder
# prose earlier drafts used, which no mechanical check could detect; the third
# is the template's own human-review checkbox, still unticked.
UNFILLED_RE = re.compile(r"<!--\s*UNFILLED:\s*(?P<field>[^>]*?)\s*-->")
PROSE_RE = re.compile(r"(?im)^[^\n]*\bfill in\b[^\n]*$")
REVIEW_RE = re.compile(r"- \[ \] Human reviewer has completed")

BASELINE_HEADING_RE = re.compile(r"^##\s+Baseline:\s*(.+?)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
CELL_SEPARATOR_RE = re.compile(r"^:?-{2,}:?$")


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


def unready_markers(body_text: str) -> List[str]:
    """Everything that still blocks a draft from being filed.

    The old gate matched two exact strings, so lowercase placeholder prose and
    any placeholder it did not know about were publishable. This returns the
    reasons themselves, which makes both the refusal and its message precise.
    """
    markers = [f"unfilled field: {field}" for field in UNFILLED_RE.findall(body_text)]
    markers += [
        f"placeholder prose: {line.strip()}" for line in PROSE_RE.findall(body_text)
    ]
    if REVIEW_RE.search(body_text):
        markers.append("human review checklist is still unticked")
    return markers


def read_sidecar(issue_bodies_dir: Path, fingerprint: str) -> Dict:
    """The title and labels the preview decided for one finding.

    The preview owns both so that the filer cannot disagree with what a reviewer
    approved: rebuilding a title from body prose and re-deriving labels from the
    class was a second, silently divergent opinion.
    """
    sidecar = issue_bodies_dir / f"issue-{fingerprint}.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"{sidecar}: metadata sidecar not found; regenerate the drafts with "
            "transformers_preview_issues.py so every body has one"
        )
    metadata = json.loads(sidecar.read_text())
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError(
            f"{sidecar}: records fingerprint {metadata.get('fingerprint')!r}, "
            f"expected {fingerprint!r}"
        )
    if not metadata.get("title"):
        raise ValueError(f"{sidecar}: records no title")
    if not metadata.get("labels"):
        raise ValueError(f"{sidecar}: records no labels")
    return metadata


def split_row(line: str) -> Optional[List[str]]:
    """Split one Markdown table row into its cells, or return None."""
    match = TABLE_ROW_RE.match(line)
    if not match:
        return None
    return [cell.strip() for cell in match.group(1).split("|")]


def is_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(CELL_SEPARATOR_RE.match(cell) for cell in cells)


def render_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def cause_table(lines: List[str]) -> Optional[Dict]:
    """Locate the cause table of the most recent ``## Baseline:`` section.

    The section's first table is the Field/Value run record, so the cause table
    is identified by its header instead: it is the first table below the heading
    that classifies causes, which means a ``Class`` and a ``Subject`` column.
    """
    heading = None
    for index, line in enumerate(lines):
        if BASELINE_HEADING_RE.match(line):
            heading = index
    if heading is None:
        return None

    for index in range(heading + 1, len(lines) - 1):
        header = split_row(lines[index])
        separator = split_row(lines[index + 1])
        if not header or not separator or not is_separator_row(separator):
            continue
        columns = [cell.strip("`* ").lower() for cell in header]
        if "class" not in columns or "subject" not in columns:
            continue
        end = index + 2
        while end < len(lines):
            cells = split_row(lines[end])
            if cells is None or is_separator_row(cells):
                break
            end += 1
        return {
            "header": index,
            "first_row": index + 2,
            "end": end,
            "columns": columns,
        }
    return None


def add_fingerprint_column(lines: List[str], table: Dict) -> None:
    """Give a cause table the column the dedup reader keys on.

    Rewriting rows in place keeps every line index valid, so the caller can keep
    using the table it just measured.
    """
    for index in range(table["header"], table["end"]):
        cells = split_row(lines[index])
        if index == table["header"]:
            cells.insert(0, "Fingerprint")
        elif is_separator_row(cells):
            cells.insert(0, "---")
        else:
            cells.insert(0, "")
        lines[index] = render_row(cells)
    table["columns"] = ["fingerprint", *table["columns"]]


def update_baseline_with_issues(
    findings_json: Dict,
    filed_issues: Dict[str, int],
    coverage_file: Path,
    repo: str,
) -> None:
    """
    Record filed issue numbers in docs/reference/hf-coverage.md.

    Rows are inserted into the most recent baseline's cause table, with the
    ``Fingerprint`` column the deduplication reader reads. The previous version
    appended a separate table after the trailing prose, which the reader never
    looked at, so a filed cause stayed "new" forever.
    """
    if not filed_issues:
        print("No issues filed, skipping baseline update")
        return

    if not coverage_file.exists():
        print(f"Warning: Coverage file not found: {coverage_file}")
        return

    filed = [
        finding
        for finding in findings_json["findings"]
        if finding["fingerprint"] in filed_issues
    ]
    if not filed:
        print(
            "No filed finding is present in the findings JSON, skipping baseline update"
        )
        return

    lines = coverage_file.read_text().splitlines()
    table = cause_table(lines)
    if table is None:
        raise ValueError(
            f"{coverage_file}: no '## Baseline:' section with a Class/Subject cause "
            "table; add the baseline entry before recording filed issues, "
            "because a row written anywhere else can never be read back"
        )
    if "fingerprint" not in table["columns"]:
        add_fingerprint_column(lines, table)

    rows = []
    for finding in filed:
        issue_num = filed_issues[finding["fingerprint"]]
        values = {
            "fingerprint": f"`{finding['fingerprint']}`",
            "class": f"`{finding['class']}`",
            "subject": finding["subject"],
            "affected tests": str(finding.get("count", "")),
            "issue": f"[#{issue_num}](https://github.com/{repo}/issues/{issue_num})",
        }
        rows.append(render_row([values.get(name, "") for name in table["columns"]]))

    lines[table["end"] : table["end"]] = rows
    coverage_file.write_text("\n".join(lines) + "\n")

    print(
        f"Updated {coverage_file} with {len(rows)} issue reference(s) "
        "in the most recent baseline cause table"
    )


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

    # Everything a filing needs is collected before the first GitHub write, so a
    # missing sidecar or an unfinished draft cannot leave a half-filed batch.
    drafts: List[Tuple[Dict, Path, Dict]] = []
    problems: List[str] = []
    for finding in to_file:
        fp = finding["fingerprint"]
        body_file = args.issue_bodies_dir / f"issue-{fp}.md"
        if not body_file.exists():
            problems.append(f"{body_file}: issue body file not found")
            continue
        try:
            metadata = read_sidecar(args.issue_bodies_dir, fp)
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
            continue
        markers = unready_markers(body_file.read_text())
        if markers:
            problems.append(f"{body_file}: " + "; ".join(markers))
            continue
        drafts.append((finding, body_file, metadata))

    if problems:
        raise ValueError(
            "Refusing to file: these drafts are not ready:\n  "
            + "\n  ".join(problems)
            + "\nFill in the drafts and re-run; nothing was filed."
        )

    if not drafts:
        print("No findings to file.")
        return

    # File issues
    filed_issues = {}  # {fingerprint: issue_number}
    failed_issues = []

    for i, (finding, body_file, metadata) in enumerate(drafts):
        fp = finding["fingerprint"]
        subject = finding["subject"]

        print(f"\n[{i + 1}/{len(drafts)}] Filing {subject} ({fp})...")
        print(f"  Title: {metadata['title']}")
        print(f"  Labels: {', '.join(metadata['labels'])}")

        issue_num = file_github_issue(
            metadata["title"],
            body_file,
            metadata["labels"],
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
        if i < len(drafts) - 1:
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
            args.repo,
        )

    print("\nDone.")
    # A run that could not file everything is not a successful run, even if some
    # issues were created; the caller has to see it in the exit code.
    return 1 if failed_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
