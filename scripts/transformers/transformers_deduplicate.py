#!/usr/bin/env python3
"""
Transformers Test Deduplication Tool

Check if findings are already tracked in:
1. Baseline (docs/reference/hf-coverage.md)
2. GitHub issues

Only outputs NEW findings that should be filed.

Usage:
    python scripts/transformers/transformers_deduplicate.py /tmp/qwen3-verified.json \
        --out /tmp/qwen3-new.json \
        --repo flagos-ai/Torch-FL

Output schema adds to each finding:
    {
      "dedup_status": "NEW"|"IN_BASELINE"|"DUPLICATE"|"COLLATERAL"|"INCONCLUSIVE"|
                      "NOT_ACTIONABLE"|"DEDUP_UNAVAILABLE"|"NOT_CHECKED",
      "dedup_ref": "issue #123" or "baseline:MUSA MTT S5000",
      "should_file": true|false
    }

Only findings whose dedup status is ``NEW`` reach ``findings``. The counts for
every status are recorded under ``summary.dedup``, which is the only way a
caller can tell "nothing new" apart from "the check never ran".
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

BASELINE_HEADING_RE = re.compile(r"^##\s+Baseline:\s*(.+?)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
FINGERPRINT_TOKEN_RE = re.compile(r"\b([a-f0-9]{12})\b")
ISSUE_REF_RE = re.compile(r"#(\d+)")
MARKER_RE = re.compile(r"Fingerprint:?\s*`?([a-f0-9]{12})`?", re.I)
SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")


class GitHubSearchUnavailable(RuntimeError):
    """The GitHub check could not run, so no finding may be called new.

    This is deliberately not ``None``. A search that never executed and a search
    that found nothing are different facts, and the old code conflated them: any
    ``gh`` failure was printed as a warning and the finding was filed again.
    """


def table_cells(line: str) -> list[str]:
    """Split one Markdown table row into its cells."""
    match = TABLE_ROW_RE.match(line)
    if not match:
        return []
    return [cell.strip() for cell in match.group(1).split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(SEPARATOR_CELL_RE.match(cell) for cell in cells)


def hardware_matches(section_label: str, hardware: str) -> bool:
    """Whether a baseline heading describes the hardware a run measured.

    A baseline is scoped to its hardware, and this is the only place the two can
    be told apart: MetaX and MUSA both register their PrivateUse1 device as
    ``flagos``, so the finding's component cannot distinguish them. Vendors name
    boards as vendor plus part number (``MetaX C550``, ``MUSA MTT S5000``), so a
    heading and a chip label describe the same hardware when they share a word.
    """
    words = {word.casefold() for word in hardware.split()}
    return any(word.casefold() in words for word in section_label.split())


def extract_baseline_fingerprints(
    coverage_file: Path, hardware: Optional[str] = None
) -> Dict[str, str]:
    """
    Extract known fingerprints from hf-coverage.md baseline.

    Two spellings are accepted, because the document carries both: the
    ``Fingerprint`` column of a baseline cause table, and the standalone
    ``Fingerprint: `hash``` line that issue bodies use. The reference is taken
    from the row's own issue cell, falling back to the nearest preceding
    ``## Baseline:`` heading, so an unrelated ``#N`` elsewhere in the file can
    no longer be attributed to this fingerprint.

    ``hardware`` scopes the read to the baseline measured on that board. Without
    it every ``## Baseline:`` section is merged, and because both vendors report
    the same device name, a finding measured on one board would be suppressed as
    already known by a section measured on another.

    Returns: {fingerprint: reference}
    """
    if not coverage_file.exists():
        print(f"Warning: Coverage baseline not found: {coverage_file}")
        return {}

    with open(coverage_file) as f:
        content = f.read()

    fingerprints: Dict[str, str] = {}
    baseline = "unknown"
    columns: Dict[str, int] = {}
    eligible = hardware is None
    measured: list[str] = []
    skipped: list[str] = []

    for line in content.splitlines():
        heading = BASELINE_HEADING_RE.match(line)
        if heading:
            baseline = heading.group(1)
            columns = {}
            eligible = hardware is None or hardware_matches(baseline, hardware)
            (measured if eligible else skipped).append(baseline)
            continue

        if not eligible:
            continue

        cells = table_cells(line)
        if cells:
            names = [cell.strip("`* ").lower() for cell in cells]
            if "fingerprint" in names:
                columns = {name: index for index, name in enumerate(names)}
                continue
            if not columns or is_separator_row(cells):
                continue
            index = columns.get("fingerprint")
            if index is None or index >= len(cells):
                continue
            token = FINGERPRINT_TOKEN_RE.search(cells[index])
            if not token:
                continue
            issue = columns.get("issue")
            issue_cell = (
                cells[issue] if issue is not None and issue < len(cells) else ""
            )
            issue_match = ISSUE_REF_RE.search(issue_cell)
            fingerprints[token.group(1)] = (
                f"issue #{issue_match.group(1)}"
                if issue_match
                else f"baseline:{baseline}"
            )
            continue

        for marker in MARKER_RE.finditer(line):
            fingerprints.setdefault(marker.group(1), f"baseline:{baseline}")

    if hardware is None:
        if measured:
            print(
                "Warning: no --hardware given; fingerprints are read from every "
                "baseline section as if this run had measured on each of them"
            )
    else:
        if skipped:
            print(
                f"Skipped {len(skipped)} baseline section(s) measured on other "
                f"hardware: {', '.join(skipped)}"
            )
        if not measured:
            print(
                f"Warning: no baseline section measures '{hardware}'; no finding "
                "can be matched against an earlier measurement on this board"
            )
    print(f"Loaded {len(fingerprints)} fingerprints from baseline")
    return fingerprints


def gh_search(cmd: list[str], description: str) -> Optional[str]:
    """Run one ``gh`` query and return its first matching line, if any.

    A missing binary, a timeout and a non-zero exit are all reported as
    ``GitHubSearchUnavailable``: from the caller's side they mean the same
    thing --- the check did not happen --- and a finding whose duplicate check
    did not happen must not be declared new.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitHubSearchUnavailable(f"{description} timed out") from exc
    except OSError as exc:
        raise GitHubSearchUnavailable(f"{description} could not run: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        reason = detail[0] if detail else f"exit {result.returncode}"
        raise GitHubSearchUnavailable(f"{description} failed: {reason}")

    output = result.stdout.strip()
    return output.split("\n")[0] if output else None


def search_github_issues(
    fingerprint: str,
    repo: str,
    subject: str,
    component: str,
) -> Optional[str]:
    """
    Search GitHub issues for fingerprint.

    Args:
        fingerprint: 12-char hex fingerprint
        repo: "owner/repo"

    Returns:
        "issue #123" if found, None if the search ran and found nothing

    Raises:
        GitHubSearchUnavailable: the search could not be performed
    """
    # Search fingerprints in issue bodies first.
    match = gh_search(
        [
            "gh",
            "api",
            f"repos/{repo}/issues",
            "--paginate",
            "-X",
            "GET",
            "-f",
            "state=all",
            "--jq",
            f'.[] | select(.body // "" | contains("{fingerprint}")) | "#\\(.number) \\(.state) \\(.title)"',
        ],
        "issue body search",
    )
    if match:
        return match

    # Then in comments: a fingerprint is often added while discussing a
    # pre-existing issue rather than in its original body.
    match = gh_search(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/comments",
            "--paginate",
            "-X",
            "GET",
            "--jq",
            f'.[] | select(.body // "" | contains("{fingerprint}")) | "comment on #\\(.issue_url | split("/") | .[-1])"',
        ],
        "issue comment search",
    )
    if match:
        return match

    # Fingerprints are new, so older issues need a semantic fallback. Search
    # only after exact body/comment checks, then require human review before
    # deciding whether the candidate has the same mechanism and component.
    query = f'"{subject}" repo:{repo} is:issue'
    match = gh_search(
        [
            "gh",
            "api",
            "search/issues",
            "-X",
            "GET",
            "-f",
            f"q={query}",
            "--jq",
            '.items[] | "#\\(.number) \\(.state) \\(.title)"',
        ],
        "semantic search",
    )
    if match:
        return f"semantic candidate for {component}: {match}"

    return None


def deduplicate_findings(
    findings_json: Dict,
    coverage_file: Path,
    repo: str,
    skip_github: bool,
    hardware: Optional[str] = None,
) -> Dict:
    """
    Deduplicate findings against baseline and GitHub.

    Args:
        findings_json: verified findings from transformers_verify.py
        coverage_file: path to docs/reference/hf-coverage.md
        repo: GitHub repo "owner/repo"
        skip_github: if True, only check baseline (faster for testing)
        hardware: the board this run measured, so only its own baselines apply

    Returns:
        findings_json with dedup info added and filtered to NEW only
    """
    findings = findings_json["findings"]

    # Load baseline fingerprints
    baseline_fps = extract_baseline_fingerprints(coverage_file, hardware)

    print(f"\nDeduplicating {len(findings)} findings...")

    new_findings = []
    dedup_counts: Dict[str, int] = {
        "NEW": 0,
        "IN_BASELINE": 0,
        "DUPLICATE": 0,
        "COLLATERAL": 0,
        "INCONCLUSIVE": 0,
        "NOT_ACTIONABLE": 0,
        "REVIEW_CANDIDATE": 0,
        "DEDUP_UNAVAILABLE": 0,
        "NOT_CHECKED": 0,
    }

    def record(finding: Dict, status: str, ref: Optional[str], index: int) -> None:
        finding["dedup_status"] = status
        finding["dedup_ref"] = ref
        finding["should_file"] = False
        dedup_counts[status] = dedup_counts.get(status, 0) + 1
        suffix = f" ({ref})" if ref else ""
        print(f"  [{index}/{len(findings)}] {finding['subject']}: {status}{suffix}")

    for i, finding in enumerate(findings, start=1):
        fp = finding["fingerprint"]
        verdict = finding.get("verdict", "UNKNOWN")

        # A finding that claims no platform defect cannot be filed, whatever a
        # later check would say about it.
        if not finding.get("actionable", True):
            record(
                finding,
                "NOT_ACTIONABLE",
                f"{finding['class']} claims no platform defect",
                i,
            )
            continue

        # Skip collateral findings entirely
        if verdict == "COLLATERAL":
            record(finding, "COLLATERAL", "isolation test passed/skipped", i)
            continue

        if verdict == "INCONCLUSIVE":
            record(
                finding,
                "INCONCLUSIVE",
                "isolation did not produce a test verdict",
                i,
            )
            continue

        if verdict != "CONFIRMED":
            record(
                finding,
                "INCONCLUSIVE",
                f"verdict {verdict} is not CONFIRMED",
                i,
            )
            continue

        # Check baseline
        if fp in baseline_fps:
            record(finding, "IN_BASELINE", baseline_fps[fp], i)
            continue

        # Check GitHub issues
        if skip_github:
            record(finding, "NOT_CHECKED", "--skip-github was requested", i)
            continue

        try:
            github_match = search_github_issues(
                fp,
                repo,
                finding["subject"],
                finding.get("component", "unknown"),
            )
        except GitHubSearchUnavailable as exc:
            record(finding, "DEDUP_UNAVAILABLE", str(exc), i)
            continue

        if github_match:
            record(
                finding,
                "REVIEW_CANDIDATE"
                if github_match.startswith("semantic candidate")
                else "DUPLICATE",
                github_match,
                i,
            )
            continue

        finding["dedup_status"] = "NEW"
        finding["dedup_ref"] = None
        finding["should_file"] = True
        dedup_counts["NEW"] += 1
        new_findings.append(finding)
        print(f"  [{i}/{len(findings)}] {finding['subject']}: NEW")

    findings_json["findings"] = new_findings
    findings_json["summary"]["dedup"] = dedup_counts

    return findings_json


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate transformers test findings"
    )
    parser.add_argument(
        "input", type=Path, help="Verified findings JSON from transformers_verify.py"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output new findings JSON"
    )
    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path("docs/reference/hf-coverage.md"),
        help="HF coverage baseline file (default: docs/reference/hf-coverage.md)",
    )
    parser.add_argument(
        "--repo",
        default="flagos-ai/Torch-FL",
        help="GitHub repo for issue search (default: flagos-ai/Torch-FL)",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help=(
            "Hardware label this run measured (e.g. 'MetaX C550'); only baseline "
            "sections measured on the same board are read"
        ),
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip GitHub issue search (faster, for testing)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    print(f"Reading {args.input}")
    with open(args.input) as f:
        findings_json = json.load(f)

    result = deduplicate_findings(
        findings_json,
        args.coverage_file,
        args.repo,
        args.skip_github,
        args.hardware,
    )

    print("\nDeduplication summary:")
    for status, count in result["summary"].get("dedup", {}).items():
        print(f"  {status}: {count}")

    print(f"\nNew findings to file: {len(result['findings'])}")

    print(f"\nWriting {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
