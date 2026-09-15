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
      "dedup_status": "NEW"|"IN_BASELINE"|"DUPLICATE"|"COLLATERAL",
      "dedup_ref": "issue #123" or "baseline:qwen3:2026-09-02",
      "should_file": true|false
    }
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional


def extract_baseline_fingerprints(coverage_file: Path) -> Dict[str, str]:
    """
    Extract known fingerprints from hf-coverage.md baseline.

    Returns: {fingerprint: reference}
    """
    if not coverage_file.exists():
        print(f"Warning: Coverage baseline not found: {coverage_file}")
        return {}

    with open(coverage_file) as f:
        content = f.read()

    fingerprints = {}

    # Pattern: | <fingerprint> | <class> | <subject> | <issue> |
    # or: Fingerprint: `<hash>`
    for match in re.finditer(r"Fingerprint:?\s*`?([a-f0-9]{12})`?", content, re.I):
        fp = match.group(1)

        # Try to find context (which baseline/issue)
        # Look backwards for "Baseline:" or issue reference
        before = content[: match.start()]
        baseline_match = re.search(r"## Baseline:\s*([^\n]+)", before)
        issue_match = re.search(r"#(\d+)", before[-200:])

        if issue_match:
            ref = f"issue #{issue_match.group(1)}"
        elif baseline_match:
            ref = f"baseline:{baseline_match.group(1)}"
        else:
            ref = "baseline:unknown"

        fingerprints[fp] = ref

    print(f"Loaded {len(fingerprints)} fingerprints from baseline")
    return fingerprints


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
        "issue #123" if found, None otherwise
    """
    # Search fingerprints in issue bodies first.
    cmd = [
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
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            first_match = result.stdout.strip().split("\n")[0]
            return first_match

        # Also search in comments
        cmd_comments = [
            "gh",
            "api",
            f"repos/{repo}/issues/comments",
            "--paginate",
            "-X",
            "GET",
            "--jq",
            f'.[] | select(.body // "" | contains("{fingerprint}")) | "comment on #\\(.issue_url | split("/") | .[-1])"',
        ]

        result = subprocess.run(
            cmd_comments,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            first_match = result.stdout.strip().split("\n")[0]
            return first_match

        # Fingerprints are new, so older issues need a semantic fallback. Search
        # only after exact body/comment checks, then require human review before
        # deciding whether the candidate has the same mechanism and component.
        query = f'"{subject}" repo:{repo} is:issue'
        semantic = subprocess.run(
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
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if semantic.returncode == 0 and semantic.stdout.strip():
            first_match = semantic.stdout.strip().split("\n")[0]
            return f"semantic candidate for {component}: {first_match}"

    except subprocess.TimeoutExpired:
        print(f"  Warning: GitHub search timed out for {fingerprint}")
    except Exception as e:
        print(f"  Warning: GitHub search failed for {fingerprint}: {e}")

    return None


def deduplicate_findings(
    findings_json: Dict,
    coverage_file: Path,
    repo: str,
    skip_github: bool,
) -> Dict:
    """
    Deduplicate findings against baseline and GitHub.

    Args:
        findings_json: verified findings from transformers_verify.py
        coverage_file: path to docs/reference/hf-coverage.md
        repo: GitHub repo "owner/repo"
        skip_github: if True, only check baseline (faster for testing)

    Returns:
        findings_json with dedup info added and filtered to NEW only
    """
    findings = findings_json["findings"]

    # Load baseline fingerprints
    baseline_fps = extract_baseline_fingerprints(coverage_file)

    print(f"\nDeduplicating {len(findings)} findings...")

    new_findings = []
    dedup_counts = {
        "NEW": 0,
        "IN_BASELINE": 0,
        "DUPLICATE": 0,
        "COLLATERAL": 0,
        "INCONCLUSIVE": 0,
        "REVIEW_CANDIDATE": 0,
    }

    for i, finding in enumerate(findings):
        fp = finding["fingerprint"]
        verdict = finding.get("verdict", "UNKNOWN")

        # Skip collateral findings entirely
        if verdict == "COLLATERAL":
            finding["dedup_status"] = "COLLATERAL"
            finding["dedup_ref"] = "isolation test passed/skipped"
            finding["should_file"] = False
            dedup_counts["COLLATERAL"] += 1
            print(
                f"  [{i + 1}/{len(findings)}] {finding['subject']}: COLLATERAL (skip)"
            )
            continue

        if verdict == "INCONCLUSIVE":
            finding["dedup_status"] = "INCONCLUSIVE"
            finding["dedup_ref"] = "isolation did not produce a test verdict"
            finding["should_file"] = False
            dedup_counts["INCONCLUSIVE"] += 1
            print(
                f"  [{i + 1}/{len(findings)}] {finding['subject']}: INCONCLUSIVE (skip)"
            )
            continue

        # Check baseline
        if fp in baseline_fps:
            finding["dedup_status"] = "IN_BASELINE"
            finding["dedup_ref"] = baseline_fps[fp]
            finding["should_file"] = False
            dedup_counts["IN_BASELINE"] += 1
            print(
                f"  [{i + 1}/{len(findings)}] {finding['subject']}: IN_BASELINE ({baseline_fps[fp]})"
            )
            continue

        # Check GitHub issues
        if not skip_github:
            github_match = search_github_issues(
                fp,
                repo,
                finding["subject"],
                finding.get("component", "unknown"),
            )
            if github_match:
                finding["dedup_status"] = (
                    "REVIEW_CANDIDATE"
                    if github_match.startswith("semantic candidate")
                    else "DUPLICATE"
                )
                finding["dedup_ref"] = github_match
                finding["should_file"] = False
                dedup_counts.setdefault(finding["dedup_status"], 0)
                dedup_counts[finding["dedup_status"]] += 1
                print(
                    f"  [{i + 1}/{len(findings)}] {finding['subject']}: "
                    f"{finding['dedup_status']} ({github_match})"
                )
                continue

        # New finding
        finding["dedup_status"] = "NEW"
        finding["dedup_ref"] = None
        finding["should_file"] = True
        dedup_counts["NEW"] += 1
        new_findings.append(finding)
        print(f"  [{i + 1}/{len(findings)}] {finding['subject']}: NEW")

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
