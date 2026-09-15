#!/usr/bin/env python3
"""
AI Agent PR Validation Script

This script validates that AI-generated PRs meet the required standards
before submission. Run this before creating a PR with `gh pr create`.

Usage:
    python scripts/tools/validate_ai_pr.py [--pr-body PR_BODY_FILE]

Exit codes:
    0 - Validation passed
    1 - Validation failed (see error messages)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class Color:
    """Terminal colors for output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_status(message: str, status: str = "INFO"):
    """Print colored status message."""
    colors = {
        "PASS": Color.GREEN,
        "FAIL": Color.RED,
        "WARN": Color.YELLOW,
        "INFO": Color.BLUE,
    }
    color = colors.get(status, "")
    symbol = (
        "✓"
        if status == "PASS"
        else "✗"
        if status == "FAIL"
        else "⚠"
        if status == "WARN"
        else "ℹ"
    )
    print(f"{color}{symbol} {message}{Color.RESET}")


def run_command(cmd: List[str], check: bool = False) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def check_linting() -> bool:
    """Check that ruff linting passes."""
    print(f"\n{Color.BOLD}=== Linting Checks ==={Color.RESET}")

    # Check ruff is available
    ret, _, err = run_command(["ruff", "--version"])
    if ret != 0:
        print_status("ruff not installed. Install with: pip install ruff", "FAIL")
        return False

    # Run ruff check
    print("Running: ruff check")
    ret, stdout, stderr = run_command(["ruff", "check"])
    if ret != 0:
        print_status("ruff check failed", "FAIL")
        print(stdout)
        print(stderr)
        return False
    print_status("ruff check passed", "PASS")

    # Run ruff format --check
    print("Running: ruff format --check")
    ret, stdout, stderr = run_command(["ruff", "format", "--check"])
    if ret != 0:
        print_status("ruff format check failed (code needs formatting)", "FAIL")
        print("Run: ruff format")
        return False
    print_status("ruff format check passed", "PASS")

    return True


def check_tests() -> bool:
    """Check that tests are available (doesn't run them)."""
    print(f"\n{Color.BOLD}=== Test Checks ==={Color.RESET}")

    # Check pytest is available
    ret, _, _ = run_command(["pytest", "--version"])
    if ret != 0:
        print_status("pytest not installed. Install with: pip install pytest", "FAIL")
        return False
    print_status("pytest is available", "PASS")

    # Check test directories exist
    test_dirs = ["tests/unit", "tests/integration"]
    for test_dir in test_dirs:
        if not Path(test_dir).exists():
            print_status(f"Test directory missing: {test_dir}", "WARN")
        else:
            print_status(f"Test directory found: {test_dir}", "PASS")

    print(f"\n{Color.YELLOW}⚠ Note: This script does NOT run tests.{Color.RESET}")
    print(
        f"{Color.YELLOW}  You must run tests manually and include output in your PR.{Color.RESET}"
    )
    print(f"{Color.YELLOW}  Minimum: pytest tests/unit/ -v{Color.RESET}")

    return True


def check_git_status() -> bool:
    """Check git status for uncommitted changes."""
    print(f"\n{Color.BOLD}=== Git Status ==={Color.RESET}")

    ret, stdout, _ = run_command(["git", "status", "--porcelain"])
    if ret != 0:
        print_status("Not in a git repository", "FAIL")
        return False

    if stdout.strip():
        print_status("Uncommitted changes detected", "WARN")
        print(stdout)
        print(
            f"{Color.YELLOW}Ensure all changes are committed before creating PR{Color.RESET}"
        )
    else:
        print_status("Working directory clean", "PASS")

    return True


def check_commit_messages() -> bool:
    """Check recent commit messages follow conventions."""
    print(f"\n{Color.BOLD}=== Commit Message Checks ==={Color.RESET}")

    # Get commits not in main
    ret, stdout, _ = run_command(
        ["git", "log", "main..HEAD", "--pretty=format:%s", "--no-merges"]
    )

    if ret != 0 or not stdout.strip():
        print_status("No commits found (may need to fetch main)", "WARN")
        return True

    commits = stdout.strip().split("\n")
    valid_prefixes = [
        "feat:",
        "fix:",
        "perf:",
        "refactor:",
        "docs:",
        "test:",
        "ci:",
        "build:",
    ]

    all_valid = True
    for commit in commits:
        has_prefix = any(commit.startswith(prefix) for prefix in valid_prefixes)
        if has_prefix:
            print_status(f"Commit OK: {commit[:60]}...", "PASS")
        else:
            print_status(f"Invalid commit format: {commit[:60]}...", "FAIL")
            all_valid = False

    if not all_valid:
        print(
            f"\n{Color.YELLOW}Commit messages should follow format: <type>: <description>{Color.RESET}"
        )
        print(f"{Color.YELLOW}Types: {', '.join(valid_prefixes)}{Color.RESET}")

    return all_valid


def check_pr_body(pr_body_file: str = None) -> bool:
    """Check PR body contains required sections."""
    print(f"\n{Color.BOLD}=== PR Body Checks ==={Color.RESET}")

    if not pr_body_file:
        print_status("No PR body file provided (use --pr-body)", "WARN")
        return True

    if not Path(pr_body_file).exists():
        print_status(f"PR body file not found: {pr_body_file}", "FAIL")
        return False

    with open(pr_body_file, "r", encoding="utf-8") as f:
        body = f.read()

    # Check for required sections in AI PR template
    required_sections = [
        "AI Agent Information",
        "Summary",
        "Problem Analysis",
        "Solution Design",
        "Verification",
        "Pre-submission Checklist",
        "Linting Results",
        "Test Results",
        "Manual Verification",
    ]

    all_present = True
    for section in required_sections:
        if section.lower() in body.lower():
            print_status(f"Section present: {section}", "PASS")
        else:
            print_status(f"Section missing: {section}", "FAIL")
            all_present = False

    # Check language
    # Simple heuristic: look for common Chinese characters
    if re.search(r"[一-鿿]", body):
        print_status("PR body contains non-English text (Chinese detected)", "FAIL")
        print(
            f"{Color.RED}ALL GitHub text must be in English per CLAUDE.md{Color.RESET}"
        )
        all_present = False
    else:
        print_status("Language check passed (English)", "PASS")

    # Check for actual test output (not just claims)
    if "pytest" in body.lower() and (
        "passed" in body.lower() or "failed" in body.lower()
    ):
        print_status("Test output appears to be included", "PASS")
    else:
        print_status(
            "Test output may be missing (should include actual pytest output)", "WARN"
        )

    return all_present


def check_no_debug_code() -> bool:
    """Check for common debug code patterns."""
    print(f"\n{Color.BOLD}=== Debug Code Checks ==={Color.RESET}")

    # Get changed files
    ret, stdout, _ = run_command(
        ["git", "diff", "main...HEAD", "--name-only", "--diff-filter=AM"]
    )

    if ret != 0:
        print_status("Could not get changed files", "WARN")
        return True

    changed_files = [
        f
        for f in stdout.strip().split("\n")
        if f.endswith((".py", ".cc", ".cpp", ".h"))
    ]

    debug_patterns = [
        (r"\bprint\s*\(", "print() call"),
        (r"#\s*TODO", "TODO comment"),
        (r"#\s*FIXME", "FIXME comment"),
        (r"#\s*XXX", "XXX comment"),
        (r"#\s*HACK", "HACK comment"),
        (r"console\.log", "console.log"),
        (r"debugger;", "debugger statement"),
    ]

    issues_found = False
    for file in changed_files:
        if not Path(file).exists():
            continue

        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            for pattern, name in debug_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    print_status(f"{file}:{line_num} contains {name}", "WARN")
                    issues_found = True

    if not issues_found:
        print_status("No obvious debug code found", "PASS")
    else:
        print(
            f"\n{Color.YELLOW}Review these carefully - some may be legitimate.{Color.RESET}"
        )

    return True  # Warning only, not a failure


def main():
    parser = argparse.ArgumentParser(
        description="Validate AI-generated PR before submission"
    )
    parser.add_argument("--pr-body", help="Path to PR body markdown file")
    parser.add_argument("--skip-lint", action="store_true", help="Skip linting checks")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings")
    args = parser.parse_args()

    print(f"{Color.BOLD}{Color.BLUE}")
    print("=" * 60)
    print("  AI Agent PR Validation")
    print("=" * 60)
    print(f"{Color.RESET}")

    checks = [
        ("Git Status", lambda: check_git_status()),
        ("Commit Messages", lambda: check_commit_messages()),
        ("PR Body", lambda: check_pr_body(args.pr_body)),
        ("Debug Code", lambda: check_no_debug_code()),
        ("Tests Available", lambda: check_tests()),
    ]

    if not args.skip_lint:
        checks.insert(0, ("Linting", lambda: check_linting()))

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_status(f"Check '{name}' raised exception: {e}", "FAIL")
            results.append((name, False))

    # Summary
    print(f"\n{Color.BOLD}=== Summary ==={Color.RESET}")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print_status(f"{name}: {status}", status)

    print(f"\n{Color.BOLD}Results: {passed}/{total} checks passed{Color.RESET}")

    if passed == total:
        print(f"{Color.GREEN}{Color.BOLD}")
        print("✓ All checks passed! You can proceed with PR creation.")
        print(f"{Color.RESET}")
        print("\nNext steps:")
        print("  1. Run tests: pytest tests/unit/ -v")
        print("  2. Create PR: gh pr create --template ai_agent_pr.md")
        return 0
    else:
        print(f"{Color.RED}{Color.BOLD}")
        print("✗ Some checks failed. Fix issues before creating PR.")
        print(f"{Color.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
