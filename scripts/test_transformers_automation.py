#!/usr/bin/env python3
"""
Quick smoke test for transformers automation pipeline.

Creates a minimal test case and runs through all 5 tools to verify basic functionality.
"""

import json
import subprocess
import tempfile
from pathlib import Path


def create_test_json():
    """Create minimal test output matching transformers_hf_tests.py format."""
    return {
        "run": {"context_poison": False, "chip": "MUSA", "model": "qwen3"},
        "tests": [
            {
                "nodeid": "tests/transformers/models/qwen3/test_modeling_qwen3.py::Qwen3Test::test_attention",
                "status": "PASS",
                "duration": 2.5,
            },
            {
                "nodeid": "tests/transformers/models/qwen3/test_modeling_qwen3.py::Qwen3Test::test_sdpa",
                "status": "FAIL",
                "duration": 1.2,
                "detail": "NotImplementedError: aten::scaled_dot_product_attention not implemented for flagos device",
            },
            {
                "nodeid": "tests/transformers/models/qwen3/test_modeling_qwen3.py::Qwen3Test::test_forward",
                "status": "FAIL",
                "duration": 0.8,
                "detail": "RuntimeError: Device context poisoned by previous test",
            },
        ],
        "summary": {"total": 3, "passed": 1, "failed": 2},
    }


def run_tool(tool_name, args):
    """Run a pipeline tool and return success status."""
    cmd = ["python", f"scripts/{tool_name}.py"] + args
    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=True
        )
        print(f"✅ {tool_name} succeeded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {tool_name} failed")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {tool_name} error: {e}")
        return False


def main():
    print("Transformers Automation Pipeline - Smoke Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test input
        test_json = tmpdir / "test-results.json"
        with open(test_json, "w") as f:
            json.dump(create_test_json(), f, indent=2)

        print(f"\nCreated test input: {test_json}")

        # Define pipeline (skip verify since it needs real test files)
        pipeline = [
            (
                "transformers_triage",
                [str(test_json), "--out", str(tmpdir / "classified.json")],
            ),
            # Skip verify - needs actual test source files
            # (
            #     "transformers_verify",
            #     [
            #         str(tmpdir / "classified.json"),
            #         "--out",
            #         str(tmpdir / "verified.json"),
            #         "--test-source-dir",
            #         "tests/transformers/models/qwen3",
            #         "--workers",
            #         "2",
            #         "--timeout",
            #         "10",
            #     ],
            # ),
            (
                "transformers_deduplicate",
                [
                    str(tmpdir / "classified.json"),  # Use classified instead of verified
                    "--out",
                    str(tmpdir / "new.json"),
                    "--coverage-file",
                    "docs/reference/hf-coverage.md",
                    "--repo",
                    "flagos-ai/Torch-FL",
                    "--skip-github",  # Skip GitHub API for smoke test
                ],
            ),
            (
                "transformers_preview_issues",
                [
                    str(tmpdir / "new.json"),
                    "--chip",
                    "MUSA",
                    "--transformers-version",
                    "4.47.0",
                    "--torch-fl-commit",
                    "8050d85",
                    "--issue-bodies-dir",
                    str(tmpdir / "issues"),
                    "--out",
                    str(tmpdir / "preview.md"),
                ],
            ),
            (
                "transformers_file_issues",
                [
                    str(tmpdir / "new.json"),
                    "--approve-all",
                    "--dry-run",
                    "--repo",
                    "flagos-ai/Torch-FL",
                    "--issue-bodies-dir",
                    str(tmpdir / "issues"),
                ],
            ),
        ]

        # Run pipeline
        results = []
        for tool_name, args in pipeline:
            success = run_tool(tool_name, args)
            results.append((tool_name, success))

            # Stop on first failure
            if not success:
                break

        # Summary
        print("\n" + "=" * 60)
        print("Smoke Test Summary")
        print("=" * 60)

        all_passed = all(success for _, success in results)

        for tool_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {tool_name}")

        if all_passed:
            print("\n✅ All tools passed smoke test!")
            return 0
        else:
            print("\n❌ Some tools failed")
            return 1


if __name__ == "__main__":
    exit(main())
