#!/usr/bin/env python3
"""
Safe wrapper for transformers testing - designed for weak models.

Weak models (like Qwen-27B or even Sonnet 5) should ONLY call this script.
This script validates all inputs and prevents dangerous operations.

Usage (what weak models should do):
    python scripts/transformers/safe_transformers_wrapper.py test bert GCU
    python scripts/transformers/safe_transformers_wrapper.py list-models
    python scripts/transformers/safe_transformers_wrapper.py batch GCU

There is no device argument. The device is whatever
``tests/manual/hf_device_spec.py`` registers, and the runner derives it from
that file; a second, unchecked copy here could only ever disagree with it and
was recorded as provenance without being enforced.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Allowlist of models (prevent typos and injections)
ALLOWED_MODELS = [
    "bert",
    "distilbert",
    "roberta",
    "gpt2",
    "t5",
    "bart",
    "qwen3",
    "llama",
    "mistral",
    "gemma",
]

# Allowlist of chips
ALLOWED_CHIPS = [
    "MUSA",
    "GCU",
    "Ascend",
    "MetaX",
    "PPU",
    "IPU",
    "Gaudi",
    "MLU",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_HELPER = REPO_ROOT / "tests" / "manual" / "transformers_hf_source.py"


def cache_root() -> Path:
    """The one cache root the runner and the verifier must both use."""
    spec = importlib.util.spec_from_file_location(
        "transformers_hf_source", SOURCE_HELPER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Path(module.cache_root())


def run_sweep(script: Path, argv: list) -> int:
    """Run one shell step with the shared cache root exported to it."""
    root = cache_root()
    env = dict(os.environ, HF_COVERAGE_CACHE=str(root))
    cmd = ["bash", str(script), *argv]
    print(f"Cache:   {root}")
    print(f"Command: {' '.join(cmd)}")
    print()
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


def validate_model(model: str) -> str:
    """Validate model name against allowlist."""
    model_lower = model.lower()

    # Check exact match
    if model_lower in [m.lower() for m in ALLOWED_MODELS]:
        return model_lower

    # Check fuzzy match (allow hyphens/underscores)
    for allowed in ALLOWED_MODELS:
        if model_lower.replace("-", "").replace("_", "") == allowed.replace(
            "-", ""
        ).replace("_", ""):
            return allowed

    print(f"ERROR: Model '{model}' not in allowlist", file=sys.stderr)
    print(f"Allowed models: {', '.join(ALLOWED_MODELS)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("To see all available models, run:", file=sys.stderr)
    print(
        "  python scripts/transformers/safe_transformers_wrapper.py list-models",
        file=sys.stderr,
    )
    sys.exit(1)


def validate_chip(chip: str) -> str:
    """Validate the vendor named by a chip label and keep the caller's spelling.

    ``--chip`` reaches the report title and the issue preview, and vendors name
    boards with the vendor plus a part number --- ``MetaX C550``,
    ``MUSA MTT S5000``, ``Enflame GCU S60``. Comparing the whole label against a
    vendor list rejected every one of those, and comparing it uppercased
    rejected the mixed-case vendors in the list itself. So the label is accepted
    when any of its words names an allowed vendor, and it is returned as the
    caller wrote it: uppercasing it would put a string in the issue title that
    no vendor uses.
    """
    words = chip.split()
    for word in words:
        for allowed in ALLOWED_CHIPS:
            if word.casefold() == allowed.casefold():
                return chip

    print(f"ERROR: Chip '{chip}' not in allowlist", file=sys.stderr)
    print(f"Allowed chips: {', '.join(ALLOWED_CHIPS)}", file=sys.stderr)
    print(
        "A board name is accepted too, as long as it names a vendor: "
        "'MetaX C550' and 'MUSA MTT S5000' are both valid.",
        file=sys.stderr,
    )
    sys.exit(1)


def cmd_test(args):
    """Run test for a single model (safe wrapper around transformers_auto_sweep.sh)."""
    model = validate_model(args.model)
    chip = validate_chip(args.chip)
    repo = args.repo or "flagos-ai/Torch-FL"

    print("▶ Running transformers test:")
    print(f"  Model:  {model}")
    print(f"  Chip:   {chip}")
    print(f"  Repo:   {repo}")
    print()

    script = REPO_ROOT / "scripts" / "transformers" / "transformers_auto_sweep.sh"
    if not script.exists():
        print(f"ERROR: Script not found: {script}", file=sys.stderr)
        sys.exit(1)

    return run_sweep(script, [model, chip, repo])


def cmd_batch(args):
    """Run batch test (bert + qwen3)."""
    chip = validate_chip(args.chip)
    repo = args.repo or "flagos-ai/Torch-FL"

    print("▶ Running batch transformers test:")
    print("  Models: bert, qwen3")
    print(f"  Chip:   {chip}")
    print(f"  Repo:   {repo}")
    print()

    script = REPO_ROOT / "scripts" / "transformers" / "transformers_batch_sweep.sh"
    if not script.exists():
        print(f"ERROR: Script not found: {script}", file=sys.stderr)
        sys.exit(1)

    return run_sweep(script, [chip, repo])


def cmd_list_models(args):
    """List all available models."""
    print("Available models:")
    for model in sorted(ALLOWED_MODELS):
        print(f"  - {model}")

    print()
    print("To test a model:")
    print(
        "  python scripts/transformers/safe_transformers_wrapper.py test <model> <chip>"
    )
    print()
    print("Example:")
    print("  python scripts/transformers/safe_transformers_wrapper.py test bert GCU")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Safe wrapper for transformers testing (designed for weak models)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # test command
    test_parser = subparsers.add_parser("test", help="test a single model")
    test_parser.add_argument(
        "model", help=f"model name (allowed: {', '.join(ALLOWED_MODELS[:5])}...)"
    )
    test_parser.add_argument(
        "chip", help="chip name for issue titles (e.g., GCU, MUSA)"
    )
    test_parser.add_argument("--repo", help="GitHub repo (default: flagos-ai/Torch-FL)")

    # batch command
    batch_parser = subparsers.add_parser("batch", help="batch test (bert + qwen3)")
    batch_parser.add_argument(
        "chip", help="chip name for issue titles (e.g., GCU, MUSA)"
    )
    batch_parser.add_argument(
        "--repo", help="GitHub repo (default: flagos-ai/Torch-FL)"
    )

    # list-models command
    subparsers.add_parser("list-models", help="list all available models")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "test":
        return cmd_test(args)
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "list-models":
        return cmd_list_models(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
