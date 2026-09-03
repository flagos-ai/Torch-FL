#!/usr/bin/env python3
"""
Safe wrapper for transformers testing - designed for weak models.

Weak models (like Qwen-27B or even Sonnet 5) should ONLY call this script.
This script validates all inputs and prevents dangerous operations.

Usage (what weak models should do):
    python scripts/safe_transformers_wrapper.py test bert GCU
    python scripts/safe_transformers_wrapper.py list-models
    python scripts/safe_transformers_wrapper.py batch GCU
"""

import argparse
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

# Allowlist of devices
ALLOWED_DEVICES = [
    "gcu",
    "musa",
    "ascend",
    "metax",
    "ppu",
    "ipu",
    "gaudi",
    "mlu",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def validate_model(model: str) -> str:
    """Validate model name against allowlist."""
    model_lower = model.lower()

    # Check exact match
    if model_lower in [m.lower() for m in ALLOWED_MODELS]:
        return model_lower

    # Check fuzzy match (allow hyphens/underscores)
    for allowed in ALLOWED_MODELS:
        if model_lower.replace("-", "").replace("_", "") == allowed.replace("-", "").replace("_", ""):
            return allowed

    print(f"ERROR: Model '{model}' not in allowlist", file=sys.stderr)
    print(f"Allowed models: {', '.join(ALLOWED_MODELS)}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"To see all available models, run:", file=sys.stderr)
    print(f"  python scripts/safe_transformers_wrapper.py list-models", file=sys.stderr)
    sys.exit(1)


def validate_chip(chip: str) -> str:
    """Validate chip name against allowlist."""
    chip_upper = chip.upper()

    if chip_upper in ALLOWED_CHIPS:
        return chip_upper

    print(f"ERROR: Chip '{chip}' not in allowlist", file=sys.stderr)
    print(f"Allowed chips: {', '.join(ALLOWED_CHIPS)}", file=sys.stderr)
    sys.exit(1)


def validate_device(device: str) -> str:
    """Validate device name against allowlist."""
    device_lower = device.lower()

    if device_lower in ALLOWED_DEVICES:
        return device_lower

    print(f"ERROR: Device '{device}' not in allowlist", file=sys.stderr)
    print(f"Allowed devices: {', '.join(ALLOWED_DEVICES)}", file=sys.stderr)
    sys.exit(1)


def cmd_test(args):
    """Run test for a single model (safe wrapper around transformers_auto_sweep.sh)."""
    model = validate_model(args.model)
    chip = validate_chip(args.chip)
    device = validate_device(args.device)
    repo = args.repo or "flagos-ai/Torch-FL"

    print(f"▶ Running transformers test:")
    print(f"  Model:  {model}")
    print(f"  Device: {device}")
    print(f"  Chip:   {chip}")
    print(f"  Repo:   {repo}")
    print()

    script = REPO_ROOT / "scripts" / "transformers_auto_sweep.sh"
    if not script.exists():
        print(f"ERROR: Script not found: {script}", file=sys.stderr)
        sys.exit(1)

    cmd = ["bash", str(script), model, device, chip, repo]

    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


def cmd_batch(args):
    """Run batch test (bert + qwen3)."""
    chip = validate_chip(args.chip)
    device = validate_device(args.device)
    repo = args.repo or "flagos-ai/Torch-FL"

    print(f"▶ Running batch transformers test:")
    print(f"  Models: bert, qwen3")
    print(f"  Device: {device}")
    print(f"  Chip:   {chip}")
    print(f"  Repo:   {repo}")
    print()

    script = REPO_ROOT / "scripts" / "transformers_batch_sweep.sh"
    if not script.exists():
        print(f"ERROR: Script not found: {script}", file=sys.stderr)
        sys.exit(1)

    cmd = ["bash", str(script), device, chip, repo]

    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


def cmd_list_models(args):
    """List all available models."""
    print("Available models:")
    for model in sorted(ALLOWED_MODELS):
        print(f"  - {model}")

    print()
    print("To test a model:")
    print(f"  python scripts/safe_transformers_wrapper.py test <model> <chip>")
    print()
    print("Example:")
    print(f"  python scripts/safe_transformers_wrapper.py test bert GCU")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Safe wrapper for transformers testing (designed for weak models)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # test command
    test_parser = subparsers.add_parser(
        "test",
        help="test a single model"
    )
    test_parser.add_argument(
        "model",
        help=f"model name (allowed: {', '.join(ALLOWED_MODELS[:5])}...)"
    )
    test_parser.add_argument(
        "chip",
        help=f"chip name for issue titles (e.g., GCU, MUSA)"
    )
    test_parser.add_argument(
        "--device",
        default="gcu",
        help="device name for torch (default: gcu)"
    )
    test_parser.add_argument(
        "--repo",
        help="GitHub repo (default: flagos-ai/Torch-FL)"
    )

    # batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="batch test (bert + qwen3)"
    )
    batch_parser.add_argument(
        "chip",
        help="chip name for issue titles (e.g., GCU, MUSA)"
    )
    batch_parser.add_argument(
        "--device",
        default="gcu",
        help="device name for torch (default: gcu)"
    )
    batch_parser.add_argument(
        "--repo",
        help="GitHub repo (default: flagos-ai/Torch-FL)"
    )

    # list-models command
    subparsers.add_parser(
        "list-models",
        help="list all available models"
    )

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
