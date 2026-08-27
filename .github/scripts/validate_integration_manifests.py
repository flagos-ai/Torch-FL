# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enforce one integration contract across every platform manifest.

Every platform declares the same baseline groups, identified by id, in the same
relative order. A platform that cannot pass one of them still declares it: the
group runs and fails, and that failure is the record the platform owners act on.
Withholding the group instead reports a green pipeline for a capability that was
never exercised.

Baseline ids, in required order:

    model-availability      preflight   model mount present
    device-availability     preflight   device, Torch, wheel, native libs
    operator-tests          functional  operator dispatch correctness
    rng-tests               functional  RNG dispatch contract
    general-tests           functional  factory / general ops
    amp-contract            functional  torch.amp contract
    profiler-contract       functional  torch.profiler contract
    compile-tests           functional  torch.compile
    inference-tests         functional  Qwen3 inference
    training-tests          functional  Qwen3 training

Platform-specific groups (vendor dispatchers, FlagGems routing, profiler parity)
are additive: any extra id is allowed anywhere, and this validator never treats
one as a substitute for a baseline id.

The command of each baseline functional group is checked too. A group that runs
pytest must name an explicit target under tests/, and must not neutralise the
selection with --deselect, -k or --ignore: an unsupported case has to fail, not
disappear from collection.

Usage:
    python .github/scripts/validate_integration_manifests.py
    python .github/scripts/validate_integration_manifests.py --configs-dir .github/configs
"""

import argparse
import re
from pathlib import Path

import yaml

BASELINE_IDS = (
    "model-availability",
    "device-availability",
    "operator-tests",
    "rng-tests",
    "general-tests",
    "amp-contract",
    "profiler-contract",
    "compile-tests",
    "inference-tests",
    "training-tests",
)

PREFLIGHT_IDS = frozenset({"model-availability", "device-availability"})

# Baseline functional groups must run a real test file or directory under tests/.
# Preflight groups legitimately are not pytest (a device probe, a mount check).
REQUIRED_PYTEST_IDS = tuple(i for i in BASELINE_IDS if i not in PREFLIGHT_IDS)

# Flags that remove tests from a selection. Allowed nowhere in a baseline group:
# a failing case is the deliverable, and deselecting it hides the platform gap.
FORBIDDEN_SELECTION_FLAGS = ("--deselect", "--ignore", "-k ", "--co", "--collect-only")


def _load(config_path):
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _check_ids(entries, errors):
    """Every entry has a unique id; the baseline ids are all present."""
    ids = []
    for index, entry in enumerate(entries, start=1):
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            errors.append(f"integration_tests[{index}] has no id")
            continue
        entry_id = entry_id.strip()
        if entry_id in ids:
            errors.append(f"duplicate id: {entry_id}")
        ids.append(entry_id)

    for baseline in BASELINE_IDS:
        if baseline not in ids:
            errors.append(f"missing baseline group id: {baseline}")
    return ids


def _check_order(ids, errors):
    """The baseline ids appear in the required relative order."""
    present = [i for i in ids if i in BASELINE_IDS]
    expected = [i for i in BASELINE_IDS if i in present]
    if present != expected:
        errors.append(
            "baseline groups are out of order: "
            f"found {present}, expected the relative order {expected}"
        )


def _check_phase_and_policy(entries, errors):
    for entry in entries:
        entry_id = (entry.get("id") or "").strip()
        phase = entry.get("phase")
        policy = entry.get("failure_policy")
        if phase not in ("preflight", "functional"):
            errors.append(
                f"{entry_id or '<no id>'}: phase must be preflight/functional"
            )
        if policy not in ("fail-fast", "continue-after-failure"):
            errors.append(
                f"{entry_id or '<no id>'}: failure_policy must be "
                "fail-fast/continue-after-failure"
            )
        if entry_id in PREFLIGHT_IDS and phase != "preflight":
            errors.append(
                f"{entry_id}: must be phase preflight so the environment is "
                "checked before the functional groups run"
            )
        if entry_id in REQUIRED_PYTEST_IDS and phase != "functional":
            errors.append(f"{entry_id}: must be phase functional")


def _check_commands(entries, errors):
    """Baseline functional groups run real pytest targets and exclude nothing."""
    for entry in entries:
        entry_id = (entry.get("id") or "").strip()
        command = entry.get("command") or ""
        if entry_id not in REQUIRED_PYTEST_IDS:
            continue

        if not re.search(r"(?:^|\s)(?:python3?\s+-m\s+)?pytest(?:\s|$)", command):
            errors.append(
                f"{entry_id}: must invoke pytest on an existing test, not an "
                "inline script or a placeholder command"
            )
            continue
        if not re.search(r"\btests/\S+", command):
            errors.append(f"{entry_id}: pytest has no explicit target under tests/")
        for flag in FORBIDDEN_SELECTION_FLAGS:
            if flag in command:
                errors.append(
                    f"{entry_id}: uses {flag.strip()}, which removes tests from the "
                    "selection; an unsupported case must fail rather than be excluded"
                )
        # A placeholder that reports an environment gap instead of running the
        # real test. The gap belongs in a preflight check, not in a test group.
        if re.search(r"^\s*(echo|printf)\b.*\n\s*exit\s+[1-9]", command, re.MULTILINE):
            errors.append(
                f"{entry_id}: is a placeholder (echo + exit) rather than the real test"
            )


def _check_torch_contract(config, errors):
    """Each platform declares its own Torch contract; there is no global default."""
    contract = config.get("torch_contract")
    if not isinstance(contract, dict) or not contract:
        errors.append("torch_contract is missing")
        return
    base = contract.get("base_version")
    if not isinstance(base, str) or not base.strip():
        errors.append("torch_contract.base_version is required")
    cuda = contract.get("cuda")
    if cuda not in ("none", "required", "any"):
        errors.append("torch_contract.cuda must be none, required or any")


def _check_model_mount(config, errors):
    """The inference/training groups need a real model, mounted read-only."""
    env = config.get("integration_environment") or {}
    model_path = env.get("MODEL_PATH")
    if not isinstance(model_path, str) or not model_path.strip():
        errors.append(
            "integration_environment.MODEL_PATH is required for the "
            "inference/training groups"
        )
        return
    volumes = config.get("container_volumes") or []
    mounted = any(
        isinstance(v, str) and v.split(":")[1:2] == [model_path] for v in volumes
    )
    if not mounted:
        errors.append(
            f"no container_volumes entry mounts MODEL_PATH ({model_path}); "
            "the inference/training groups would fail on a missing model"
        )


def _check_platform(config_path):
    config = _load(config_path)
    entries = config.get("integration_tests") or []
    errors = []
    ids = _check_ids(entries, errors)
    _check_order(ids, errors)
    _check_phase_and_policy(entries, errors)
    _check_commands(entries, errors)
    _check_torch_contract(config, errors)
    _check_model_mount(config, errors)
    return errors, len(entries)


def validate(configs_dir):
    platforms = []
    for config_path in sorted(configs_dir.glob("*.yml")):
        config = _load(config_path)
        # A config without integration_tests is not a hardware platform manifest.
        if not config.get("integration_tests"):
            continue
        platforms.append(config_path)

    if not platforms:
        print(f"::error::no platform manifests found in {configs_dir}")
        return 1

    failed = {}
    for config_path in platforms:
        platform = config_path.stem
        errors, total = _check_platform(config_path)
        if errors:
            failed[platform] = errors
        else:
            print(f"{platform}: OK ({total} groups, all baseline ids present)")

    if failed:
        print("\nManifest contract failures:")
        for platform, errors in failed.items():
            print(f"  {platform}:")
            for error in errors:
                print(f"    - {error}")
        return 1

    print(
        f"\nAll {len(platforms)} platform manifests declare the baseline "
        "contract in order."
    )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".github" / "configs",
        help="Directory containing the platform manifests (default: .github/configs)",
    )
    args = parser.parse_args()
    if not args.configs_dir.is_dir():
        parser.error(f"configs dir not found: {args.configs_dir}")
    return validate(args.configs_dir)


if __name__ == "__main__":
    raise SystemExit(main())
