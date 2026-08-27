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

"""Check the platform Torch contract, and that installing the wheel kept it.

Each platform declares its own contract in .github/configs/<platform>.yml:

    torch_contract:
      base_version: "2.10.0"      # required: torch.__version__ before "+"
      local_version: cpu          # optional: exact local segment after "+"
      cuda: none                  # required: none | required | any
      forbidden_torch_paths:      # optional: substrings torch.__file__ must not contain
        - /opt/conda/

There is no global default. A CPU-wheel platform and a CUDA-variant platform
cannot share one assertion, so each config states its own and a missing
torch_contract is a configuration error rather than a silent pass.

Run with --stage pre-install to record the interpreter and Torch identity the
platform setup produced, then with --stage post-install to require the wheel
install left it unchanged. `pip install` without --no-deps re-resolves the
wheel's torch pin (torch>=2.10,<2.11) against the default index and replaces the
platform Torch; that shows up here as a changed path, version or CUDA variant.
"""

import argparse
import json
import os
import sys
from pathlib import Path

STAGES = ("pre-install", "post-install")


def _load_contract():
    raw = os.environ.get("TORCH_CONTRACT", "").strip()
    if not raw:
        raise SystemExit("TORCH_CONTRACT is not set")
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid TORCH_CONTRACT JSON: {error}") from error
    if not isinstance(contract, dict) or not contract:
        raise SystemExit("torch_contract must be a non-empty mapping")
    if (
        not isinstance(contract.get("base_version"), str)
        or not contract["base_version"].strip()
    ):
        raise SystemExit("torch_contract.base_version is required")
    if contract.get("cuda", "none") not in ("none", "required", "any"):
        raise SystemExit("torch_contract.cuda must be none, required or any")
    return contract


def _identity(state_file):
    """Probe torch identity in an isolated subprocess to prevent SIGABRT.

    Native backend registration can trigger abort() when the setup environment
    already loaded vendor libraries, and Python cannot catch SIGABRT. Running
    the probe in a subprocess with TORCH_DEVICE_BACKEND_AUTOLOAD=0 prevents
    the backend from initializing while still allowing version/path checks.

    If the subprocess fails, write a structured failure record to state_file
    (executable, return code, signal, stderr excerpt) so the job artifact
    preserves full diagnostic context. Print only a summary to the console.
    """
    import subprocess

    probe = """
import sys
import json
from pathlib import Path
import torch

identity = {
    "executable": sys.executable,
    "torch_path": str(Path(torch.__file__).resolve()),
    "version": torch.__version__,
    "cuda": torch.version.cuda,
}
print(json.dumps(identity))
"""
    env = {**__import__("os").environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    if result.returncode != 0:
        signal_desc = ""
        signal_num = None
        if result.returncode == 134:
            signal_desc = "SIGABRT"
            signal_num = 6
        elif result.returncode < 0:
            signal_num = -result.returncode
            signal_desc = f"signal {signal_num}"

        # Write structured failure record for artifact analysis
        failure_record = {
            "success": False,
            "executable": sys.executable,
            "returncode": result.returncode,
            "signal": signal_num,
            "signal_desc": signal_desc,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps(failure_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Print only summary to console, full stderr is in artifact
        print(f"::error::Torch probe failed: rc={result.returncode} ({signal_desc})")
        stderr_lines = result.stderr.strip().split("\n")
        if stderr_lines:
            print(f"::error::First stderr line: {stderr_lines[0][:200]}")
        print(f"::error::Full diagnostic context saved to artifact: {state_file.name}")
        raise SystemExit(
            f"Cannot probe torch identity: subprocess exited {result.returncode} ({signal_desc})"
        )
    return json.loads(result.stdout)


def _report(stage, identity):
    print(f"[{stage}] sys.executable     = {identity['executable']}")
    print(f"[{stage}] torch.__file__     = {identity['torch_path']}")
    print(f"[{stage}] torch.__version__  = {identity['version']}")
    print(f"[{stage}] torch.version.cuda = {identity['cuda']}")


def _contract_errors(contract, identity):
    errors = []
    expected = contract["base_version"]
    if identity["version"].split("+", 1)[0] != expected:
        errors.append(f"expected Torch {expected}, got {identity['version']}")
    local = contract.get("local_version")
    if isinstance(local, str) and local.strip():
        parts = identity["version"].split("+", 1)
        actual = parts[1] if len(parts) == 2 else ""
        if actual != local:
            errors.append(
                f"expected local version '+{local}', got '{identity['version']}'"
            )
    cuda = contract.get("cuda", "none")
    if cuda == "none" and identity["cuda"] is not None:
        errors.append(
            f"expected a Torch built without CUDA (cuda=None), got cuda={identity['cuda']}"
        )
    if cuda == "required" and identity["cuda"] is None:
        errors.append("expected a CUDA-variant Torch, got cuda=None")
    for fragment in contract.get("forbidden_torch_paths") or ():
        if fragment in identity["torch_path"]:
            errors.append(
                f"Torch resolves inside forbidden path '{fragment}': {identity['torch_path']}"
            )
    return errors


def _drift_errors(before, after):
    """Report every field the wheel install changed."""
    labels = {
        "executable": "sys.executable",
        "torch_path": "torch.__file__",
        "version": "torch.__version__",
        "cuda": "torch.version.cuda",
    }
    errors = []
    for key, label in labels.items():
        if before.get(key) != after.get(key):
            errors.append(
                f"{label} changed across the wheel install: "
                f"{before.get(key)} -> {after.get(key)}"
            )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument(
        "--state-file",
        type=Path,
        required=True,
        help="Where the pre-install identity is recorded and later read back",
    )
    args = parser.parse_args()

    contract = _load_contract()
    identity = _identity(args.state_file)
    _report(args.stage, identity)

    errors = _contract_errors(contract, identity)

    if args.stage == "pre-install":
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        if not args.state_file.is_file():
            raise SystemExit(f"pre-install state file not found: {args.state_file}")
        before = json.loads(args.state_file.read_text(encoding="utf-8"))
        errors.extend(_drift_errors(before, identity))

    if errors:
        print("\n=== Torch Contract Violation ===")
        print(f"Stage: {args.stage}")
        if args.stage == "post-install":
            print("\nBefore install:")
            print(json.dumps(before, indent=2, sort_keys=True))
        print("\nAfter install:")
        print(json.dumps(identity, indent=2, sort_keys=True))
        print("\nErrors:")
        for error in errors:
            print(f"::error::Torch contract ({args.stage}): {error}")
        if args.stage == "post-install":
            print(
                "\n::error::Install the wheel with "
                "`pip install --force-reinstall --no-deps` so the wheel's torch pin "
                "is not re-resolved against the default index."
            )
        return 1

    print(f"Torch contract satisfied at {args.stage}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
