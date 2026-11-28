#!/usr/bin/env python3
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

"""Record hardware identity for L2+ runs.

When hardware is available, record the machine model, SDK version and
accelerator count.

Emits hardware-<platform>.json with:
- machine_model: from /proc/cpuinfo model name or platform.machine()
- sdk_version: from installed wheel's compatibility.json build.sdk_version
- accelerator_count: from torch_fl.flagos.device_count()
- current_device: from torch_fl.flagos.current_device(), when available

Called after wheel install in integration-tests.yml; travels with pytest artifact.
"""

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys


def get_machine_model():
    """Read machine model from /proc/cpuinfo or fall back to platform.machine()."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def get_sdk_version():
    """Read SDK version from installed wheel's compatibility.json."""
    try:
        import torch_fl

        compat_path = Path(torch_fl.__file__).parent / "compatibility.json"
        if compat_path.exists():
            compat = json.loads(compat_path.read_text())
            return compat.get("build", {}).get("sdk_version", "unknown")
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
    ):
        pass
    return "unknown"


def get_accelerator_count():
    """Get accelerator count from torch_fl.flagos.device_count()."""
    try:
        import torch_fl.flagos

        return torch_fl.flagos.device_count()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return 0


def get_current_device():
    """Record the runtime's active device index for cases that use flagos:0."""
    try:
        import torch_fl.flagos

        return int(torch_fl.flagos.current_device())
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def write_identity(output_path, identity):
    output_path.write_text(json.dumps(identity, indent=2) + "\n")


def collect_identity(output_path, identity):
    for name, probe in (
        ("sdk_version", get_sdk_version),
        ("accelerator_count", get_accelerator_count),
        ("current_device", get_current_device),
    ):
        print(f"Collecting hardware identity: {name}", flush=True)
        identity[name] = probe()
        write_identity(output_path, identity)


def main():
    parser = argparse.ArgumentParser(description="Record hardware identity")
    parser.add_argument("--platform", required=True, help="Platform name")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_path = args.output
    if args.probe:
        collect_identity(output_path, json.loads(output_path.read_text()))
        return

    identity = {
        "machine_model": get_machine_model(),
        "sdk_version": "unknown",
        "accelerator_count": None,
        "current_device": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_identity(output_path, identity)
    try:
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--probe"],
            check=True,
            timeout=args.timeout,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as error:
        try:
            identity.update(json.loads(output_path.read_text()))
        except (OSError, ValueError):
            pass
        identity.update(available=False, probe_error=str(error))
        write_identity(output_path, identity)
        print(f"::warning::Hardware identity probe did not finish: {error}", flush=True)

    print(f"Recorded hardware identity for {args.platform}:")
    print(output_path.read_text())


if __name__ == "__main__":
    main()
