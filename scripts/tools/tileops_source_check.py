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

"""Run TileOPs code-generation checks without a built ``torch_fl._C``.

``codegen_tileops.py`` needs ``torch_fl.tileops.spec`` but not the compiled
extension. Importing that module from the checkout nevertheless executes
``torch_fl/__init__.py`` and fails on a fresh source tree. The temporary copy
uses a minimal package initializer that exposes only the source modules this
gate needs. This keeps imports rooted at the checkout as well as avoiding the
compiled extension. The generated files are compared with the original
checkout, so this is a read-only source gate.

The caller supplies the pinned TileOPs environment. Keeping the copy logic here
instead of in workflow YAML makes it testable and reusable on a developer host.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (".github", "cmake", "scripts", "csrc", "torch_fl", "tests")
FILES = ("pyproject.toml", "setup.py")
STAGED_INITIALIZER = """from torch_fl import _env, _platform, _vendor  # noqa: F401\n"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter with the pinned TileOPs, TileLang and PyYAML stack",
    )
    args, _ = parser.parse_known_args()
    return args


def stage_source(root: Path, destination: Path) -> None:
    """Copy the generator's inputs and replace the top-level initializer."""
    for name in SOURCE_PATHS:
        source = root / name
        target = destination / name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    for name in FILES:
        shutil.copy2(root / name, destination / name)
    initializer = destination / "torch_fl" / "__init__.py"
    if not initializer.is_file():
        raise SystemExit(f"expected {initializer} in the staged source tree")
    initializer.write_text(STAGED_INITIALIZER, encoding="utf-8")


def run_command(root: Path, command: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="torch-fl-tileops-source-") as temp:
        staged = Path(temp)
        stage_source(root, staged)
        env = dict(os.environ)
        pythonpath = [str(staged)]
        if existing := os.environ.get("PYTHONPATH"):
            pythonpath.append(existing)
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(pythonpath),
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "GEMS_VENDOR": "cuda",
            }
        )
        subprocess.run(command, cwd=staged, env=env, check=True)


def main() -> int:
    args = parse_args()
    forwarded = sys.argv[1:]
    # ``--python`` belongs to this wrapper; argparse has already consumed it.
    filtered = []
    index = 0
    while index < len(forwarded):
        if forwarded[index] == "--python":
            index += 2
            continue
        filtered.append(forwarded[index])
        index += 1
    forwarded = filtered
    if forwarded and forwarded != ["--check"]:
        if forwarded[0] == "--":
            forwarded = forwarded[1:]
        command = forwarded
        if command and command[0] in ("python", "python3"):
            command[0] = args.python
        elif command:
            command = [args.python, *command]
    else:
        command = [
            args.python,
            "scripts/codegen/codegen_tileops.py",
            "--check",
        ]
    run_command(ROOT, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
