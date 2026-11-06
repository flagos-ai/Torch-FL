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

"""Prepare and merge line coverage of the installed torch_fl wheel."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import coverage

ROOT = Path(__file__).resolve().parents[2]


def prepare(directory, env_file):
    """Resolve wheel files through metadata without importing the accelerator."""
    package = Path(
        importlib.metadata.distribution("torch_fl").locate_file("torch_fl")
    ).resolve()
    if not package.is_dir() or package == ROOT / "torch_fl":
        raise ValueError(f"Expected an installed torch_fl wheel, found {package}")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"package_root": str(package)}), encoding="utf-8"
    )
    with open(env_file, "a", encoding="utf-8") as handle:
        handle.write(f"FLAGCICD_COVERAGE_DIR={directory}\n")
    print(f"Python line coverage source: {package}")


def merge(directory, output, platform, outcome, source_repository, source_ref):
    """Combine executed lines, preserving missing sessions as incomplete."""
    directory = Path(directory).resolve()
    output = Path(output).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    package = Path(manifest["package_root"]).resolve()
    if not package.is_dir():
        raise ValueError(f"Installed package disappeared: {package}")
    data_files, sessions, incomplete = [], [], []
    for marker in sorted(directory.glob("coverage-*.pending")):
        data_file = marker.with_suffix(".data")
        session = {"data_file": data_file.name}
        try:
            session.update(json.loads(marker.read_text(encoding="utf-8")))
            if not data_file.is_file():
                raise ValueError("Coverage process did not save its data")
            data = coverage.CoverageData(basename=str(data_file))
            data.read()
            if data.has_arcs():
                raise ValueError("Expected line coverage, found branch coverage")
            data_files.append(str(data_file))
            session.update(
                json.loads(marker.with_suffix(".complete").read_text(encoding="utf-8"))
            )
        except (
            OSError,
            ValueError,
            TypeError,
            coverage.exceptions.CoverageException,
        ) as error:
            session["error"] = str(error)
            incomplete.append(data_file.name)
        sessions.append(session)
    if not data_files:
        raise ValueError("No pytest coverage data was saved; no report to upload")

    # Initialize/report from site-packages so the native coverage.py JSON uses
    # portable torch_fl/... filenames, while its denominator is the wheel's
    # Python source, including modules no test imported. Never import torch_fl.
    previous = Path.cwd()
    try:
        os.chdir(package.parent)
        combined = coverage.Coverage(
            data_file=str(directory / "combined.data"),
            config_file=False,
            source=[str(package)],
            branch=False,
        )
        combined.erase()
        combined.combine(data_paths=data_files, strict=True, keep=True)
        combined.save()
        data = combined.get_data()
        if not any(data.lines(filename) for filename in data.measured_files()):
            raise ValueError("No installed torch_fl Python files were measured")
        combined.json_report(outfile=str(output))
    finally:
        os.chdir(previous)
    report = json.loads(output.read_text(encoding="utf-8"))
    metadata = {
        "platform": platform,
        "context": json.loads(os.environ.get("CI_PLAN_CONTEXT") or "{}"),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        "scope": "installed torch_fl Python files; line coverage",
        "outcome": outcome,
        "source_repository": source_repository,
        "source_ref": source_ref,
        "complete": outcome == "success" and not incomplete,
        "sessions": sessions,
        "incomplete_sessions": incomplete,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    print(
        f"Merged {len(data_files)} coverage sessions into {output}: "
        f"{report['totals']['percent_covered']:.2f}% Python line coverage"
    )
    if not metadata["complete"]:
        print("::warning::Python coverage is partial; see coverage metadata.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    setup = subcommands.add_parser("prepare")
    setup.add_argument("--report-dir", required=True)
    setup.add_argument("--env-file", required=True)
    combine = subcommands.add_parser("merge")
    combine.add_argument("--report-dir", required=True)
    combine.add_argument("--output", required=True)
    combine.add_argument("--platform", required=True)
    combine.add_argument("--outcome", choices=["success", "failure"], required=True)
    combine.add_argument("--source-repository", required=True)
    combine.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.report_dir, args.env_file)
    else:
        merge(
            args.report_dir,
            args.output,
            args.platform,
            args.outcome,
            args.source_repository,
            args.source_ref,
        )


if __name__ == "__main__":
    main()
