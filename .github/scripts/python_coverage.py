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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# These adapters belong to other devices, even when setuptools ships them in
# the NVIDIA wheel. Common modules and mixed-platform files remain in scope.
NVIDIA_EXCLUDED_VENDORS = (
    "apple",
    "ascend",
    "bpu",
    "cambricon",
    "enflame",
    "hygon",
    "iluvatar",
    "kunlunxin",
    "metax",
    "mthreads",
    "thead",
    "tsingmicro",
)
NVIDIA_OMIT_DIRECTORIES = tuple(
    f"torch_fl/accelerator/{vendor}/" for vendor in NVIDIA_EXCLUDED_VENDORS
)
NVIDIA_OMIT_MODULES = {
    *(
        f"torch_fl/native_host/backends/{vendor}.py"
        for vendor in NVIDIA_EXCLUDED_VENDORS
    ),
    "torch_fl/compile/flagtree_ascend_policy.py",
    "torch_fl/compile/metal.py",
}
NVIDIA_SCOPE_VERSION = "nvidia-v1"


def nvidia_scope(full_report):
    """Describe wheel files excluded from the NVIDIA line-coverage report."""
    excluded = {
        name: {
            "covered_lines": entry["summary"]["covered_lines"],
            "num_statements": entry["summary"]["num_statements"],
        }
        for name, entry in sorted(full_report["files"].items())
        if name.startswith(NVIDIA_OMIT_DIRECTORIES) or name in NVIDIA_OMIT_MODULES
    }
    return {
        "scope": "installed torch_fl Python files applicable to NVIDIA",
        "scope_version": NVIDIA_SCOPE_VERSION,
        "excluded_files": excluded,
        "excluded_statements": sum(
            data["num_statements"] for data in excluded.values()
        ),
        "scope_conflicts": [
            name for name, data in excluded.items() if data["covered_lines"]
        ],
    }


def summarize_nvidia(report, metadata, source_ref):
    """Recalculate the NVIDIA scope from a complete full-wheel artifact."""
    if (
        metadata.get("platform") not in {"nvidia", "cuda"}
        or metadata.get("source_ref") != source_ref
        or metadata.get("complete") is not True
        or report["meta"]["branch_coverage"]
    ):
        raise ValueError("Expected complete NVIDIA line coverage for the source ref")
    if set(metadata.get("excluded_files", ())).difference(report["files"]):
        raise ValueError("Expected full-wheel coverage, see coverage-full-wheel.json")

    scope = nvidia_scope(report)
    totals = report["totals"]
    statements = totals["num_statements"] - scope["excluded_statements"]
    available = statements > 0 and not scope["scope_conflicts"]
    return {
        "platform": metadata["platform"],
        "source_ref": source_ref,
        "profile": metadata.get("context", {}).get("profile"),
        "run_attempt": metadata["run_attempt"],
        "full_wheel": totals,
        "applicable": {
            "available": available,
            "covered_lines": totals["covered_lines"] if available else None,
            "num_statements": statements,
            "percent_covered": 100 * totals["covered_lines"] / statements
            if available
            else None,
        },
        **scope,
    }


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
    """Report the ordinary wheel and the optional NVIDIA native-host pilot."""
    from coverage.exceptions import CoverageException

    directory = Path(directory).resolve()
    run = {
        "platform": platform,
        "outcome": outcome,
        "source_repository": source_repository,
        "source_ref": source_ref,
        "context": json.loads(os.environ.get("CI_PLAN_CONTEXT") or "{}"),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
    }
    native = directory / "native-host"
    if platform == "nvidia" and native.is_dir():
        try:
            merge_wheel(
                native, native / "coverage-native-host.json", run, native_host=True
            )
        except (OSError, ValueError, KeyError, TypeError, CoverageException) as error:
            print(
                f"::warning::Cannot report native-host Python coverage: {error}",
                file=sys.stderr,
            )
    merge_wheel(directory, Path(output).resolve(), run)


def merge_wheel(directory, output, run, *, native_host=False):
    """Combine executed lines from a single installed wheel."""
    import coverage

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

    # Report from site-packages for portable torch_fl/... paths. Starting with
    # the whole wheel also finds in-scope modules no test imported.
    is_nvidia = run["platform"] in {"nvidia", "cuda"}
    prefix = "native-host-" if native_host else ""
    full_output = (
        directory / f"coverage-{prefix}full-wheel.json" if is_nvidia else output
    )
    scope = {}
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
        combined.json_report(outfile=str(full_output))
        if is_nvidia:
            scope = nvidia_scope(json.loads(full_output.read_text(encoding="utf-8")))
            combined.json_report(
                outfile=str(output), omit=list(scope["excluded_files"])
            )
    finally:
        os.chdir(previous)
    report = json.loads(output.read_text(encoding="utf-8"))
    metadata = {
        **run,
        "scope": "installed torch_fl Python files; line coverage",
        "complete": run["outcome"] == "success"
        and not incomplete
        and not scope.get("scope_conflicts"),
        "sessions": sessions,
        "incomplete_sessions": incomplete,
        **scope,
    }
    if native_host:
        warning_file = directory / "collection-warnings.txt"
        warnings = (
            warning_file.read_text(encoding="utf-8").splitlines()
            if warning_file.is_file()
            else []
        )
        metadata["wheel_variant"] = "native-host"
        metadata["wheel_sha256"] = manifest["wheel_sha256"]
        metadata["collection_warnings"] = warnings
        metadata["complete"] = metadata["complete"] and not warnings
    metadata_file = directory / f"{prefix}metadata.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    if scope.get("scope_conflicts"):
        paths = ", ".join(scope["scope_conflicts"])
        raise ValueError(f"NVIDIA coverage executed excluded files: {paths}")
    print(
        f"Merged {len(data_files)} coverage sessions into {output}: "
        f"{report['totals']['percent_covered']:.2f}% Python line coverage"
    )
    if not metadata["complete"]:
        print(f"::warning::Python coverage is partial; see {metadata_file.name}.")


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
    summarize = subcommands.add_parser("summarize-nvidia")
    summarize.add_argument("--report", type=Path, required=True)
    summarize.add_argument("--metadata", type=Path, required=True)
    summarize.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.report_dir, args.env_file)
    elif args.command == "summarize-nvidia":
        try:
            summary = summarize_nvidia(
                json.loads(args.report.read_text(encoding="utf-8")),
                json.loads(args.metadata.read_text(encoding="utf-8")),
                args.source_ref,
            )
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(str(error)) from error
        print(json.dumps(summary, indent=2))
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
