# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Copy matching runtime wheels into a separate venv, without accelerator imports."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys
import time


def copy_distribution(distribution, target_site, target_bin, target_python):
    """Copy RECORD-owned files, never expose a source site-packages directory."""
    root = Path(distribution.locate_file("")).resolve()
    record_text = distribution.read_text("RECORD")
    files = distribution.files
    if not record_text or not files:
        return False
    console_scripts = {
        entry.name: entry
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
        and entry.name not in {".", ".."}
        and Path(entry.name).name == entry.name
    }
    for name in console_scripts:
        launcher = target_bin / name
        if launcher.exists() or launcher.is_symlink():
            return False
    # Python 3.12 metadata filters missing files out of distribution.files.
    # Check the complete RECORD before accepting that filtered installation.
    # Missing console launchers are safe because they are regenerated below.
    listed = {str(entry) for entry in files}
    for row in csv.reader(record_text.splitlines()):
        if len(row) != 3:
            return False
        if row[0] not in listed:
            relative = Path(row[0])
            if (
                ".." in relative.parts
                and relative.parent.name == "bin"
                and relative.name in console_scripts
            ):
                continue
            return False
    planned = []
    record = None
    for entry in files:
        relative = Path(str(entry))
        # Console scripts live outside site-packages; regenerate their shebangs
        # for the target interpreter instead of copying the runtime's launchers.
        if relative.is_absolute():
            return False
        if ".." in relative.parts:
            if relative.parent.name == "bin" and relative.name in console_scripts:
                continue
            # Other external files may be essential headers or data. A partial
            # copy would fool pip through its metadata, so let pip install it.
            return False
        if relative.suffix == ".pth" or relative.parts[0].startswith(
            ("torch_npu", "sitecustomize", "usercustomize")
        ):
            return False
        source = Path(distribution.locate_file(entry))
        if not source.resolve().is_relative_to(root) or not source.is_file():
            return False
        destination = target_site / relative
        if destination.exists() or destination.is_symlink():
            return False
        if relative.name == "RECORD" and relative.parent.name.endswith(".dist-info"):
            record = destination
        else:
            planned.append((source, destination, entry))
    if not planned or record is None:
        return False
    # Install metadata last, so pip cannot mistake a partial copy for an
    # installed wheel. Each file is a new copy; later pip operations cannot
    # modify the runtime's packages through symlinks or hardlinks.
    planned.sort(key=lambda pair: ".dist-info" in str(pair[1].relative_to(target_site)))
    records = []
    for source, destination, entry in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = f"{entry.hash.mode}={entry.hash.value}" if entry.hash else ""
        size = entry.size if entry.size is not None else ""
        records.append((str(entry), digest, size))
    for entry in console_scripts.values():
        launcher = target_bin / entry.name
        launcher.write_text(
            f"#!{target_python}\n"
            "import sys\nfrom importlib.metadata import distribution\n"
            f"entry = next(e for e in distribution({distribution.metadata['Name']!r})"
            f".entry_points if e.group == 'console_scripts' and e.name == {entry.name!r})\n"
            "sys.exit(entry.load()())\n"
        )
        launcher.chmod(0o755)
        content = launcher.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
        records.append(
            (
                os.path.relpath(launcher, target_site),
                "sha256=" + digest.decode().rstrip("="),
                len(content),
            )
        )
    records.append((str(record.relative_to(target_site)), "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(records)
    return True


def requirement_type():
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    return Requirement


def requirement_satisfied(value):
    """Check the installed version and dependency closure without a network probe."""
    Requirement = requirement_type()
    seen = set()

    def check(requirement):
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            return True
        distribution = metadata.distribution(requirement.name)
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            return False
        key = (
            requirement.name.lower().replace("_", "-"),
            tuple(sorted(requirement.extras)),
        )
        if key in seen:
            return True
        seen.add(key)
        for text in distribution.requires or []:
            child = Requirement(text)
            if child.marker and not any(
                child.marker.evaluate({"extra": extra})
                for extra in {"", *requirement.extras}
            ):
                continue
            child.marker = None
            if not check(child):
                return False
        return True

    try:
        return check(Requirement(value))
    except (metadata.PackageNotFoundError, ValueError):
        return False


def tool_requirements(path):
    if path is None:
        return {}
    try:
        Requirement = requirement_type()
    except ImportError:
        # A plain SDK image may not have packaging. Keep the original pip path
        # available there; runtime images have it through FlagGems.
        print("Tool reuse skipped: source interpreter has no packaging module")
        return {}
    result = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        result[requirement.name.lower().replace("_", "-")] = requirement.specifier
    return result


def seed(
    target_python,
    torch_version,
    flagtree_version,
    flaggems_version,
    flagcx_version=None,
    requirements_file=None,
):
    target = json.loads(
        subprocess.check_output(
            [
                str(target_python),
                "-I",
                "-c",
                "import json, sys, sysconfig; "
                "print(json.dumps({'site': sysconfig.get_paths()['purelib'], "
                "'version': list(sys.version_info[:2])}))",
            ],
            text=True,
        )
    )
    if target["version"] != list(sys.version_info[:2]):
        print("Runtime reuse skipped: source and target Python versions differ")
        return
    expected = {
        "torch": torch_version + "+cpu",
        "flagtree": flagtree_version,
        "flag-gems": flaggems_version,
        "numpy": None,
    }
    # A runtime may carry FlagCX for its compiler without the CI lane using
    # that communication stack. Copy it only when the caller requests a pin.
    if flagcx_version:
        expected["flagcx"] = flagcx_version
    tools = tool_requirements(requirements_file)
    target_site = Path(target["site"])
    source_paths = site.getsitepackages() + ["/opt/flagtree"]
    started = time.monotonic()
    reused = set()
    for distribution in metadata.distributions(path=source_paths):
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if (name not in expected and name not in tools) or name in reused:
            continue
        if name in expected:
            required = expected[name]
            matches = (
                distribution.version.startswith("1.")
                if name == "numpy"
                else distribution.version == required
            )
        else:
            required = str(tools[name]) or "any installed version"
            matches = tools[name].contains(distribution.version)
        if not matches:
            print(f"Runtime reuse skipped: {name} {distribution.version} != {required}")
            continue
        if copy_distribution(
            distribution, target_site, target_python.parent, target_python
        ):
            reused.add(name)
            print(f"Reused runtime wheel: {name}=={distribution.version}")
        else:
            print(f"Runtime reuse skipped: {name} has incomplete or conflicting files")
    print(f"Runtime reuse: {len(reused)} wheels in {time.monotonic() - started:.1f}s")


def environment_report():
    """Describe the prepared venv without importing any accelerator packages."""
    tools = {}
    for name in ("gcc", "g++", "make", "cmake"):
        path = shutil.which(name)
        tool = {"path": path}
        if path:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                lines = (result.stdout or result.stderr).splitlines()
                tool.update(
                    version=lines[0] if lines else "", exit_code=result.returncode
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                tool["error"] = type(error).__name__
        tools[name] = tool
    return {
        "schema_version": "torch-fl.ci-environment/v1",
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "architecture": platform.machine(),
        "tools": tools,
        "packages": sorted(
            (
                {
                    "name": dist.metadata["Name"],
                    "version": dist.version,
                    "location": str(Path(dist.locate_file("")).resolve()),
                }
                for dist in metadata.distributions()
            ),
            key=lambda package: (package["name"].lower(), package["location"]),
        ),
    }


def main():
    if sys.argv[1:] == ["--report-environment"]:
        print(json.dumps(environment_report(), indent=2))
        return
    if "--check-installed" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Check an installed requirement")
        parser.add_argument("--check-installed", required=True)
        args = parser.parse_args()
        raise SystemExit(0 if requirement_satisfied(args.check_installed) else 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-python", required=True, type=Path)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--flagtree-version", required=True)
    parser.add_argument("--flaggems-version", required=True)
    parser.add_argument("--flagcx-version")
    parser.add_argument("--requirements-file", type=Path)
    args = parser.parse_args()
    seed(
        args.target_python,
        args.torch_version,
        args.flagtree_version,
        args.flaggems_version,
        args.flagcx_version,
        args.requirements_file,
    )


if __name__ == "__main__":
    main()
