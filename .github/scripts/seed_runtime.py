# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare, inspect and validate isolated runtime-image CI environments."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import site
import subprocess
import sys
import tempfile
import time
import zipfile


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


def constraint_requirements():
    """Read the selected local pip constraints, including relative -c files."""
    selected = os.environ.get("PIP_CONSTRAINT")
    if not selected:
        return {}
    Requirement = requirement_type()
    result = {}

    def read(path, parents):
        path = path.resolve()
        if path in parents:
            raise ValueError(f"Cyclic CI constraints include: {path}")
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(("-c ", "--constraint ")):
                _, included = shlex.split(line)
                read(path.parent / included, parents | {path})
                continue
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            name = re.sub(r"[-_.]+", "-", requirement.name).lower()
            if name in result:
                result[name] &= requirement.specifier
            else:
                result[name] = requirement.specifier

    read(Path(selected), set())
    return result


def requirement_satisfied(value):
    """Check the installed version and dependency closure without a network probe."""
    Requirement = requirement_type()
    constraints = constraint_requirements()
    seen = set()

    def check(requirement):
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            return True
        distribution = metadata.distribution(requirement.name)
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            return False
        name = re.sub(r"[-_.]+", "-", requirement.name).lower()
        if name in constraints and not constraints[name].contains(
            distribution.version, prereleases=True
        ):
            return False
        key = (
            name,
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
    constraints = constraint_requirements()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        name = re.sub(r"[-_.]+", "-", requirement.name).lower()
        limits = requirement.specifier
        if name in constraints:
            limits &= constraints[name]
        result[name] = limits
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
    constraints = constraint_requirements()
    target_site = Path(target["site"])
    source_paths = site.getsitepackages() + ["/opt/flagtree"]
    started = time.monotonic()
    reused = set()
    for distribution in metadata.distributions(path=source_paths):
        name = re.sub(r"[-_.]+", "-", distribution.metadata.get("Name", "")).lower()
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
        if name in expected and name in tools:
            matches = matches and tools[name].contains(
                distribution.version, prereleases=True
            )
        if name in constraints:
            matches = matches and constraints[name].contains(
                distribution.version, prereleases=True
            )
            required = f"{required}; CI constraints {constraints[name]}"
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
    packages = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        location = str(Path(distribution.locate_file("")).resolve())
        if not isinstance(name, str) or not name.strip() or not version:
            raise ValueError(
                f"Invalid installed distribution metadata at {location}: "
                f"Name={name!r}, Version={version!r}"
            )
        packages.append({"name": name, "version": version, "location": location})
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
            packages,
            key=lambda package: (package["name"].lower(), package["location"]),
        ),
    }


def defer_missing_flagcx(checks, manifest, platform):
    """Recognize only the explicitly deferred GCU/MUSA FlagCX dependency."""
    if platform not in {"enflame", "gcu", "mthreads", "musa"}:
        return False
    try:
        metadata.distribution("flagcx")
        return False
    except metadata.PackageNotFoundError:
        pass
    if manifest.get("build", {}).get("distributions", {}).get("flagcx"):
        return False
    Requirement = requirement_type()
    required = []
    for text in manifest.get("requirements", []):
        requirement = Requirement(text)
        if requirement.name.lower() == "flagcx" and (
            not requirement.marker or requirement.marker.evaluate({"extra": ""})
        ):
            required.append(text)
    if not required:
        return False
    expected = {f"ERROR: Required {text}; installed missing" for text in required}
    missing = rf"torch[-_.]fl {re.escape(manifest['wheel_version'])} requires flagcx, which is not installed\."
    deferred = False
    for record in checks:
        if record["exit_code"] != 1 or record.get("truncated"):
            continue
        stdout, stderr = (
            record.get("stdout", "").strip(),
            record.get("stderr", "").strip(),
        )
        if record["name"] == "pip-check":
            accepted = not stderr and re.fullmatch(missing, stdout, re.IGNORECASE)
        elif record["name"] == "installed-wheel":
            lines = stderr.splitlines()
            errors = {line for line in lines if line.startswith("ERROR:")}
            accepted = (
                not stdout
                and bool(errors)
                and errors <= expected
                and all(
                    line in expected or line.startswith("WARNING:") for line in lines
                )
            )
        else:
            accepted = False
        if accepted:
            record["deferred_dependency"] = "flagcx"
            deferred = True
    return deferred


def finalize_environment(wheel, output, requirements_file=None):
    """Validate the installed test environment and retain evidence on failure."""
    wheel = wheel.resolve()
    report = {
        "schema_version": "torch-fl.ci-environment/v1",
        "phase": "before-tests",
        "source_ref": os.environ.get("SOURCE_REF", os.environ.get("GITHUB_SHA")),
    }
    report["runtime"] = {
        name: os.environ.get(name)
        for name in (
            "FLAGOS_ACCELERATOR",
            "FLAGOS_SDK_VERSION",
            "FLAGOS_VENDOR_TORCH_VERSION",
            "TORCH_FL_VENDOR_TORCH_VERSION",
            "FLAGOS_VENDOR_TORCH_LIB",
            "FLAGCX_TORCH_BACKEND",
        )
    }
    checks = []

    def check(name, command, timeout=60):
        try:
            with tempfile.TemporaryDirectory(prefix="torch-fl-final-env-") as workdir:
                result = subprocess.run(
                    command,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
                )
            record = {
                "name": name,
                "exit_code": result.returncode,
                "stdout": result.stdout[-20000:],
                "stderr": result.stderr[-20000:],
                "truncated": len(result.stdout) > 20000 or len(result.stderr) > 20000,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            record = {"name": name, "exit_code": 1, "stderr": str(error)}
        checks.append(record)
        print(f"Final environment check: {name}: {record['exit_code']}")
        for stream in ("stdout", "stderr"):
            if record.get(stream):
                print(record[stream])

    try:
        report.update(environment_report())
        digest = hashlib.sha256()
        with wheel.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with zipfile.ZipFile(wheel) as archive:
            manifest = json.loads(archive.read("torch_fl/compatibility.json"))
        report["wheel"] = {
            "file": wheel.name,
            "sha256": digest.hexdigest(),
            "manifest": manifest,
        }
        limits = constraint_requirements()
        report["constraints"] = {
            "path": os.environ.get("PIP_CONSTRAINT"),
            "requirements": {name: str(value) for name, value in limits.items()},
        }
        errors = []
        installed_wheels = [
            item["version"]
            for item in report["packages"]
            if re.sub(r"[-_.]+", "-", item["name"]).lower() == "torch-fl"
        ]
        if installed_wheels != [manifest["wheel_version"]]:
            errors.append(
                f"Installed Torch-FL {installed_wheels} differs from wheel {manifest['wheel_version']}"
            )
        required_tools = tool_requirements(requirements_file)
        report["requirements"] = {
            name: str(value) for name, value in required_tools.items()
        }
        for name, specifier in required_tools.items():
            if not requirement_satisfied(name + str(specifier)):
                errors.append(
                    f"Required CI package or dependency is missing/incompatible: {name}{specifier}"
                )
        for distribution in metadata.distributions():
            name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
            if name in limits and not limits[name].contains(
                distribution.version, prereleases=True
            ):
                errors.append(f"{name}=={distribution.version} violates {limits[name]}")
        for error in errors:
            print(f"::error::{error}")
        checks.append(
            {"name": "ci-ranges", "exit_code": int(bool(errors)), "errors": errors}
        )
        check("pip-check", [sys.executable, "-m", "pip", "check"])
        command = [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2] / "scripts/tools/torch-fl-preflight"
            ),
            "--wheel",
            str(wheel),
            "--check-installed",
            "--check-build-env",
        ]
        if os.environ.get("FLAGOS_ACCELERATOR"):
            command += ["--platform", os.environ["FLAGOS_ACCELERATOR"]]
        if os.environ.get("FLAGOS_SDK_VERSION"):
            command += ["--sdk-version", os.environ["FLAGOS_SDK_VERSION"]]
        check("installed-wheel", command)
        if defer_missing_flagcx(checks, manifest, os.environ.get("FLAGOS_ACCELERATOR")):
            report["deferred_dependencies"] = [
                {
                    "name": "flagcx",
                    "reason": "Published vendor build requires a Torch extension excluded from this CI venv; compatible binary is pending.",
                }
            ]
            print(
                "::warning::FlagCX is explicitly deferred on GCU/MUSA; final dependencies are not complete."
            )
        if not any(
            item["exit_code"] and not item.get("deferred_dependency") for item in checks
        ):
            try:
                metadata.distribution("flagcx")
            except metadata.PackageNotFoundError:
                pass
            else:
                check(
                    "flagcx-import",
                    [
                        sys.executable,
                        "-c",
                        """
import os
os.environ["FLAGCX_TORCH_BACKEND"] = "flagos"
import torch_fl
import flagcx
import torch.distributed as dist
from pathlib import Path
assert dist.is_backend_available("flagcx"), "FlagCX backend was not registered"
maps = Path("/proc/self/maps").read_text()
for forbidden in ("libtorch_npu.so", "libtorch_gcu.so", "libmusa_python.so"):
    assert forbidden not in maps, f"FlagCX loaded forbidden vendor Torch extension: {forbidden}"
for line in maps.splitlines():
    assert not ("/flagos/" in line and "/torch/" in line), "FlagCX escaped the CI Torch core: " + line
print("FlagCX import and backend registration validated")
""",
                    ],
                )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"::error::Final environment record failed: {error}")
        checks.append(
            {"name": "environment-record", "exit_code": 1, "stderr": str(error)}
        )
    report["validation"] = checks
    report["valid"] = all(item["exit_code"] == 0 for item in checks)
    report["ready_for_tests"] = all(
        item["exit_code"] == 0 or item.get("deferred_dependency") == "flagcx"
        for item in checks
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Final environment: valid={report['valid']}, ready_for_tests={report['ready_for_tests']}"
    )
    print(f"Final environment snapshot: {output}")
    return 0 if report["ready_for_tests"] else 1


def main():
    if sys.argv[1:] == ["--report-environment"]:
        print(json.dumps(environment_report(), indent=2))
        return
    if "--finalize-environment" in sys.argv[1:]:
        parser = argparse.ArgumentParser(
            description="Validate the final test environment"
        )
        parser.add_argument(
            "--finalize-environment", action="store_true", required=True
        )
        parser.add_argument("--wheel", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--requirements-file", type=Path)
        args = parser.parse_args()
        raise SystemExit(
            finalize_environment(args.wheel, args.output, args.requirements_file)
        )
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
