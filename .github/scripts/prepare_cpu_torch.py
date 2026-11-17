#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Prefetch original CPU and accelerator release wheels and verify delivery."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(directory, version, expected_sha=None, index_url=None, pinned_wheels=()):
    record = json.loads((directory / "cohort.json").read_text())
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported CPU wheelhouse schema")
    actual_version = record["torch_wheel"].split("-", 2)[1]
    if (
        actual_version != version
        if "+" in version
        else actual_version.split("+", 1)[0] != version
    ):
        raise ValueError("CPU wheelhouse version differs from the requested pin")
    if index_url and record["index_url"].rstrip("/") != index_url.rstrip("/"):
        raise ValueError("CPU wheelhouse index differs from the requested pin")
    files = record["files"]
    if set(files) != {p.name for p in directory.glob("*.whl")}:
        raise ValueError("CPU wheelhouse file set differs from its manifest")
    if record["torch_wheel"] not in files:
        raise ValueError("CPU wheelhouse has no pinned Torch wheel")
    for name, expected in files.items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f"CPU wheelhouse checksum mismatch: {name}")
    if expected_sha and files[record["torch_wheel"]] != expected_sha:
        raise ValueError("CPU Torch wheel differs from the pinned artifact")
    for name, expected in pinned_wheels:
        if files.get(name) != expected:
            raise ValueError(
                f"Published wheel differs from the pinned artifact: {name}"
            )
    return record


def download(
    directory,
    version,
    python_version,
    index_url,
    torch_wheel,
    sha256,
    release_wheels=(),
):
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("CPU wheelhouse output must be empty")
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--platform",
        "linux_aarch64",
        "--platform",
        "manylinux2014_aarch64",
        "--platform",
        "manylinux_2_28_aarch64",
        "--python-version",
        python_version,
        "--implementation",
        "cp",
        "--abi",
        "cp" + python_version.replace(".", ""),
        "--dest",
        str(directory),
    ]
    subprocess.run(
        [*command, "--index-url", index_url, "torch==" + version],
        check=True,
    )
    for distribution, release, index, _wheel, _checksum in release_wheels:
        subprocess.run(
            [
                *command,
                "--no-deps",
                "--index-url",
                index,
                distribution + "===" + release,
            ],
            check=True,
        )
    record = {
        "schema_version": 1,
        "index_url": index_url,
        "torch_wheel": torch_wheel,
        "files": {p.name: digest(p) for p in sorted(directory.glob("*.whl"))},
    }
    (directory / "cohort.json").write_text(json.dumps(record, indent=2) + "\n")
    verify(
        directory,
        version,
        sha256,
        index_url,
        [(wheel, checksum) for _, _, _, wheel, checksum in release_wheels],
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "verify"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--python-version")
    parser.add_argument("--torch-wheel")
    parser.add_argument(
        "--release-wheel",
        action="append",
        nargs=5,
        default=[],
        metavar=("DIST", "VERSION", "INDEX", "WHEEL", "SHA256"),
    )
    parser.add_argument(
        "--pinned-wheel",
        action="append",
        nargs=2,
        default=[],
        metavar=("WHEEL", "SHA256"),
    )
    args = parser.parse_args()
    if args.action == "download":
        if not args.python_version or not args.torch_wheel:
            parser.error("download requires --python-version and --torch-wheel")
        record = download(
            args.directory,
            args.version,
            args.python_version,
            args.index_url,
            args.torch_wheel,
            args.sha256,
            args.release_wheel,
        )
    else:
        record = verify(
            args.directory, args.version, args.sha256, args.index_url, args.pinned_wheel
        )
    print(
        f"Verified CPU Torch artifact: {record['torch_wheel']} "
        f"({len(record['files'])} published wheels)"
    )


if __name__ == "__main__":
    main()
