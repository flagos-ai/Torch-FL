# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Record the verified native host wheel and its installed package."""

import argparse
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import zipfile


def record_wheel(wheel: Path, evidence: Path):
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()

    installed = distribution("torch_fl")
    direct_url = json.loads(installed.read_text("direct_url.json") or "{}")
    archive = direct_url.get("archive_info", {})
    expected = archive.get("hashes", {}).get("sha256")
    if expected is None:
        expected = archive.get("hash", "").removeprefix("sha256=")
    if sha256 != expected:
        raise ValueError(
            "the installed torch_fl archive does not match the tested wheel"
        )

    # Read the wheel without importing the package or its native libraries.
    with zipfile.ZipFile(wheel) as bundle:
        names = ["torch_fl/_build_config.py", "torch_fl/compatibility.json"]
        names.extend(
            name
            for name in bundle.namelist()
            if name.startswith("torch_fl/native_host/_generated/")
            and name.endswith(("manifest.json", "coverage.json", "diagnostics.json"))
        )
        for name in names:
            path = evidence / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.read(name))

    record = {
        "wheel": wheel.name,
        "sha256": sha256,
        "package": str(Path(installed.locate_file("torch_fl")).resolve()),
        "plan": json.loads(os.environ.get("CI_PLAN_CONTEXT") or "{}"),
    }
    (evidence / "wheel.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    record_wheel(args.wheel, args.evidence)
