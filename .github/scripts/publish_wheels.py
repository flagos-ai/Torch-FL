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

"""Publish built wheels to the vendor Nexus lanes.

Reads the artifact directories the release workflow downloaded, works out which
lane each platform publishes to from ``cmake/flagos_platforms.json`` (the
``pypi_lane`` column), and uploads with twine. Credentials come from
``NEXUS_TOKEN`` in ``user:token`` form -- the same org secret
``flagos-ai/build-infra`` uses -- and are handed to twine through
``TWINE_USERNAME``/``TWINE_PASSWORD`` so they never reach a command line or a
log line.

Run it directly for a dry run:

    NEXUS_TOKEN=... python3 .github/scripts/publish_wheels.py \
        --wheels wheels --tag v2.10.0 --dry-run

The directory layout it expects is what ``actions/download-artifact`` with
``pattern: wheel-*`` produces: one subdirectory per artifact, named
``wheel-<platform>``.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_TABLE = REPO_ROOT / "cmake" / "flagos_platforms.json"
WHEEL_ROOT = "https://resource.flagos.net/repository"
ARTIFACT_PREFIX = "wheel-"


def lanes() -> dict:
    """{platform: lane} for every platform that publishes somewhere."""
    table = json.loads(PLATFORM_TABLE.read_text(encoding="utf-8"))
    return {
        name: row["pypi_lane"]
        for name, row in table["accelerators"].items()
        if row.get("pypi_lane")
    }


def collect(wheels: Path) -> list:
    """[(platform, wheel_path)] from the downloaded artifact tree."""
    search = lanes()
    found = []
    for directory in sorted(p for p in wheels.iterdir() if p.is_dir()):
        if not directory.name.startswith(ARTIFACT_PREFIX):
            continue
        platform = directory.name[len(ARTIFACT_PREFIX) :]
        if platform not in search:
            raise SystemExit(
                f"artifact directory {directory.name!r} does not name a platform "
                f"that publishes a wheel; known: {sorted(search)}"
            )
        for wheel in sorted(directory.glob("*.whl")):
            found.append((platform, wheel))
    if not found:
        raise SystemExit(f"no wheels under {wheels}; nothing to publish")
    return found


def assert_tag_names_the_version(pairs: list, tag: str) -> None:
    """The tag is the base version; each wheel adds its SDK segment.

    `v2.10.0` is the release and `2.10.0+cuda13.3` is one artifact of it, so the
    two agree on everything before the `+`. Checking it here means a tag that
    disagrees with what was built fails before anything is uploaded, rather than
    publishing a version nobody asked for.
    """
    base = tag[1:] if tag.startswith("v") else tag
    for platform, wheel in pairs:
        # torch_fl-2.10.0+cuda13.3-cp312-cp312-linux_x86_64.whl
        match = re.match(r"torch_fl-(?P<version>[^-]+(?:-[^-]+)*?)-cp\d+", wheel.name)
        if not match:
            raise SystemExit(f"cannot read a version out of {wheel.name}")
        version = match.group("version").replace("_", "-")
        if version.split("+")[0] != base:
            raise SystemExit(
                f"{wheel.name} is version {version}, whose base is not the tag "
                f"{tag!r}; refusing to publish"
            )
    print(f"tag {tag} matches the base version of all {len(pairs)} wheel(s)")


def already_published(package: str, version: str, index_url: str) -> bool:
    """Whether the lane already serves this exact version.

    Nexus rejects a duplicate with a 400 and twine reports it as an opaque
    failure, so the check is done first to say something useful instead.
    """
    url = f"{index_url}/{package}/"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            page = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    escaped = re.escape(f"{package}-{version}")
    return re.search(rf"{escaped}-[^\"#<]*\.whl", page) is not None


def upload(wheel: Path, index_url: str, token: str) -> None:
    user, _, password = token.partition(":")
    if not user or not password:
        raise SystemExit("NEXUS_TOKEN must be in 'user:token' form")
    environment = dict(os.environ, TWINE_USERNAME=user, TWINE_PASSWORD=password)
    repository = index_url.removesuffix("/simple")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository-url",
            f"{repository}/",
            "--non-interactive",
            str(wheel),
        ],
        check=True,
        env=environment,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v2.10.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pairs = collect(args.wheels)
    assert_tag_names_the_version(pairs, args.tag)

    token = os.environ.get("NEXUS_TOKEN", "")
    if not token and not args.dry_run:
        raise SystemExit("NEXUS_TOKEN is empty or unset")

    search = lanes()
    uploaded = skipped = 0
    for platform, wheel in pairs:
        index_url = f"{WHEEL_ROOT}/flagos-pypi-{search[platform]}/simple"
        version = wheel.name.split("-cp")[0].removeprefix("torch_fl-")
        print(f"{platform:<11} {wheel.name} -> flagos-pypi-{search[platform]}")
        if already_published("torch_fl", version, index_url):
            # Re-running a release, or a second platform's tag on the same
            # version. Not an error: every other wheel still has to go.
            print("            already published; skipping")
            skipped += 1
            continue
        if args.dry_run:
            print("            [dry-run] would upload")
            uploaded += 1
            continue
        upload(wheel, index_url, token)
        print("            uploaded")
        uploaded += 1

    print(f"\n{uploaded} uploaded, {skipped} already present, {len(pairs)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
