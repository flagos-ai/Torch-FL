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
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    Nexus rejects a duplicate with a 400 that twine reports as an opaque
    failure, so the check is done first to say something useful instead.

    Two things about Nexus's PyPI layout this has to get right, both found by
    running it against the real lanes rather than a fixture:

    * **The index path is the normalised name.** `/simple/torch_fl/` is a 404
      and `/simple/torch-fl/` is the page; Nexus does not redirect between
      them. Asking with the underscore form made this return `False` for every
      package whose name contains one -- which is `torch_fl` itself, so the
      idempotent skip never fired and a re-run would have tried to re-upload an
      artifact the lane already had.
    * **The version is followed by `+` when the wheel has a local segment** and
      `-` when it does not: `torch_fl-2.10.0rc1+cann9.0.0-cp311-...` against
      `torch_fl-2.10.0-cp312-...`. Matching only `-` reported "not published"
      for every platform that carries a segment, which is all of them now.
    """
    normalised = package.replace("_", "-").replace(".", "-").lower()
    url = f"{index_url}/{normalised}/"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            page = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    escaped = re.escape(f"{package}-{version}")
    return re.search(rf"{escaped}(\+|-)[^\"#<]*\.whl", page) is not None


def upload(
    wheel: Path, index_url: str, token: str, timeout: int, attempts: int = 2
) -> float:
    """Upload one wheel and return how long it took, in seconds.

    Two things this does that a bare `twine upload` does not:

    * it passes a timeout, because a stalled transfer through the egress proxy
      hangs on a socket read with no output and otherwise consumes the whole
      publish window. Measured from the CI runner, a 388 MB download from this
      same Nexus takes 19 s (~20 MB/s), so several minutes for one wheel is
      already a stall and not slowness;
    * it retries a failure, because the same proxy answers an intermittent
      `503` -- observed repeatedly from a dev box against the same host in the
      same window that other requests succeeded -- and one transient refusal
      should not fail a release. A timeout is not retried: it already cost the
      full budget once, and a stall is not usually transient.
    """
    user, _, password = token.partition(":")
    if not user or not password:
        raise SystemExit("NEXUS_TOKEN must be in 'user:token' form")
    environment = dict(os.environ, TWINE_USERNAME=user, TWINE_PASSWORD=password)
    repository = index_url.removesuffix("/simple")
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "twine",
                    "upload",
                    "--repository-url",
                    f"{repository}/",
                    "--non-interactive",
                    "--disable-progress-bar",
                    str(wheel),
                ],
                check=True,
                env=environment,
                timeout=timeout,
            )
            return time.monotonic() - started
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"            attempt {attempt} failed; retrying", flush=True)
            time.sleep(10)
    raise AssertionError("unreachable")


def publish_one(
    platform: str, lane: str, wheel: Path, index_url: str, token: str, args
) -> tuple[str, str]:
    """Publish one platform's wheel. Returns (platform, outcome)."""
    version = wheel.name.split("-cp")[0].removeprefix("torch_fl-")
    size_mb = wheel.stat().st_size / 1e6
    prefix = f"{platform:<11} {version} -> flagos-pypi-{lane}"
    try:
        if already_published("torch_fl", version, index_url):
            # A re-run of a release, or the same version published twice.
            print(f"{prefix}\n            already published; skipping", flush=True)
            return platform, "skipped"
        if args.dry_run:
            print(f"{prefix}\n            [dry-run] would upload", flush=True)
            return platform, "dry-run"
        print(f"{prefix}\n            uploading {size_mb:.0f} MB...", flush=True)
        elapsed = upload(wheel, index_url, token, args.upload_timeout)
        print(
            f"            uploaded in {elapsed:.1f}s ({size_mb / elapsed:.1f} MB/s)",
            flush=True,
        )
        return platform, "uploaded"
    except subprocess.TimeoutExpired:
        print(
            f"{prefix}\n            TIMED OUT after {args.upload_timeout}s "
            f"({size_mb:.0f} MB)",
            flush=True,
        )
        return platform, "timed out"
    except subprocess.CalledProcessError as error:
        print(
            f"{prefix}\n            FAILED (twine exit {error.returncode})", flush=True
        )
        return platform, "failed"
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        # The idempotency GET, not the upload: this host answers an intermittent
        # 503 (see the note in upload() below), and re-raising it would abort the
        # whole job with a traceback instead of reporting the one lane that needs
        # re-running.
        print(f"{prefix}\n            FAILED (index unreachable: {error})", flush=True)
        return platform, "failed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v2.10.0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--upload-timeout",
        type=int,
        default=900,
        help="seconds any single wheel may take before it is called a stall",
    )
    args = parser.parse_args(argv)

    pairs = collect(args.wheels)
    assert_tag_names_the_version(pairs, args.tag)

    token = os.environ.get("NEXUS_TOKEN", "")
    if not token and not args.dry_run:
        raise SystemExit("NEXUS_TOKEN is empty or unset")

    search = lanes()
    # One wheel per platform, and every platform publishes to its own
    # repository, so no two uploads touch the same lane and there is nothing for
    # them to contend on. Serial uploads bought nothing and cost the whole
    # release: one stalled transfer held the other six, which is what happened
    # to v2.10.0rc1. The org's own release_flaggems.py uploads concurrently for
    # the same reason.
    workers = 1 if args.dry_run else len(pairs)
    outcomes = {}
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {
            pool.submit(
                publish_one,
                platform,
                search[platform],
                wheel,
                f"{WHEEL_ROOT}/flagos-pypi-{search[platform]}/simple",
                token,
                args,
            ): platform
            for platform, wheel in pairs
        }
        for future in as_completed(futures):
            platform, outcome = future.result()
            outcomes[platform] = outcome

    print("\n--- summary ---")
    for platform, _ in pairs:
        print(f"  {platform:<11} {outcomes.get(platform, '?')}")
    failed = sorted(
        platform
        for platform, outcome in outcomes.items()
        if outcome in ("failed", "timed out")
    )
    uploaded = sum(1 for o in outcomes.values() if o == "uploaded")
    skipped = sum(1 for o in outcomes.values() if o == "skipped")
    print(f"{len(pairs)} total: {uploaded} uploaded, {skipped} already present")
    if failed:
        # Every platform was attempted; this is the list that needs re-running,
        # and re-running skips whatever did land.
        print(f"::error::failed for {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
