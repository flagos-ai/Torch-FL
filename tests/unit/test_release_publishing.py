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

"""The release path: tag -> build every platform -> upload to its lane.

`release.yml` on a `v*` tag builds one wheel per publishing platform and
`.github/scripts/publish_wheels.py` uploads each to the lane named by the
`pypi_lane` column of cmake/flagos_platforms.json. The parts that can be wrong
without a runner are checked here: that the workflow covers every publishing
platform (a missing job would silently ship a release missing a vendor), that
the lanes are one-per-platform, and that the version the tag names is the
version in each wheel.

Nothing here touches the network; the upload itself is exercised by
`--dry-run` in the release job's own log.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_TABLE = REPO_ROOT / "cmake" / "flagos_platforms.json"
RELEASE = REPO_ROOT / ".github" / "workflows" / "release.yml"
CONFIGS = REPO_ROOT / ".github" / "configs"
PUBLISHER = REPO_ROOT / ".github" / "scripts" / "publish_wheels.py"


def _table() -> dict:
    return json.loads(PLATFORM_TABLE.read_text(encoding="utf-8"))["accelerators"]


def publishing_platforms() -> dict:
    """{platform: lane} for every platform that publishes a wheel."""
    return {
        name: row["pypi_lane"] for name, row in _table().items() if row["pypi_lane"]
    }


def _publisher():
    spec = importlib.util.spec_from_file_location("publish_wheels", PUBLISHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_publishing_platform_has_a_build_job_in_the_release_workflow():
    """A platform that publishes but has no job would ship nothing, quietly.

    Scanned as text rather than parsed: this asserts that every platform is
    named somewhere in the workflow, which a regex answers, and tests/unit has
    no YAML parser dependency to add for it. The two spellings it looks for are
    the two the file actually uses -- a platform's own wrapper, and the explicit
    `platform:` input that drives all-tests-common.yml.
    """
    text = RELEASE.read_text(encoding="utf-8")
    missing = [
        platform
        for platform in publishing_platforms()
        if f"build-wheel-{platform}.yml" not in text
        and f"platform: {platform}" not in text
    ]
    assert not missing, f"release.yml has no build job for: {sorted(missing)}"


def test_every_publishing_platform_has_a_ci_config():
    """The release reuses the platform's own pipeline, so the config must exist."""
    for platform in publishing_platforms():
        assert (CONFIGS / f"{platform}.yml").is_file(), platform


def test_pypi_lanes_are_unique_across_platforms():
    """One wheel per lane: two platforms sharing a lane would overwrite."""
    lanes = list(publishing_platforms().values())
    assert len(lanes) == len(set(lanes)), sorted(lanes)


def test_the_publisher_reads_the_same_platforms_the_table_does():
    assert _publisher().lanes() == publishing_platforms()


def test_the_publisher_reads_the_platform_out_of_the_artifact_directory(tmp_path):
    publisher = _publisher()
    for platform in publishing_platforms():
        directory = tmp_path / f"wheel-{platform}"
        directory.mkdir()
        (directory / "torch_fl-2.10.0+x-cp312-cp312-linux_x86_64.whl").touch()
    found = publisher.collect(tmp_path)
    assert {platform for platform, _ in found} == set(publishing_platforms())


def test_an_unexpected_artifact_directory_is_refused(tmp_path):
    """Better to stop than to upload a wheel to a lane nobody chose."""
    publisher = _publisher()
    (tmp_path / "wheel-nonesuch").mkdir()
    (
        tmp_path / "wheel-nonesuch" / "torch_fl-2.10.0-cp312-cp312-linux_x86_64.whl"
    ).touch()
    with pytest.raises(SystemExit, match="does not name a platform"):
        publisher.collect(tmp_path)


@pytest.mark.parametrize(
    "wheel, tag, ok",
    [
        ("torch_fl-2.10.0+cuda13.3-cp312-cp312-linux_x86_64.whl", "v2.10.0", True),
        ("torch_fl-2.10.0+dtk2604-cp310-cp310-linux_x86_64.whl", "v2.10.0", True),
        ("torch_fl-2.10.0+maca3.8.1.3-cp312-cp312-linux_x86_64.whl", "v2.10.0", True),
        ("torch_fl-2.10.0-cp312-cp312-linux_x86_64.whl", "v2.10.0", True),
        # The tag is the release and the segment is the SDK, so a mismatch is a
        # release that would publish a version nobody asked for.
        ("torch_fl-2.10.1+cuda13.3-cp312-cp312-linux_x86_64.whl", "v2.10.0", False),
        ("torch_fl-2.11.0+cuda13.3-cp312-cp312-linux_x86_64.whl", "v2.10.0", False),
    ],
)
def test_the_tag_must_name_the_versions_base(wheel, tag, ok, tmp_path):
    publisher = _publisher()
    path = tmp_path / wheel
    path.touch()
    pairs = [("cuda", path)]
    if ok:
        publisher.assert_tag_names_the_version(pairs, tag)
    else:
        with pytest.raises(SystemExit, match="not the tag"):
            publisher.assert_tag_names_the_version(pairs, tag)


# --- Pre-release tags --------------------------------------------------------

RELEASE_GUARD_PATTERNS = {
    # The two regexes release.yml applies, in order.
    "stable": r"^v?[0-9]+(\.[0-9]+){0,3}(\.post[0-9]+)?$",
    "prerelease": r"^v?[0-9]+(\.[0-9]+){0,3}(a|b|rc)[0-9]+$",
}


def _guard(tag: str) -> bool:
    """Whether release.yml would publish a wheel for this tag."""
    import re

    for name in ("stable", "prerelease"):
        if re.fullmatch(RELEASE_GUARD_PATTERNS[name], tag):
            return True
    return False


@pytest.mark.parametrize(
    "tag, publishes",
    [
        ("v2.10.0", True),
        ("v2.10.0.post1", True),
        # A candidate is publishable on purpose: an rc nobody can install
        # cannot be tested, which is the only reason to cut one.
        ("v2.10.0rc1", True),
        ("v2.10.0rc10", True),
        ("v2.10.0a1", True),
        ("v2.10.0b2", True),
        # A variant tag carries the SDK the artifact already carries; letting it
        # through would publish the same wheel twice under two tags.
        ("v2.10.0+cuda13.3", False),
        ("v2.10.0-rc1", False),
        ("v2.10.0.dev1", False),
    ],
)
def test_the_release_guard_publishes_releases_and_candidates_only(tag, publishes):
    assert _guard(tag) is publishes, tag


def test_the_release_guard_patterns_are_the_ones_in_the_workflow():
    """The table above is only worth anything if it matches the real workflow.

    Three separate things have to hold, and each has been got wrong once while
    writing this: the patterns are the ones the workflow defines, the pre-release
    branch is actually reached (an `elif` that assigns `publish=true`), and the
    stable branch comes first so `v2.10.0` is not read as a candidate.
    """
    text = RELEASE.read_text(encoding="utf-8")
    for name, pattern in RELEASE_GUARD_PATTERNS.items():
        assert f"{name}='{pattern}'" in text, name
    stable_at = text.index('if [[ "$REF_NAME" =~ $stable ]]')
    prerelease_at = text.index('elif [[ "$REF_NAME" =~ $prerelease ]]')
    assert stable_at < prerelease_at, "the stable branch has to be tested first"
    branch = text[prerelease_at : text.index("else", prerelease_at)]
    assert 'echo "publish=true"' in branch, "the candidate branch must publish"


def test_the_build_workflow_derives_the_pre_release_from_the_tag():
    """A candidate tag is inert unless the build learns about it.

    `v2.10.0rc1` reaches the wheel only through FLAGOS_WHEEL_PRERELEASE, which
    build-wheel-common.yml sets from the tag before `python -m build`. Without
    it the wheels would say `2.10.0` and the tag/version check at upload would
    correctly refuse to publish them.
    """
    text = (REPO_ROOT / ".github" / "workflows" / "build-wheel-common.yml").read_text(
        encoding="utf-8"
    )
    assert "FLAGOS_WHEEL_PRERELEASE" in text
    assert "GITHUB_REF_TYPE" in text and "GITHUB_REF_NAME" in text
