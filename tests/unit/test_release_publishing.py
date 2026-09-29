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
`pypi_lane` column of cmake/flagos_platforms.json. Each platform uploads its own
wheel from the runner that built it, so the upload never crosses the border the
GitHub-hosted way did. The parts that can be wrong without a runner are checked
here: that the workflow covers every publishing platform (a missing job would
silently ship a release missing a vendor), that each of those jobs carries the
upload, that the lanes are one-per-platform, and that the version the tag names
is the version in each wheel.

Nothing here touches the network.
"""

import importlib.util
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

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


def _release_jobs(text: str) -> dict:
    """{job name: its YAML}, split on the two-space job keys under `jobs:`.

    Line-based rather than a YAML parse, for the same reason the checks below are
    text scans: tests/unit has no YAML parser it would otherwise need. A job key is
    the only thing indented exactly two spaces, so a job's body is every line
    indented further than that, and a two-space comment between jobs belongs to
    neither.
    """
    jobs: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        stripped = line.strip()
        key = stripped[:-1] if stripped.endswith(":") else ""
        is_key = (
            line.startswith("  ")
            and not line.startswith("   ")
            and bool(key)
            and key.replace("-", "").replace("_", "").isalnum()
        )
        if is_key:
            name = key
            jobs[name] = []
        elif name and line.startswith("   "):
            jobs[name].append(line)
    return {name: "\n".join(body) for name, body in jobs.items()}


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


def test_a_dispatch_must_come_from_the_tag_ref():
    """A dispatch from a branch builds a branch tip under the tag's version.

    The wheels are built from the run's own ref, and the pre-release suffix comes
    from that ref as well, so `v2.10.0rc1` dispatched from `main` produces
    `2.10.0+<sdk>` -- a version the tag does not name, built from whatever that
    branch pointed at. The upload's version check would then refuse all seven
    wheels, after seven builds. Requiring the tag ref rejects it in seconds.
    """
    text = RELEASE.read_text(encoding="utf-8")
    branch = text[
        text.index('if [ "$EVENT" != "push" ]') : text.index('echo "tag=${REF_NAME}"')
    ]
    assert 'if [ "$DISPATCH_TAG" != "$REF_NAME" ]' in branch, branch
    assert "::error::" in branch and "exit 1" in branch


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


# --- Uploading from the platform's own runner --------------------------------

# A wheel used to leave the runner that built it and be pushed to Nexus from one
# GitHub-hosted job, which put 2.35 GB on a cross-border link at ~0.13 MB/s and
# was killed by its own timeout on v2.10.0rc1. Each platform now uploads its own
# wheel in the job that built it, so what has to hold is that every build job
# carries the upload. A platform wired without it would produce a green release
# that is silently missing a wheel, which nothing else here would catch: naming
# the platform somewhere in the workflow (the test above) is satisfied either way.


def test_every_release_build_job_carries_the_upload():
    jobs = _release_jobs(RELEASE.read_text(encoding="utf-8"))
    builds = {
        name: body
        for name, body in jobs.items()
        if "build-wheel-" in body or "all-tests-common.yml" in body
    }
    assert len(builds) == len(publishing_platforms()), sorted(builds)
    for name, body in builds.items():
        assert "publish: true" in body, f"{name} builds a wheel but does not publish it"
        assert "NEXUS_TOKEN: ${{ secrets.NEXUS_TOKEN }}" in body, name


def test_ci_still_builds_without_publishing():
    """The release flag must be opt-in, or every push would try to upload.

    ci.yml calls the same all-tests-common.yml without it and passes no secrets,
    which only works because the input defaults to false and the token is not
    required. Flipping either would fail every push at run creation.
    """
    common = (REPO_ROOT / ".github" / "workflows" / "all-tests-common.yml").read_text(
        encoding="utf-8"
    )
    interfaces = common[common.index("on:") : common.index("\njobs:")]
    publish_at = interfaces.index("publish:")
    assert "type: boolean" in interfaces[publish_at : publish_at + 120]
    assert "default: false" in interfaces[publish_at : publish_at + 120]

    secrets_at = interfaces.index("    secrets:")
    assert "NEXUS_TOKEN:" in interfaces[secrets_at:]
    assert "required: true" not in interfaces[secrets_at:], (
        "an optional secret must not become required: ci.yml passes none"
    )

    assert "publish" not in (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    ), "ci.yml must not ask for a release upload"


def test_the_publish_step_runs_before_the_artifact_upload():
    """The upload crosses the border; publishing must not wait behind it.

    Measured on the 2.10.0 release, per platform: publishing takes 18-32 s (the
    vendor lane is in China, like the runner), while the artifact step is 2m25s
    for 476 MB, 3m15s for 316 MB, 9m43s for 672 MB, and on DCU's 889 MB wheel it
    failed outright after 21 minutes. With the artifact step first, that failure
    skipped the publish step and DCU shipped nothing -- twice. So the order is
    load-bearing, not cosmetic: publish first, artifact after.

    The artifact still has to be produced when publishing fails, because it is
    the documented way to re-upload one lane without rebuilding, hence
    `always()`.
    """
    text = (REPO_ROOT / ".github" / "workflows" / "build-wheel-common.yml").read_text(
        encoding="utf-8"
    )
    build_at = text.index("python -m build --wheel --no-isolation")
    verify_at = text.index("python -m pip install --no-deps dist/*.whl")
    publish_at = text.index("name: Publish the wheel to its vendor lane")
    artifact_at = text.index("name: Upload wheel")
    assert build_at < verify_at < publish_at < artifact_at, (
        build_at,
        verify_at,
        publish_at,
        artifact_at,
    )

    step = text[publish_at:artifact_at]
    # Only release.yml sets it, so a normal CI build of this same workflow is
    # untouched.
    assert "if: inputs.publish" in step[:200], step[:200]
    # The upload decides its own route to Nexus, which is not the same on every
    # runner: the PPU pod's proxy refuses the CONNECT to resource.flagos.net.
    assert "lib/proxy_route.sh" in step
    assert "prefer_direct_route" in step
    assert "publish_wheels.py" in step
    # A release is a tag; a manual dispatch of this workflow must not publish a
    # branch tip under a release version.
    assert "GITHUB_REF_TYPE" in step and '!= "tag"' in step

    artifact = text[artifact_at : artifact_at + 220]
    assert "if: always()" in artifact, artifact


# --- The upload itself -------------------------------------------------------

# Serial uploads cost the whole release when one transfer stalls, which is what
# happened to v2.10.0rc1: the 476 MB CUDA wheel hung behind a publisher that
# uploads one at a time, and the other six never started.


def _pairs(tmp_path):
    """A downloaded-artifact tree, one wheel per publishing platform."""
    pairs = []
    for platform in publishing_platforms():
        directory = tmp_path / f"wheel-{platform}"
        directory.mkdir(parents=True, exist_ok=True)
        wheel = directory / "torch_fl-2.10.0rc1+x-cp312-cp312-linux_x86_64.whl"
        wheel.write_bytes(b"x" * 1024)
        pairs.append((platform, wheel))
    return pairs


def test_every_platform_is_uploaded_with_a_timeout(tmp_path, monkeypatch):
    """One bad lane must not be able to consume the publish window."""
    publisher = _publisher()
    monkeypatch.setattr(publisher, "already_published", lambda *_a, **_k: False)
    timeouts = []

    def fake_run(args, **kwargs):
        timeouts.append(kwargs.get("timeout"))

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(publisher.time, "sleep", lambda _s: None)

    pairs = _pairs(tmp_path)
    lanes = publishing_platforms()
    outcomes = {}
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        futures = [
            pool.submit(
                publisher.publish_one,
                platform,
                lanes[platform],
                wheel,
                "http://example.invalid/simple",
                "user:token",
                SimpleNamespace(dry_run=False, upload_timeout=42),
            )
            for platform, wheel in pairs
        ]
        for future in as_completed(futures):
            platform, outcome = future.result()
            outcomes[platform] = outcome

    assert set(outcomes) == set(lanes)
    assert set(outcomes.values()) == {"uploaded"}
    assert timeouts == [42] * len(pairs)


def test_a_stalled_upload_is_reported_as_a_timeout(tmp_path, monkeypatch):
    """The thing that actually went wrong, named rather than left as a hang."""
    publisher = _publisher()
    monkeypatch.setattr(publisher, "already_published", lambda *_a, **_k: False)

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    wheel = tmp_path / "torch_fl-2.10.0rc1+x-cp312-cp312-linux_x86_64.whl"
    wheel.write_bytes(b"x" * 1024)
    _, outcome = publisher.publish_one(
        "dcu",
        "hygon",
        wheel,
        "http://example.invalid/simple",
        "user:token",
        SimpleNamespace(dry_run=False, upload_timeout=42),
    )
    assert outcome == "timed out"


def test_a_failed_upload_is_reported_and_not_retried_forever(tmp_path, monkeypatch):
    publisher = _publisher()
    monkeypatch.setattr(publisher, "already_published", lambda *_a, **_k: False)
    attempts = []

    def fake_run(args, **kwargs):
        attempts.append(1)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(publisher.time, "sleep", lambda _s: None)

    wheel = tmp_path / "torch_fl-2.10.0rc1+x-cp312-cp312-linux_x86_64.whl"
    wheel.write_bytes(b"x" * 1024)
    _, outcome = publisher.publish_one(
        "cuda",
        "nvidia",
        wheel,
        "http://example.invalid/simple",
        "user:token",
        SimpleNamespace(dry_run=False, upload_timeout=42),
    )
    assert outcome == "failed"
    # Two attempts: a transient refusal is worth one retry, a permanent one
    # is not worth more.
    assert len(attempts) == 2


def test_a_transient_failure_is_retried(tmp_path, monkeypatch):
    """The egress proxy answers an intermittent 503; one must not fail a release."""
    publisher = _publisher()
    attempts = []

    def fake_run(args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(publisher.time, "sleep", lambda _s: None)

    wheel = tmp_path / "torch_fl-2.10.0rc1+x-cp312-cp312-linux_x86_64.whl"
    wheel.write_bytes(b"x" * 1024)
    publisher.upload(wheel, "http://example.invalid/simple", "user:token", 42)
    assert len(attempts) == 2


def test_an_index_refusal_is_reported_as_a_lane_failure(tmp_path, monkeypatch):
    """The idempotency check goes out over the same link the upload does.

    This host answers an intermittent 503, and `already_published` re-raises any
    non-404 status. That used to escape as a traceback and abort the job, instead
    of naming the lane that needs re-running -- which is the whole point of the
    skip check existing.
    """
    publisher = _publisher()

    def refuse(*_args, **_kwargs):
        raise publisher.urllib.error.HTTPError(
            "https://x/simple/torch-fl/", 503, "Service Unavailable", None, None
        )

    monkeypatch.setattr(publisher, "already_published", refuse)

    wheel = tmp_path / "torch_fl-2.10.0rc1+x-cp312-cp312-linux_x86_64.whl"
    wheel.write_bytes(b"x" * 1024)
    _, outcome = publisher.publish_one(
        "cuda",
        "nvidia",
        wheel,
        "http://example.invalid/simple",
        "user:token",
        SimpleNamespace(dry_run=False, upload_timeout=42),
    )
    assert outcome == "failed"


def test_main_runs_the_platforms_concurrently(tmp_path, monkeypatch):
    """The test above proves the helper can overlap; this proves main asks it to.

    Without this, `workers = 1` in main would pass every other test here, and
    the release would go back to one upload at a time.
    """
    publisher = _publisher()
    _pairs(tmp_path)
    live = []
    peak = []

    def fake_publish_one(platform, lane, wheel, index_url, token, args):
        live.append(1)
        peak.append(len(live))
        time.sleep(0.05)
        live.pop()
        return platform, "uploaded"

    monkeypatch.setattr(publisher, "publish_one", fake_publish_one)
    monkeypatch.setenv("NEXUS_TOKEN", "user:token")

    rc = publisher.main(
        ["--wheels", str(tmp_path), "--tag", "v2.10.0rc1", "--upload-timeout", "1"]
    )

    assert rc == 0
    assert max(peak) == len(publishing_platforms()), max(peak)


# --- The idempotency check ---------------------------------------------------

# Both of these were wrong in a way a fixture would not have caught, and both
# only show up against the real index:
#
#   * `/simple/torch_fl/` is a 404 and `/simple/torch-fl/` is the page -- Nexus
#     normalises the name and does not redirect. Asking with the underscore
#     form made the check return False for `torch_fl` itself, so the skip that
#     makes a re-run safe never fired.
#   * the version is followed by `+` when the wheel has a local segment
#     (`torch_fl-2.10.0rc1+cann9.0.0-cp311-...`) and `-` when it does not, so
#     matching only `-` reported "not published" for every platform with a
#     segment -- which is all of them.


class _FakeResponse:
    def __init__(self, page):
        self._page = page

    def read(self):
        return self._page.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _serve(monkeypatch, page, seen=None):
    publisher = _publisher()

    def fake_urlopen(url, timeout=None):
        if seen is not None:
            seen.append(url)
        return _FakeResponse(page)

    monkeypatch.setattr(publisher.urllib.request, "urlopen", fake_urlopen)
    return publisher


def test_the_index_is_asked_for_the_normalised_name(monkeypatch):
    seen = []
    publisher = _serve(monkeypatch, "no wheels here", seen)

    publisher.already_published(
        "torch_fl", "2.10.0rc1", "https://resource.flagos.net/r/simple"
    )

    assert seen == ["https://resource.flagos.net/r/simple/torch-fl/"], seen


def test_a_local_segment_counts_as_published(monkeypatch):
    """The form every platform produces now."""
    _serve(monkeypatch, "torch_fl-2.10.0rc1+cann9.0.0-cp311-cp311-linux_aarch64.whl")
    publisher = _publisher()
    assert publisher.already_published("torch_fl", "2.10.0rc1", "https://x/simple")


def test_a_wheel_without_a_segment_counts_as_published(monkeypatch):
    """The form tsingmicro and bpu would produce."""
    _serve(monkeypatch, "torch_fl-2.10.0-cp312-cp312-linux_x86_64.whl")
    publisher = _publisher()
    assert publisher.already_published("torch_fl", "2.10.0", "https://x/simple")


def test_a_different_version_is_not_published(monkeypatch):
    _serve(monkeypatch, "torch_fl-2.10.0+cuda13.3-cp312-cp312-linux_x86_64.whl")
    publisher = _publisher()
    assert not publisher.already_published("torch_fl", "2.10.0rc1", "https://x/simple")
    assert not publisher.already_published("torch_fl", "2.10.0rc2", "https://x/simple")
