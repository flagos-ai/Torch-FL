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

"""Hold the shared integration manifest to the shape the loader relies on.

The steps a platform runs used to be written in that platform's own config, so
"which platforms run this group" was a question you answered by reading eight
files and comparing them by hand -- and the answer drifted, because nothing
made two platforms run the same step in the same order. The manifest is one
ordered list that all eight read, and this module is the price of that: the
composer is exercised through its real entry point, and every deviation that
remains is a number in `_KNOWN_GAPS` rather than a difference nobody wrote
down.

The gaps are recorded rather than asserted away on purpose. This is the first
of three changes; the platforms still differ in coverage, and a test that
demanded uniformity today would fail until the last of them lands. What it
demands *now* is that the difference is a written decision: adding a platform
to a step's `platforms:` list has to be paired with editing `_KNOWN_GAPS`, and
that edit is where a reviewer sees how much is still per-platform.
"""

import copy
import json
import re
from pathlib import Path

import pytest

from manifest_composer import (
    composed_steps,
    composer,
    environment_check,
    manifest,
    platform_config,
    platform_names,
)


# The eight platforms this layer has to keep in step. Written out rather than
# taken from the directory: a config that quietly disappeared would otherwise
# take its platform out of every check below and leave the suite green.
_PLATFORMS = ("ascend", "cuda", "dcu", "gcu", "kunlunxin", "metax", "musa", "ppu")

_BOUND_KEYS = {"at_least", "more_than", "exactly"}
_FLAGTREE_KEYS = {"version_env", "version_default", "triton_prefix"}
# `{{...}}` in a name or a command. The composer substitutes exactly one.
_TOKEN_RE = re.compile(r"\{\{[^}]*\}\}")

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO / ".github/configs"

# The steps that do not run on every platform yet, as `platforms` -> the
# platforms missing from it. This is the ratchet: it has to match the
# composition exactly, so wiring a platform in or out is a visible edit here,
# and the check cannot quietly become a description of a pipeline nobody runs.
_KNOWN_GAPS = {
    "rng": ("ppu",),
    "soft_lowp_matrix": ("ascend", "cuda", "gcu", "kunlunxin", "musa", "ppu"),
    "amp_contract": ("kunlunxin", "ppu"),
    "profiler_contract": ("gcu", "kunlunxin"),
    "profiler_parity": ("ascend", "gcu", "kunlunxin", "metax", "musa", "ppu"),
    "torch_compile": ("ascend", "dcu", "musa", "ppu"),
    "inference": ("ascend", "dcu", "gcu", "kunlunxin", "metax", "musa"),
    "training": ("ascend", "dcu", "gcu", "kunlunxin", "metax", "musa"),
    "native_comm_bridge": (
        "ascend",
        "cuda",
        "gcu",
        "kunlunxin",
        "metax",
        "musa",
        "ppu",
    ),
    "musa_dispatch": ("ascend", "cuda", "dcu", "gcu", "kunlunxin", "metax", "ppu"),
    "musa_runtime": ("ascend", "cuda", "dcu", "gcu", "kunlunxin", "metax", "ppu"),
    "musa_flaggems_hybrid": (
        "ascend",
        "cuda",
        "dcu",
        "gcu",
        "kunlunxin",
        "metax",
        "ppu",
    ),
    "unit_tests_per_file": (
        "ascend",
        "cuda",
        "gcu",
        "kunlunxin",
        "metax",
        "musa",
        "ppu",
    ),
}


def _entries() -> list[dict]:
    """The manifest's entries, in the order they run."""
    return manifest()["steps"]


def _assigned() -> dict[str, set[str]]:
    """Each step id mapped to the platforms that run it, as composed."""
    return composer.validate_manifest(manifest(), platform_names())[2]


def _expected_entry(platform: str, entry: dict) -> bool:
    """Whether one manifest entry applies to a platform."""
    return platform in (entry.get("platforms") or _PLATFORMS)


def _environment_entry() -> dict:
    """The one manifest entry that reads a block out of the platform config."""
    entries = [entry for entry in _entries() if entry.get("environment_from_config")]
    assert len(entries) == 1, (
        "expected exactly one manifest entry to read a platform config block, "
        f"found {len(entries)}"
    )
    return entries[0]


def test_the_platform_list_is_the_config_directory():
    """The platform list is the directory, and every file in it is a platform.

    The composer reads the platforms from `.github/configs/*.yml` so that a
    manifest naming a platform no file defines fails instead of dropping that
    step everywhere. That reading is only sound while the directory holds
    nothing else: ci.yml's Discover platforms step and
    `.github/scripts/ci/generate_matrix.py` enumerate the same glob to decide
    which hardware pipelines to schedule, so a second kind of file in there is
    a platform to them.
    """
    assert platform_names() == list(_PLATFORMS)


def test_only_platform_configs_live_in_the_config_directory():
    """A non-platform `.yml` there is scheduled as a platform.

    This is the check the manifest's own first CI run needed and did not have:
    it lived at `.github/configs/integration-manifest.yml`, discovery read the
    directory, and a job named "Platform pipeline (integration-manifest)" was
    dispatched at hardware to load a manifest as if it were a config. The file
    moved to `.github/integration-manifest.yml`; this keeps the next artefact
    that wants to live beside the configs from repeating it.
    """
    stray = sorted(
        path.name
        for path in _CONFIG_DIR.iterdir()
        if path.suffix in {".yml", ".yaml"} and path.stem not in _PLATFORMS
    )
    assert stray == [], (
        f"{stray} would be discovered as platform name(s) and scheduled as "
        "hardware pipelines; platform configs are the only files this "
        "directory may hold"
    )


def test_the_manifest_composes_at_all():
    """A reader that found nothing would make every check below vacuous."""
    steps, seen_order, assigned = composer.validate_manifest(
        manifest(), platform_names()
    )
    assert steps and seen_order and assigned
    assert set(seen_order) == set(assigned)
    assert all(assigned.values()), "a step id was composed for no platform"


def test_the_manifest_still_records_which_steps_are_not_universal():
    """The deviation is a key in one file, and there is at least one of them.

    If this ever went to zero the composer's `platforms:` handling would stop
    being exercised, and the gaps check below would pass by having nothing to
    look at.
    """
    steps = _entries()
    restricted = [entry for entry in steps if entry.get("platforms")]
    assert restricted, "no manifest entry restricts itself to a platform"
    known = set(_PLATFORMS)
    for entry in restricted:
        assert set(entry["platforms"]) < known, entry
        assert len(set(entry["platforms"])) == len(entry["platforms"]), entry
    # Variants are separate entries for the same id, so ids outnumber groups.
    assert len(_assigned()) < len(steps)


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_each_platform_runs_the_manifest_in_manifest_order(platform):
    """Order is the manifest's for every platform, and nothing is added or lost."""
    composed = composed_steps(platform)
    expected = [entry for entry in _entries() if _expected_entry(platform, entry)]
    assert [group["command"] for group in composed] == [
        entry["command"] for entry in expected
    ]
    display_name = platform_config(platform)["display_name"]
    assert [group["name"] for group in composed] == [
        entry["name"].replace("{{display_name}}", display_name) for entry in expected
    ]


def test_the_only_substitution_is_the_display_name():
    """A second token would be a variable nothing expands, or a shell accident.

    The composer substitutes `{{display_name}}` in a name and passes every
    command through untouched, because a command's `${...}` belongs to the
    shell on the hardware runner.
    """
    for entry in _entries():
        assert set(_TOKEN_RE.findall(entry["name"])) <= {"{{display_name}}"}, entry
        assert not _TOKEN_RE.findall(entry["command"]), entry


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_the_environment_step_is_handed_this_platforms_expectations(platform):
    """One check per platform, and the data it asserts on is the config's.

    This is the seam that replaced eight hand-written heredocs: the platform
    states what its wheel and device must look like, the manifest step hands
    that block over as JSON, and the check asserts it. The value has to survive
    the round trip unchanged, or the platform is checked against a different
    environment than it describes.
    """
    entry = _environment_entry()
    ((config_key, variable),) = (entry["environment_from_config"]).items()
    config = platform_config(platform)
    assert config_key in config, (platform, config_key)

    steps = composed_steps(platform)
    groups = [group for group in steps if "environment" in group]
    matching = [group for group in groups if variable in group["environment"]]
    assert len(matching) == 1, [group["name"] for group in steps]
    assert steps.index(matching[0]) == 0, "the environment check is not run first"
    assert matching[0]["environment"][variable] == json.dumps(
        config[config_key], separators=(",", ":"), sort_keys=True
    )
    assert json.loads(matching[0]["environment"][variable]) == config[config_key]


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_every_expectation_key_is_one_the_check_script_reads(platform):
    """A misspelled key asserts nothing, so the vocabulary is closed.

    `check_platform_environment.py` refuses a key it does not read, and this
    holds the other end: a config written against a key the script has is a
    real expectation, and the shape of a bound or a FlagTree block is the one
    the script indexes into.
    """
    block = platform_config(platform)["environment_check"]
    assert block, f"{platform} states no environment expectations"
    unknown = sorted(set(block) - environment_check.SUPPORTED_KEYS)
    assert not unknown, (
        f"{platform} states {unknown}, which check_platform_environment.py does "
        "not read; add the key to SUPPORTED_KEYS and assert it"
    )
    for key in ("flagos_devices", "cuda_devices"):
        bound = block.get(key)
        if bound is not None:
            assert set(bound) <= _BOUND_KEYS, (platform, key, bound)
            assert len(bound) == 1, (platform, key, bound)
    if block.get("flagtree"):
        assert set(block["flagtree"]) == _FLAGTREE_KEYS, (platform, block["flagtree"])
    for key in ("require_package_files", "forbid_package_files", "require_imports"):
        if key in block:
            assert isinstance(block[key], list) and block[key], (platform, key)


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_no_platform_config_carries_a_step_list_of_its_own(platform):
    """The migration is complete, or the loader would refuse the config.

    `load-platform-config.yml` fails a config that still defines
    `integration_tests`, because the leftover list would read as if it were
    running while the pipeline ran the manifest instead. This is that refusal
    stated where it can run without a checkout and a runner.
    """
    assert "integration_tests" not in platform_config(platform)


def test_the_steps_that_do_not_run_everywhere_are_exactly_the_recorded_ones():
    """The remaining per-platform coverage, held to the list above.

    Read the failure as a to-do list: each entry is a step one platform runs
    and another does not, which this layer's follow-ups close one at a time.
    """
    assigned = _assigned()
    gaps = {
        step_id: tuple(p for p in _PLATFORMS if p not in platforms)
        for step_id, platforms in assigned.items()
        if len(platforms) != len(_PLATFORMS)
    }
    assert gaps == _KNOWN_GAPS


_REFUSED = [
    pytest.param([], id="not-a-mapping"),
    pytest.param({}, id="no-steps-key"),
    pytest.param({"steps": {}}, id="steps-not-a-list"),
    pytest.param({"steps": []}, id="empty-steps"),
    pytest.param({"steps": [None]}, id="step-not-a-mapping"),
    pytest.param({"steps": [{"name": "N", "command": "c"}]}, id="no-id"),
    pytest.param({"steps": [{"id": "", "name": "N", "command": "c"}]}, id="empty-id"),
    pytest.param({"steps": [{"id": "a", "command": "c"}]}, id="no-name"),
    pytest.param(
        {"steps": [{"id": "a", "name": " ", "command": "c"}]}, id="blank-name"
    ),
    pytest.param({"steps": [{"id": "a", "name": "N"}]}, id="no-command"),
    pytest.param(
        {"steps": [{"id": "a", "name": "N", "command": "c", "if": "x"}]},
        id="unknown-key",
    ),
    pytest.param(
        {"steps": [{"id": "a", "name": "N", "command": "c", "platforms": []}]},
        id="empty-platforms",
    ),
    pytest.param(
        {"steps": [{"id": "a", "name": "N", "command": "c", "platforms": "cuda"}]},
        id="platforms-not-a-list",
    ),
    pytest.param(
        {"steps": [{"id": "a", "name": "N", "command": "c", "platforms": ["nvidia"]}]},
        id="unknown-platform",
    ),
    pytest.param(
        {
            "steps": [
                {"id": "a", "name": "N", "command": "c", "platforms": ["cuda", "cuda"]}
            ]
        },
        id="repeated-platform",
    ),
    pytest.param(
        {"steps": [{"id": "a", "name": "N", "command": "c", "environment": "X"}]},
        id="environment-not-a-mapping",
    ),
    pytest.param(
        {
            "steps": [
                {"id": "a", "name": "N", "command": "c", "environment": {"A": ["x"]}}
            ]
        },
        id="environment-value-not-scalar",
    ),
    pytest.param(
        {
            "steps": [
                {"id": "a", "name": "N", "command": "c", "environment": {"A": True}}
            ]
        },
        id="environment-value-is-a-bool",
    ),
    pytest.param(
        {
            "steps": [
                {"id": "a", "name": "N", "command": "c", "environment_from_config": {}}
            ]
        },
        id="empty-environment_from_config",
    ),
    pytest.param(
        {
            "steps": [
                {
                    "id": "a",
                    "name": "N",
                    "command": "c",
                    "environment_from_config": {"k": ""},
                }
            ]
        },
        id="blank-environment_from_config-value",
    ),
]


@pytest.mark.parametrize("document", _REFUSED)
def test_a_manifest_that_cannot_be_composed_is_refused(document):
    """Every shape the composer cannot read fails loudly, and before a runner.

    The composer runs in the loader job, on ubuntu-latest, ahead of every
    hardware job: a manifest mistake caught here costs seconds, and the same
    mistake caught on a vendor runner costs a provisioning run and a queue.
    """
    with pytest.raises(composer.ManifestError):
        composer.validate_manifest(document, platform_names())


def test_a_step_whose_variants_are_not_adjacent_is_refused():
    """Variants of one step sit together, so merging them is a local edit."""
    document = {
        "steps": [
            {"id": "a", "name": "A", "command": "x", "platforms": ["cuda"]},
            {"id": "b", "name": "B", "command": "y"},
            {"id": "a", "name": "A", "command": "x", "platforms": ["dcu"]},
        ]
    }
    with pytest.raises(composer.ManifestError, match="not contiguous"):
        composer.validate_manifest(document, platform_names())


def test_a_step_that_claims_a_platform_twice_is_refused():
    """Two variants covering one platform would run the step twice, silently."""
    document = {
        "steps": [
            {"id": "a", "name": "A", "command": "x", "platforms": ["cuda"]},
            {"id": "a", "name": "A", "command": "x", "platforms": ["cuda", "dcu"]},
        ]
    }
    with pytest.raises(composer.ManifestError, match="would run it twice"):
        composer.validate_manifest(document, platform_names())


def test_an_unknown_platform_is_refused():
    with pytest.raises(composer.ManifestError, match="unknown platform"):
        composer.compose(
            manifest(), platform_config("cuda"), "nvidia", platform_names()
        )


def test_a_platform_config_that_is_not_a_mapping_is_refused():
    with pytest.raises(composer.ManifestError, match="must be a mapping"):
        composer.compose(manifest(), None, "cuda", platform_names())


def test_a_config_without_a_display_name_is_refused():
    """The display name reaches job and step names, so it cannot be blank."""
    config = platform_config("cuda")
    del config["display_name"]
    with pytest.raises(composer.ManifestError, match="display_name"):
        composer.compose(manifest(), config, "cuda", platform_names())


def test_a_config_that_omits_the_block_a_step_reads_is_refused():
    """A platform whose expectations never arrived must not pass the check."""
    config = platform_config("metax")
    config.pop("environment_check")
    with pytest.raises(composer.ManifestError, match="does not define it"):
        composer.compose(manifest(), config, "metax", platform_names())


def test_a_block_stated_as_a_non_mapping_is_refused():
    """The step is handed a mapping; a scalar would reach it as JSON text."""
    config = copy.deepcopy(platform_config("metax"))
    config["environment_check"] = "metax"
    with pytest.raises(composer.ManifestError, match="rather than the mapping"):
        composer.compose(manifest(), config, "metax", platform_names())


def test_a_config_whose_expectations_no_running_step_reads_is_refused():
    """The other direction: data nobody consumes is data nobody checks.

    Left alone, a platform can carry a whole block of expectations while the
    step that would assert them is restricted to some other platform.
    """
    document = manifest()
    for entry in document["steps"]:
        if entry.get("environment_from_config"):
            entry["platforms"] = ["cuda"]
    with pytest.raises(
        composer.ManifestError, match="no step that runs on this platform"
    ):
        composer.compose(document, platform_config("metax"), "metax", platform_names())


def test_a_manifest_that_leaves_a_platform_with_no_steps_is_refused():
    """An empty pipeline is a broken manifest, not a fast platform."""
    document = {
        "steps": [{"id": "a", "name": "A", "command": "x", "platforms": ["cuda"]}]
    }
    with pytest.raises(composer.ManifestError, match="no steps apply"):
        composer.compose(document, platform_config("dcu"), "dcu", platform_names())


def test_the_loader_invocation_composes_the_same_groups(tmp_path, capsys):
    """Exercise the loader's own invocation, not just the functions.

    `load-platform-config.yml` runs this script with `yq`-produced JSON and
    takes the stdout as the pipeline's groups; a test that only called
    `compose()` would not notice a broken CLI.
    """
    manifest_json = tmp_path / "manifest.json"
    config_json = tmp_path / "cuda.json"
    manifest_json.write_text(json.dumps(manifest()))
    config_json.write_text(json.dumps(platform_config("cuda")))

    assert (
        composer.main(
            [
                "--platform",
                "cuda",
                "--manifest",
                str(manifest_json),
                "--platform-config",
                str(config_json),
                "--config-dir",
                str(_CONFIG_DIR),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == composed_steps("cuda")
    # The loader prints how far the platforms still differ, once per platform.
    assert "cuda:" in captured.err
    assert "do not run on every platform yet" in captured.err


def test_the_loader_refuses_a_platform_with_no_config(tmp_path, capsys):
    manifest_json = tmp_path / "manifest.json"
    manifest_json.write_text(json.dumps(manifest()))
    (tmp_path / "nvidia.json").write_text(json.dumps(platform_config("cuda")))

    assert (
        composer.main(
            [
                "--platform",
                "nvidia",
                "--manifest",
                str(manifest_json),
                "--platform-config",
                str(tmp_path / "nvidia.json"),
                "--config-dir",
                str(_CONFIG_DIR),
            ]
        )
        == 1
    )
    assert "::error::" in capsys.readouterr().err
