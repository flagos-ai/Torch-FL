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
ordered list that all eight read, every entry in it runs on every platform, and
this module is the price of that: the composer is exercised through its real
entry point, and the shapes it refuses are refused here rather than on a vendor
runner.

What a platform still does not exercise is *not* in this file. It is the pytest
gate table in ``tests/integration/platform_gates.py``, whose entries name a
manifest step and the platforms where that step's target skips itself;
``tests/unit/test_platform_gates.py`` holds that table to the manifest and to
the eight platforms, and every platform's run prints it. A per-platform step
list here would be the second answer that table exists to remove.
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


def _entries() -> list[dict]:
    """The manifest's entries, in the order they run."""
    return manifest()["steps"]


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
    """A reader that found nothing would make every check below vacuous.

    The composed count is the manifest's own count on every platform, which is
    the property this layer exists for: the same groups, in the same order,
    with a platform contributing values rather than steps.
    """
    steps, ids = composer.validate_manifest(manifest())
    assert steps and ids
    assert len(set(ids)) == len(ids), "a step id is defined twice"
    for platform in _PLATFORMS:
        composed = composed_steps(platform)
        assert len(composed) == len(steps), (
            f"{platform} composes {len(composed)} group(s) from {len(steps)} "
            "manifest entries"
        )


def test_no_manifest_entry_restricts_itself_to_a_platform():
    """The key that made eight step lists out of one is gone.

    The composer refuses a `platforms:` key as an unknown one, which is the
    mechanical half of the rule; this is the half stated over the real
    manifest, where a reader can see which entry broke it.
    """
    for entry in _entries():
        assert "platforms" not in entry, (
            f"{entry['id']!r} restricts itself to some platforms; a platform "
            "that cannot exercise a step is recorded in "
            "tests/integration/platform_gates.py instead, where it is printed "
            "per platform and held to the test tree by a unit test"
        )


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_each_platform_runs_the_manifest_in_manifest_order(platform):
    """Order is the manifest's for every platform, and nothing is added or lost."""
    composed = composed_steps(platform)
    entries = _entries()
    assert [group["command"] for group in composed] == [
        entry["command"] for entry in entries
    ]
    display_name = platform_config(platform)["display_name"]
    assert [group["name"] for group in composed] == [
        entry["name"].replace("{{display_name}}", display_name) for entry in entries
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


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_a_configs_step_environment_names_steps_the_manifest_has(platform):
    """The values a platform adds to a step are keyed by an id that exists.

    `step_environment` is how a platform states the environment one step alone
    needs -- CUDA's FlagTree switch for the compile step, Ascend's process-start
    `LD_PRELOAD` for the profiler step. Keyed by id rather than by position, so
    a step that moves in the manifest does not silently carry someone else's
    environment, and a key naming no step is refused by the composer rather
    than dropped, because it would read as a step getting its environment while
    no such step runs.
    """
    declared = platform_config(platform).get("step_environment") or {}
    ids = set(composer.validate_manifest(manifest())[1])
    assert set(declared) <= ids, (
        f"{platform} states step_environment for {sorted(set(declared) - ids)}, "
        "which the manifest does not define"
    )


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_a_step_gets_the_shared_environment_plus_this_platforms_own(platform):
    """The composition rule, asserted group by group rather than sampled.

    A group's environment is the manifest entry's own value, plus whatever
    block the entry reads out of the config as JSON, plus the platform's
    `step_environment` for that id folded in last. Writing the whole mapping
    out is what catches the two ways this goes wrong: a platform's exception
    replacing the shared value instead of sitting beside it, and a variable
    stated for one step leaking onto a neighbour it did not name.
    """
    config = platform_config(platform)
    declared = config.get("step_environment") or {}
    entries = _entries()
    composed = composed_steps(platform)
    assert len(composed) == len(entries), platform
    for group, entry in zip(composed, entries):
        expected = {
            key: str(value) for key, value in (entry.get("environment") or {}).items()
        }
        for config_key, variable in (
            entry.get("environment_from_config") or {}
        ).items():
            expected[variable] = json.dumps(
                config[config_key], separators=(",", ":"), sort_keys=True
            )
        expected.update(
            {
                key: str(value)
                for key, value in (declared.get(entry["id"]) or {}).items()
            }
        )
        assert group.get("environment", {}) == expected, (platform, entry["id"])


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
        {"steps": [{"id": "a", "name": "N", "command": "c", "platforms": ["cuda"]}]},
        id="platforms-restricted-entry",
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
        composer.validate_manifest(document)


def test_a_step_id_defined_twice_is_refused():
    """An id used twice would run one step twice, silently.

    It is also the shape a platform-restricted variant of a step used to take:
    two entries for one id, each naming the platforms it applied to. Both
    readings are refused here, and the message says where a restriction belongs
    instead.
    """
    document = {
        "steps": [
            {"id": "a", "name": "A", "command": "x"},
            {"id": "a", "name": "A", "command": "x"},
        ]
    }
    with pytest.raises(composer.ManifestError, match="repeats"):
        composer.validate_manifest(document)


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


def test_a_step_environment_that_is_not_a_mapping_is_refused():
    """`step_environment` is a mapping of step id to environment, or nothing."""
    config = copy.deepcopy(platform_config("cuda"))
    config["step_environment"] = ["torch_compile"]
    with pytest.raises(composer.ManifestError, match="step_environment"):
        composer.compose(manifest(), config, "cuda", platform_names())


def test_a_step_environment_for_an_unknown_step_is_refused():
    """A key naming no step would read as a step getting its environment."""
    config = copy.deepcopy(platform_config("cuda"))
    config["step_environment"] = {"torch_compiel": {"FLAGOS_USE_FLAGTREE": "1"}}
    with pytest.raises(composer.ManifestError, match="not a step in the manifest"):
        composer.compose(manifest(), config, "cuda", platform_names())


def test_a_step_environment_value_that_is_not_scalar_is_refused():
    """The environment reaches the runner as strings, and only as strings."""
    config = copy.deepcopy(platform_config("cuda"))
    config["step_environment"] = {"torch_compile": {"FLAGOS_USE_FLAGTREE": ["1"]}}
    with pytest.raises(composer.ManifestError, match="must be a scalar"):
        composer.compose(manifest(), config, "cuda", platform_names())


def test_the_loader_invocation_composes_the_same_groups(tmp_path, capsys):
    """Exercise the loader's own invocation, not just the functions.

    `load-platform-config.yml` runs this script with `yq`-produced JSON and
    takes the stdout as the pipeline's groups; a test that only called
    `compose()` would not notice a broken CLI. The stderr it also produces is
    the coverage statement a reader of a run gets: one line saying every entry
    runs on every platform, then the pytest-level skips that remain, which are
    the only thing left that can differ.
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
    # The loader says how far the platforms still differ, once per platform.
    assert "cuda:" in captured.err
    assert "every entry on every platform" in captured.err
    for line in composer.gate_report("cuda"):
        assert line in captured.err, line


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
