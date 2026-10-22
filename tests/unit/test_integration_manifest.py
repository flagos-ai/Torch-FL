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
"which platforms run this group" was a question you answered by reading ten
files and comparing them by hand -- and the answer drifted, because nothing
made two platforms run the same step in the same order. The manifest is one
ordered list that all ten read, and this module is the price of that: the
composer is exercised through its real entry point, and every deviation that
remains is a number in `_KNOWN_GAPS` rather than a difference nobody wrote
down.

The gaps are recorded rather than asserted away on purpose. Platform coverage
still differs for groups whose backend capability does not exist yet, so adding
a platform to a step's `platforms:` list has to be paired with editing
`_KNOWN_GAPS`. Model groups have no such gap: all ten platforms run the same
six Qwen3 groups through one path and checkpoint.
"""

import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from manifest_composer import (
    composed_steps,
    composer,
    environment_check,
    load_module,
    manifest,
    platform_config,
    platform_names,
)


# The platforms this layer has to keep in step. Written out rather than taken
# from the directory: a config that quietly disappeared would otherwise take its
# platform out of every check below and leave the suite green.
_PLATFORMS = (
    "ascend",
    "cuda",
    "dcu",
    "gcu",
    "iluvatar",
    "kunlunxin",
    "metax",
    "mps",
    "musa",
    "ppu",
)

_BOUND_KEYS = {"at_least", "more_than", "exactly"}
_FLAGTREE_KEYS = {"version_env", "version_default", "triton_prefix"}
# `{{...}}` in a name or a command. The composer substitutes exactly one.
_TOKEN_RE = re.compile(r"\{\{[^}]*\}\}")

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO / ".github/configs"
_LOADER = _REPO / ".github/workflows/load-platform-config.yml"
_HARDWARE = _REPO / ".github/workflows/integration-tests.yml"


RUNNER_VALIDATION = "--validate-only"
_HELPER = _REPO / ".github/scripts/integration_inventory_plan.py"


# The steps that do not run on every platform yet, as `platforms` -> the
# platforms missing from it. This is the ratchet: it has to match the
# composition exactly, so wiring a platform in or out is a visible edit here,
# and the check cannot quietly become a description of a pipeline nobody runs.
_KNOWN_GAPS = {
    "iluvatar_runtime": (
        "ascend",
        "cuda",
        "dcu",
        "gcu",
        "kunlunxin",
        "metax",
        "mps",
        "musa",
        "ppu",
    ),
    "rng": ("mps", "ppu"),
    "soft_lowp_matrix": (
        "ascend",
        "cuda",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "mps",
        "musa",
        "ppu",
    ),
    "amp_contract": ("kunlunxin", "mps", "ppu"),
    "profiler_contract": ("mps",),
    "profiler_parity": (
        "ascend",
        "gcu",
        "kunlunxin",
        "metax",
        "mps",
        "musa",
        "ppu",
    ),
    "torch_compile": ("ascend", "dcu", "mps", "musa", "ppu"),
    "native_comm_bridge": (
        "ascend",
        "cuda",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "metax",
        "mps",
        "musa",
        "ppu",
    ),
    "musa_dispatch": (
        "ascend",
        "cuda",
        "dcu",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "metax",
        "mps",
        "ppu",
    ),
    "musa_runtime": (
        "ascend",
        "cuda",
        "dcu",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "metax",
        "mps",
        "ppu",
    ),
    "vendor_ops": ("mps",),
    "backend_ops": (
        "ascend",
        "cuda",
        "dcu",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "metax",
        "musa",
        "ppu",
    ),
    "mps_runtime": (
        "ascend",
        "cuda",
        "dcu",
        "gcu",
        "iluvatar",
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
    """Whether one manifest entry runs, by the composer's requirement rule."""
    if platform not in (entry.get("platforms") or _PLATFORMS):
        return False
    config = platform_config(platform)
    return composer._applicable(entry, platform, config, platform_names())


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
def test_every_platform_has_the_same_model_groups_and_environment(platform):
    """A model case runs everywhere and reads one published checkpoint."""
    config = platform_config(platform)
    groups = {step["id"]: step for step in composed_steps(platform)}
    model_groups = (
        "model",
        "fallback_trace",
        "inference",
        "training",
        "fallback_trace_train",
        "qwen3_profiler",
    )
    assert set(model_groups) <= groups.keys()
    for group in model_groups:
        assert groups[group]["environment"]["MODEL_PATH"] == "/models/Qwen3-0.6B"

    if config["ci_image"]:
        assert "/var/lib/torch-fl-ci/models:/models:ro" in config["container_volumes"]
    else:
        assert config["container_volumes"] == []


def test_every_platform_runs_the_same_model_cases():
    """The same model files and arguments run on every hardware platform."""
    model_groups = {
        "model",
        "fallback_trace",
        "inference",
        "training",
        "fallback_trace_train",
        "qwen3_profiler",
    }
    expected = [
        step for step in composed_steps(_PLATFORMS[0]) if step["id"] in model_groups
    ]
    assert len(expected) == len(model_groups)
    for platform in _PLATFORMS[1:]:
        actual = [
            step for step in composed_steps(platform) if step["id"] in model_groups
        ]
        assert actual == expected, platform


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
    bound = block["flagos_devices"]
    count = bound.get("at_least", 0)
    if "more_than" in bound:
        count = bound["more_than"] + 1
    if "exactly" in bound:
        count = bound["exactly"]
    assert count >= 1, f"{platform} must expose a flagos device, got {bound}"
    if block.get("flagtree"):
        assert set(block["flagtree"]) == _FLAGTREE_KEYS, (platform, block["flagtree"])
    for key in ("require_package_files", "forbid_package_files", "require_imports"):
        if key in block:
            assert isinstance(block[key], list) and block[key], (platform, key)


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_every_flaggems_build_asserts_the_native_compiler(platform):
    """A FlagGems lane tests the Triton variant provisioned for its chip."""
    config = platform_config(platform)
    block = config["environment_check"]
    flag_gems = "flaggems" in block.get("require_build_kernels", [])
    if not flag_gems:
        assert not block.get("flagtree"), platform
        return
    flagtree = block.get("flagtree")
    assert flagtree, f"{platform} installs FlagTree without asserting it"
    pin = environment_check._flagtree_version_from_pin_table(platform)
    if pin is None:
        hook = (_REPO / ".github/scripts/hooks" / f"set_env_{platform}.sh").read_text()
        fallback = f"TORCH_FL_FLAGTREE_VERSION:-{flagtree['version_default']}"
        assert fallback in hook, (platform, flagtree["version_default"])
    else:
        assert flagtree["version_default"] == pin, (platform, pin)


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_every_platform_runs_the_shared_flaggems_conformance_steps(platform):
    """Chip support differs; the flow and its case sets do not.

    Both passes are generated for every platform and contain the same cases.
    The kernel pass is removed only when the platform's build forbids the
    FlagGems set. Every platform that can build FlagGems must run it.
    """
    steps = composed_steps(platform)
    matching = [
        step
        for step in steps
        if step["id"] in {"flaggems_conformance", "flaggems_kernel_conformance"}
    ]
    expected = [
        ("flaggems_conformance", "python tests/integration/flaggems/run.py native"),
        (
            "flaggems_kernel_conformance",
            "python tests/integration/flaggems/run.py flaggems",
        ),
    ]
    block = platform_config(platform)["environment_check"]
    if "flaggems" in block.get("forbid_build_kernels", []):
        assert "flaggems" not in block.get("require_build_kernels", []), platform
        expected = expected[:1]
    else:
        assert block.get("require_build_kernels") == ["flaggems"], platform
    assert [(step["id"], step["command"]) for step in matching] == expected, platform


def test_mps_is_the_only_platform_that_forbids_flaggems_kernels():
    """The MPS build has no FlagGems kernels; every other platform requires them."""
    forbidden = {
        platform
        for platform in _PLATFORMS
        if "flaggems"
        in platform_config(platform)["environment_check"].get(
            "forbid_build_kernels", []
        )
    }
    assert forbidden == {"mps"}


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_platform_build_switches_agree_with_the_cmake_kernel_defaults(platform):
    """The environment check and the build state the same kernel sets."""
    defaults = json.loads((_REPO / "cmake/flagos_platforms.json").read_text())[
        "accelerators"
    ][platform]["kernel_defaults"]
    expected = ["flaggems"]
    block = platform_config(platform)["environment_check"]
    if defaults["FLAGOS_BUILD_FLAGGEMS"]:
        assert block.get("require_build_kernels") == expected, platform
        assert "flaggems" not in block.get("forbid_build_kernels", []), platform
    else:
        assert block.get("forbid_build_kernels") == expected, platform
        assert "flaggems" not in block.get("require_build_kernels", []), platform


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
    """Only the recorded backend gaps differ between platform plans."""
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


def _expected_bindings(platform):
    """The inventory bindings the composed plan implies for one platform."""
    entries = {entry["id"]: entry for entry in _entries()}
    return [
        (group, kind)
        for group, kind in composer.INVENTORY_GROUPS.items()
        if group in {step["id"] for step in composed_steps(platform)}
        and all(
            name
            not in platform_config(platform)["environment_check"].get(
                "forbid_build_kernels", []
            )
            for name in entries[group].get("requires_build_kernels", [])
        )
    ]


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_every_composed_plan_uses_the_same_shared_inventories(platform):
    """Every shared inventory a plan runs binds the one kind it was written for.

    A restricted step (the FlagGems kernel pass) drops out of a plan whose
    build forbids its kernel set; every other shared inventory must remain.
    """
    steps = composed_steps(platform)
    identifiers = [step["id"] for step in steps]
    expected = _expected_bindings(platform)
    for group, kind in expected:
        assert group in identifiers, platform
    assert composer.inventory_bindings(steps) == expected, platform


def test_the_loader_hands_the_runner_the_composers_inventory_plan(tmp_path):
    """The loader's runner arguments come from the composer's plan.

    Run the real composer and helper CLIs: the loader must not carry a second
    copy of the group-to-kind table, and the arguments it emits must resolve
    to the runner's own bindings.
    """
    notes = tmp_path / "composer.notes"
    notes.write_text(_compose_notes(tmp_path, "cuda"))
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO / ".github/scripts/integration_inventory_plan.py"),
            "--composer-output",
            str(notes),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outputs = _read_github_outputs(result.stdout)
    expected = _expected_bindings("cuda")
    assert json.loads(outputs["integration_inventories"]) == [
        group for group, _ in expected
    ]
    assert outputs["test_inventory_kinds"] == " ".join(kind for _, kind in expected)
    assert set(json.loads(outputs["test_inventories"])) == {
        kind for _, kind in expected
    }
    assert outputs["integration_inventory_arguments"] == " ".join(
        f"--inventory {group}={kind}" for group, kind in expected
    )
    assert outputs["loader_inventory_arguments"] == " ".join(
        f"--inventory {group}={composer.INVENTORY_SOURCES[group]}"
        for group, _ in expected
    )

    folded = " ".join(_LOADER.read_text(encoding="utf-8").split())
    assert "integration_inventory_plan.py" in folded
    assert "integration_inventory_arguments" in folded
    assert "run_integration_tests.py --validate-only" in folded
    for group in composer.INVENTORY_GROUPS:
        assert f'--inventory "{group}="' not in folded


def test_the_composer_uses_the_hardware_runners_inventory_tables():
    """A plan label and the report verifier have one source for each group."""
    source = (_REPO / ".github/scripts/run_integration_tests.py").read_text()
    harness = (_REPO / ".github/scripts/compose_integration_manifest.py").read_text()
    assert "INVENTORY_GROUPS = {" in source
    assert "INVENTORY_SOURCES = {" in source
    assert source.count("INVENTORY_GROUPS = {") == 1
    assert source.count("INVENTORY_SOURCES = {") == 1
    assert "INVENTORY_GROUPS = _RUNNER.INVENTORY_GROUPS" in harness
    assert "INVENTORY_SOURCES = _RUNNER.INVENTORY_SOURCES" in harness


def _compose_notes(directory, platform):
    """Run the composer CLI and return its machine-readable notes."""
    manifest_json = directory / "manifest.json"
    config_json = directory / "config.json"
    manifest_json.write_text(json.dumps(manifest()))
    config_json.write_text(json.dumps(platform_config(platform)))
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO / ".github/scripts/compose_integration_manifest.py"),
            "--platform",
            platform,
            "--manifest",
            str(manifest_json),
            "--platform-config",
            str(config_json),
            "--config-dir",
            str(_CONFIG_DIR),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_the_inventory_helper_rejects_an_incomplete_or_inconsistent_plan(
    tmp_path,
):
    """A plan must name each kind once and only the kinds the runner knows."""
    notes = _compose_notes(tmp_path, "cuda").splitlines()
    plan = next(line for line in notes if line.startswith("inventory-plan="))
    other = "\n".join(line for line in notes if line != plan)
    entries = json.loads(plan.split("=", 1)[1])
    unknown = [*entries, {**entries[0], "kind": "unknown", "group": "unknown"}]
    repeated = [entries[0], *entries]
    for body, message in (
        ("", "exactly one inventory-plan"),
        (plan + "\n" + plan, "exactly one inventory-plan"),
        ("inventory-plan=" + json.dumps(unknown), "expected"),
        ("inventory-plan=" + json.dumps(repeated), "repeats group"),
    ):
        path = tmp_path / (message.replace(" ", "-") + ".notes")
        path.write_text(other + "\n" + body + "\n")
        result = subprocess.run(
            [sys.executable, str(_HELPER), "--composer-output", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert message in result.stderr, result.stderr


def _read_github_outputs(source):
    """Parse GitHub heredoc outputs from one helper invocation."""
    outputs = {}
    current = None
    lines = []
    for line in source.splitlines():
        if current is None and "=" in line:
            key, value = line.split("=", 1)
            outputs[key] = value
            continue
        if current is None and "<<" in line:
            current = line.split("<<", 1)[0]
            continue
        if current is not None and line == f"__{current}__":
            outputs[current] = "\n".join(lines)
            current = None
            lines = []
        elif current is not None:
            lines.append(line)
    assert current is None, f"unterminated output {current}"
    return outputs


def test_the_configuration_step_writes_every_declared_output(tmp_path):
    """Every declared output survives the loader's own output writers.

    The scalar and heredoc writers are separate. A missing scalar writer used
    to fail every configuration job with exit 127 only after all of the
    inventories had been validated. With yq installed, execute the complete
    step; otherwise execute its post-validation output block. The CI pin is
    the version exercised here because yq operators vary across releases.
    """
    yq = shutil.which("yq")

    document = yaml.safe_load(_LOADER.read_text(encoding="utf-8"))
    job = document["jobs"]["configuration"]
    step = next(entry for entry in job["steps"] if entry.get("id") == "config")
    output = tmp_path / "github-output"
    env = {
        "PATH": os.environ["PATH"],
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(output),
    }
    if yq is None:
        # Run the actual writer definitions and their calls against fixture
        # values. The inventory block is produced by the same helper the step
        # consumes, so every declared output must still be present.
        tail = "emit_list() {" + step["run"].split("emit_list() {", 1)[1]
        inputs = {
            "DISPLAY_NAME": "CUDA",
            "SETUP_SCRIPT": ".github/scripts/set_env.sh",
            "SETUP_PLATFORM": "cuda",
            "SETUP_STAGE": "integration",
            "CI_IMAGE": "image",
            "RUNS_ON": "[]",
            "VOLUMES": "[]",
            "OPTIONS": "options",
            "TIMEOUT_MINUTES": "60",
            "INTEGRATION_TIMEOUT": "90",
            "ARTIFACT_NAME": "wheel-cuda",
            "INTEGRATION_TESTS": "[]",
            "INTEGRATION_ENVIRONMENT": "{}",
            "BUILD_TOOLS": "",
            "TEST_DEPENDENCIES": "",
            "FLAG_GEMS_REQUIREMENT": "required",
            "TORCH_FL_PACKAGE_ROOT": str(tmp_path / "site-packages"),
            "INVENTORY_OUTPUTS": str(tmp_path / "inventory-outputs"),
        }
        script = "\n".join(
            f"{name}={shlex.quote(value)}" for name, value in inputs.items()
        )
        script += "\nset -euo pipefail\n" + tail
        (tmp_path / "inventory-outputs").write_text(
            "\n".join(
                f"{name}<<__{name}__\n{value}\n__{name}__"
                for name, value in (
                    ("test_inventories", "{}"),
                    ("test_inventory_kinds", ""),
                    ("integration_inventories", "[]"),
                    ("loader_inventory_arguments", ""),
                    ("integration_inventory_arguments", ""),
                )
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        # Use the local yq in place of the Linux download; the rest of the step
        # still runs as written in CI.
        script = step["run"].replace(
            "sudo wget -qO /usr/local/bin/yq \\\n"
            "  https://github.com/mikefarah/yq/releases/download/"
            "v4.45.1/yq_linux_amd64\n"
            "sudo chmod +x /usr/local/bin/yq",
            f"export PATH={shlex.quote(str(Path(yq).parent))}:$PATH",
        )
        assert "yq_linux_amd64" not in script
        python = tmp_path / "python"
        python.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
            encoding="utf-8",
        )
        python.chmod(0o755)
        env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
        env["GITHUB_WORKSPACE"] = str(_REPO)
        env["PLATFORM"] = "cuda"

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        cwd=_REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr, script[-800:])
    values = _read_github_outputs(output.read_text(encoding="utf-8"))
    declared = {
        match.group(1)
        for value in job["outputs"].values()
        if (match := re.fullmatch(r"\$\{\{ steps\.config\.outputs\.(.+) \}\}", value))
    }
    assert set(values) == declared
    assert values["display_name"] == "CUDA"
    assert values["setup_stage"] == "integration"
    assert values["timeout_minutes"] == "60"
    if yq is None:
        assert values["integration_timeout_minutes"] == "90"
    else:
        assert json.loads(values["integration_tests"]) == composed_steps("cuda")
        assert values["flag_gems_requirement"] == "required"

        # yq 4.45.1 is installed by the loader in CI. It has no general `if`
        # expression, so run the same complete step for each platform that
        # explicitly forbids FlagGems and verify its derived output.
        forbidden = [
            platform
            for platform in _PLATFORMS
            if "flaggems"
            in platform_config(platform)["environment_check"].get(
                "forbid_build_kernels", []
            )
        ]
        assert forbidden
        for platform in forbidden:
            output.unlink(missing_ok=True)
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
                cwd=_REPO,
                env={**env, "PLATFORM": platform},
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, (
                platform,
                result.stdout,
                result.stderr,
            )
            platform_values = _read_github_outputs(output.read_text(encoding="utf-8"))
            assert platform_values["flag_gems_requirement"] == "forbidden"


def test_the_hardware_job_materializes_and_retains_each_inventory():
    """The job writes the kinds its plan runs and retains their directory."""
    source = _HARDWARE.read_text(encoding="utf-8")
    materialize = source.split("- name: Materialize shared test inventories", 1)[1]
    run = materialize.split("- name:", 1)[0]
    assert "SHARED_TEST_INVENTORY_KINDS" in run
    assert "set(inventories) != required" in run
    assert 'root="$INTEGRATION_INVENTORY_ROOT"' in run
    assert "tests/$kind/google-test.json" not in run.replace(" ", "")

    folded = " ".join(source.split())
    assert "integration_inventory_arguments" in folded
    assert '"${inventory_arguments[@]}"' in folded
    retained = source.split("- name: Retain shared test inventories", 1)
    assert len(retained) == 2, "the hardware job does not retain the inventories"
    assert "${{ env.PIP_CACHE_DIR }}/../torch-fl-inventories/" in retained[1]
    assert "shared-test-inventory-" not in source


def test_checkout_per_file_reports_satisfy_one_shared_unit_inventory(
    monkeypatch, tmp_path
):
    """A hardware report names a checkout file while the loader names its case."""
    checkout = tmp_path / "checkout"
    test_file = checkout / "tests/unit/test_one.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_one():\n    assert True\n")
    reports = tmp_path / "reports"
    case = "tests/unit/test_one.py"
    inventory = {"cases": [case], "aliases": {}}
    reports.mkdir()
    (reports / "google-test.json").write_text(json.dumps(inventory))
    (reports / "unit-one.json").write_text(
        json.dumps(
            {
                "args": [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_file),
                    "-q",
                ],
                "tests": [
                    {
                        "nodeid": case + "::test_one",
                        "outcome": "passed",
                        "setup": {"outcome": "passed"},
                        "call": {"outcome": "passed"},
                        "teardown": {"outcome": "passed"},
                    }
                ],
                "summary": {"passed": 1, "failed": 0, "error": 0},
            }
        )
    )
    (reports / "unit-one.unit").write_text(json.dumps({"args": [str(test_file)]}))
    (reports / "unit-one.pending").write_text(json.dumps({"group": "unit"}))
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))
    runner = load_module(
        _REPO / ".github/scripts/run_integration_tests.py",
        "unit_inventory_runner",
    )
    runner.verify_inventory(
        "unit_tests_per_file",
        runner.report_cases(reports, allow_unit_files=True, expected=inventory),
        inventory,
    )


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_the_repository_loader_arguments_validate_the_source_cases(tmp_path, platform):
    """The loader passes test trees; hardware passes report directories."""
    manifest_json = tmp_path / "manifest.json"
    config_json = tmp_path / "config.json"
    manifest_json.write_text(json.dumps(manifest()))
    config_json.write_text(json.dumps(platform_config(platform)))
    composed = subprocess.run(
        [
            sys.executable,
            str(_REPO / ".github/scripts/compose_integration_manifest.py"),
            "--platform",
            platform,
            "--manifest",
            str(manifest_json),
            "--platform-config",
            str(config_json),
            "--config-dir",
            str(_CONFIG_DIR),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    helper = subprocess.run(
        [sys.executable, str(_HELPER), "--composer-output", "/dev/stdin"],
        input=composed.stderr,
        text=True,
        capture_output=True,
        check=True,
    )
    arguments = _read_github_outputs(helper.stdout)["loader_inventory_arguments"]
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO / ".github/scripts/run_integration_tests.py"),
            RUNNER_VALIDATION,
            *shlex.split(arguments),
        ],
        env={
            **os.environ,
            "INTEGRATION_TESTS": composed.stdout,
            "INTEGRATION_ENVIRONMENT": "{}",
            "INTEGRATION_SOURCE_ROOT": str(_REPO),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        f"Validated {len(composed_steps(platform))} configured integration test"
        in result.stdout
    )


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
