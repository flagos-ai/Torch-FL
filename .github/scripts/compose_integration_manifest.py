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

"""Expand the shared integration manifest for one platform.

The manifest (.github/integration-manifest.yml) is the one ordered
list of groups the integration pipeline runs. A platform config is data: it
has no step list of its own, and cannot add, drop or reword a step. What a
platform *can* do is state the values a step needs -- its device
expectations, the environment its runner sets -- and this script folds those
into the step so `run_integration_tests.py` receives the same
`integration_tests` JSON it always has.

That shape is the point. Eight per-platform step lists made a step that only
one platform ran look like a step every platform ran; here every step is one
entry that every platform runs, and what a platform contributes to it is a
value in its own config.

Two kinds of value a platform states reach a step:

  * `environment_check` (and any other block named by a step's
    `environment_from_config`), handed over as JSON in one environment
    variable;
  * `step_environment`, a mapping of step id to the environment variables that
    step alone needs on this platform -- CUDA's `FLAGOS_USE_FLAGTREE=1` for the
    compile step, Ascend's process-start `LD_PRELOAD` for the profiler step.
    Keys are step ids rather than step numbers so a step that moves in the
    manifest does not silently carry someone else's environment.

The second kind is what keeps a *value* from having to be a step: the alternative
is a `platforms:`-restricted variant of the step per platform, which is how the
eight-way divergence grew in the first place.

Input is JSON, not YAML: this runs in the loader job on ubuntu-latest, where
`yq` is already the YAML reader, and the vendor images are not required to
carry a Python YAML module. The parsing side of the seam is `yq -o=json`; this
side is the standard library alone, so it can also be imported by the unit
tests that hold the manifest's invariants.
"""

import argparse
import json
import sys
from pathlib import Path


# The one substitution the composer performs. Anything else that looks like a
# variable belongs to the shell on the hardware runner.
DISPLAY_NAME_TOKEN = "{{display_name}}"

_ENTRY_KEYS = {
    "id",
    "name",
    "command",
    "environment",
    "environment_from_config",
}

# A platform config key: step id -> environment variables that step alone needs.
_STEP_ENVIRONMENT_KEY = "step_environment"

# The pytest-level gate table, read here only to print what each platform will
# skip. It lives under tests/ because the hardware job runs the wheel-only test
# workspace, which holds `tests` and nothing else; this loader job runs in the
# checkout, so it can read the same file and say the same thing before a runner
# is queued. Loading it is optional: a missing file prints nothing rather than
# failing a load, and tests/unit/test_platform_gates.py holds it to existence.
_GATES_PATH = (
    Path(__file__).resolve().parents[2] / "tests/integration/platform_gates.py"
)


class ManifestError(Exception):
    """A manifest or platform config that cannot be composed."""


def known_platforms(config_dir) -> list[str]:
    """Every platform that has a config, by basename.

    Read from the directory rather than from a list in the manifest: a
    platform is a file, and a manifest that names a platform no file defines
    would otherwise drop that step for every platform at once and look fine.

    Nothing else may live in that directory. It is also the directory ci.yml's
    Discover platforms step and generate_matrix.py enumerate, so a second kind
    of `.yml` in there becomes a platform and gets scheduled hardware: the
    manifest sits at `.github/integration-manifest.yml` for that reason, and
    this function reads the directory without a name to skip.
    """
    directory = Path(config_dir)
    if not directory.is_dir():
        raise ManifestError(f"config directory not found: {directory}")
    return sorted(path.stem for path in directory.glob("*.yml"))


def _require_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def validate_manifest(manifest):
    """Check the manifest's shape and return its entries and their ids.

    The invariants that matter: every entry is complete, and every id is defined
    exactly once. An id used twice would run one step twice, silently -- and it
    is also the shape a platform-restricted *variant* of a step used to take, so
    the refusal covers both readings: this manifest states steps, and the only
    way a platform stops running one is if the step leaves this file.

    `platforms:` is not in `_ENTRY_KEYS`, so an entry carrying one is refused as
    an unknown key. That is deliberate: a step a platform does not run is a
    decision about the platform, and every such decision belongs in the
    pytest-level gate table (tests/integration/platform_gates.py), where it is
    printed per platform, recorded in
    docs/reference/platform-capability-matrix.md and reasoned about in one
    place. A `platforms:` key here would let the same decision be made twice, in
    two files, with nothing comparing them.
    """
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest must be a mapping with a steps list")
    steps = manifest.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ManifestError("the manifest must define a non-empty steps list")

    ids: list[str] = []
    for index, entry in enumerate(steps):
        if not isinstance(entry, dict):
            raise ManifestError(f"steps[{index}] must be a mapping")
        unknown = set(entry) - _ENTRY_KEYS
        if unknown:
            raise ManifestError(
                f"steps[{index}] has unknown key(s): {', '.join(sorted(unknown))}"
            )
        step_id = _require_string(entry.get("id"), f"steps[{index}].id")
        _require_string(entry.get("name"), f"steps[{index}].name")
        _require_string(entry.get("command"), f"steps[{index}].command")

        _scalar_environment(entry.get("environment"), f"steps[{index}].environment")
        environment_from = entry.get("environment_from_config")
        if environment_from is not None:
            if not isinstance(environment_from, dict) or not environment_from:
                raise ManifestError(
                    f"steps[{index}].environment_from_config must be a non-empty "
                    "mapping of platform config key to environment variable name"
                )
            for key, variable in environment_from.items():
                _require_string(key, f"steps[{index}].environment_from_config key")
                _require_string(
                    variable, f"steps[{index}].environment_from_config[{key}]"
                )

        if step_id in ids:
            raise ManifestError(
                f"steps[{index}].id repeats {step_id!r}: one step is one entry that "
                "every platform runs, so a second entry for an id would either run "
                "it twice or restrict it to one platform, and a restriction belongs "
                "in tests/integration/platform_gates.py instead"
            )
        ids.append(step_id)

    return steps, ids


def _scalar_environment(value, field):
    """One environment mapping, normalized to a string-to-string dict."""
    if value is None:
        # A step that needs none is the common case, and validate_manifest
        # accepts the key's absence; composing must read it the same way.
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be a mapping")
    for key, item in value.items():
        _require_string(key, f"{field} key")
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            raise ManifestError(f"{field}[{key}] must be a scalar")
    return {str(key): str(item) for key, item in value.items()}


def _step_environment(config, platform, ids):
    """The per-step environment this platform states, validated.

    Keys are step ids, not step numbers, so a step that moves in the manifest
    does not silently carry someone else's environment -- and a key naming an id
    the manifest does not define is refused rather than dropped, because it
    would read as a step getting its environment while no such step runs.
    """
    block = config.get(_STEP_ENVIRONMENT_KEY)
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ManifestError(
            f"the {platform} config states {_STEP_ENVIRONMENT_KEY} as "
            f"{type(block).__name__} rather than a mapping of step id to the "
            "environment variables that step needs on this platform"
        )
    resolved = {}
    for step_id, environment in block.items():
        if step_id not in ids:
            raise ManifestError(
                f"the {platform} config sets {_STEP_ENVIRONMENT_KEY} for "
                f"{step_id!r}, which is not a step in the manifest; the steps are "
                f"{', '.join(ids)}"
            )
        resolved[step_id] = _scalar_environment(
            environment, f"{_STEP_ENVIRONMENT_KEY}.{step_id} of the {platform} config"
        )
    return resolved


def compose(manifest, config, platform, known):
    """The `integration_tests` list for one platform.

    Order is the manifest's, never the config's, so two platforms that run the
    same steps run them in the same sequence.
    """
    steps, ids = validate_manifest(manifest)
    if platform not in known:
        raise ManifestError(
            f"unknown platform {platform!r}; known platforms are {', '.join(known)}"
        )
    if not isinstance(config, dict):
        raise ManifestError("the platform config must be a mapping")

    display_name = _require_string(config.get("display_name"), "display_name")

    step_environment = _step_environment(config, platform, ids)

    composed = []
    for entry in steps:
        environment = _scalar_environment(
            entry.get("environment"), f"step {entry['id']}.environment"
        )
        for key, variable in (entry.get("environment_from_config") or {}).items():
            block = config.get(key)
            if not block:
                raise ManifestError(
                    f"step {entry['id']!r} reads {key!r} from the {platform} config, "
                    "which does not define it"
                )
            if not isinstance(block, dict):
                raise ManifestError(
                    f"step {entry['id']!r} reads {key!r} from the {platform} "
                    f"config, which states it as {type(block).__name__} rather "
                    "than the mapping the step is handed"
                )
            environment[variable] = json.dumps(
                block, separators=(",", ":"), sort_keys=True
            )
        # Last, so a platform states an exception beside the shared value rather
        # than instead of it: the step's own `environment` is the value every
        # platform gets, and `step_environment` is what this one adds or
        # overrides. `LD_PRELOAD` on the profiler step is the case that reads
        # this way -- no other platform sets it, and no platform has to know.
        environment.update(step_environment.get(entry["id"], {}))
        step = {
            "name": entry["name"].replace(DISPLAY_NAME_TOKEN, display_name),
            "command": entry["command"],
        }
        if environment:
            step["environment"] = environment
        composed.append(step)

    if not composed:
        raise ManifestError(f"no steps apply to {platform!r}")
    return composed


def _gates_module():
    """The pytest-level gate table, loaded by path, or None when unavailable."""
    import importlib.util

    if not _GATES_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("platform_gates", _GATES_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_report(platform):
    """Lines describing the pytest-level skips that apply to one platform.

    Every manifest entry runs on every platform, so a platform that cannot
    exercise one reads as full coverage unless the skips it still makes are
    printed. This is that print, from the same table the unit test holds to the
    gates in the test tree -- so what a run says it will skip is what it skips.
    """
    try:
        module = _gates_module()
    except (ImportError, AttributeError, OSError, SyntaxError):
        # A gate table that cannot be read must not fail a load on a vendor
        # runner: the tests themselves carry the gates, and
        # tests/unit/test_platform_gates.py is where the file is required to
        # exist and be importable.
        return []
    if module is None:
        return []
    return [
        f"  {gate.step}: {gate.target} ({gate.reason})"
        for gate in module.gates_for(platform)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--manifest", required=True, help="manifest as JSON")
    parser.add_argument("--platform-config", required=True, help="config as JSON")
    parser.add_argument(
        "--config-dir",
        default=".github/configs",
        help="directory the platform list is read from",
    )
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(Path(args.manifest).read_text())
        config = json.loads(Path(args.platform_config).read_text())
        known = known_platforms(args.config_dir)
        steps, ids = validate_manifest(manifest)
        composed = compose(manifest, config, args.platform, known)
    except ManifestError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as error:
        print(f"::error::cannot read the manifest inputs: {error}", file=sys.stderr)
        return 1

    # Printed on every load, in every platform's run: one line saying the
    # pipeline no longer varies by platform, and then the skips that remain,
    # which are pytest-level and are the only thing left that can differ.
    print(
        f"{args.platform}: {len(composed)} groups from {len(steps)} manifest entries "
        f"({len(ids)} step id(s), every entry on every platform)",
        file=sys.stderr,
    )
    report = gate_report(args.platform)
    if report:
        print(
            f"{len(report)} pytest-level gate(s) apply to {args.platform}:",
            file=sys.stderr,
        )
        for line in report:
            print(line, file=sys.stderr)

    print(json.dumps(composed, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
