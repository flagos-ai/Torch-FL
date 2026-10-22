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
one platform ran look like a step every platform ran; here the deviation is a
`platforms:` key in one file, and how much is still per-platform is a number
the loader prints on every run.

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
    "platforms",
    "environment",
    "environment_from_config",
}


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


def _entry_platforms(entry, known):
    platforms = entry.get("platforms")
    if platforms is None:
        return list(known)
    if not isinstance(platforms, list) or not platforms:
        raise ManifestError(
            f"step {entry.get('id')!r}: platforms must be a non-empty list"
        )
    for platform in platforms:
        if platform not in known:
            raise ManifestError(
                f"step {entry.get('id')!r}: platforms names {platform!r}, which is "
                f"not a platform config; known platforms are {', '.join(known)}"
            )
    if len(set(platforms)) != len(platforms):
        raise ManifestError(f"step {entry.get('id')!r}: platforms repeats a platform")
    return list(platforms)


def validate_manifest(manifest, known):
    """Check the manifest's shape and return its entries.

    The invariants that matter: every entry is complete, a step's variants sit
    together and never overlap (a platform that got the same step twice would
    run it twice, silently), and every entry names a platform that exists.
    """
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest must be a mapping with a steps list")
    steps = manifest.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ManifestError("the manifest must define a non-empty steps list")

    seen_order: list[str] = []
    assigned: dict[str, set[str]] = {}
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
        platforms = _entry_platforms(entry, known)

        if entry.get("environment") is not None:
            _environment(entry, f"steps[{index}].environment")
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

        if step_id in assigned:
            previous = seen_order[-1] if seen_order else None
            if previous != step_id:
                raise ManifestError(
                    f"step {step_id!r} is not contiguous: all of a step's variants "
                    "belong together, so collapsing them into one is a local edit"
                )
            overlap = assigned[step_id] & set(platforms)
            if overlap:
                raise ManifestError(
                    f"step {step_id!r} lists {', '.join(sorted(overlap))} in more "
                    "than one variant, so those platforms would run it twice"
                )
            assigned[step_id].update(platforms)
        else:
            assigned[step_id] = set(platforms)
            seen_order.append(step_id)

    return steps, seen_order, assigned


def _environment(entry, field):
    environment = entry.get("environment")
    if environment is None:
        # A step that needs none is the common case, and validate_manifest
        # accepts the key's absence; composing must read it the same way.
        return {}
    if not isinstance(environment, dict):
        raise ManifestError(f"{field} must be a mapping")
    for key, value in environment.items():
        _require_string(key, f"{field} key")
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ManifestError(f"{field}[{key}] must be a scalar")
    return {str(key): str(value) for key, value in environment.items()}


def compose(manifest, config, platform, known):
    """The `integration_tests` list for one platform.

    Order is the manifest's, never the config's, so two platforms that run the
    same steps run them in the same sequence.
    """
    steps, _ids, _assigned = validate_manifest(manifest, known)
    if platform not in known:
        raise ManifestError(
            f"unknown platform {platform!r}; known platforms are {', '.join(known)}"
        )
    if not isinstance(config, dict):
        raise ManifestError("the platform config must be a mapping")

    display_name = _require_string(config.get("display_name"), "display_name")

    consumed_from_config: set[str] = set()
    declared_from_config = {
        key for entry in steps for key in (entry.get("environment_from_config") or {})
    }

    composed = []
    for entry in steps:
        if platform not in _entry_platforms(entry, known):
            continue
        environment = _environment(entry, f"step {entry['id']}.environment")
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
            consumed_from_config.add(key)
        step = {
            "name": entry["name"].replace(DISPLAY_NAME_TOKEN, display_name),
            "command": entry["command"],
        }
        if environment:
            step["environment"] = environment
        composed.append(step)

    # The other direction: data a config states that no step of this platform
    # reads is data nobody checks. Left unchecked, a platform can carry a whole
    # expectation block and run against nothing.
    unread = sorted(
        key
        for key in declared_from_config
        if config.get(key) and key not in consumed_from_config
    )
    if unread:
        raise ManifestError(
            f"the {platform} config defines {', '.join(unread)}, which no step that "
            "runs on this platform consumes"
        )

    if not composed:
        raise ManifestError(f"no steps apply to {platform!r}")
    return composed


def coverage_report(assigned, known):
    """Lines describing how far the platforms still differ, worst first."""
    lines = []
    for step_id, platforms in assigned.items():
        missing = [platform for platform in known if platform not in platforms]
        if missing:
            lines.append(
                f"  {step_id}: {len(known) - len(missing)}/{len(known)} platforms "
                f"(no {', '.join(missing)})"
            )
    return lines


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
        steps, _ids, assigned = validate_manifest(manifest, known)
        composed = compose(manifest, config, args.platform, known)
    except ManifestError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as error:
        print(f"::error::cannot read the manifest inputs: {error}", file=sys.stderr)
        return 1

    # Printed on every load, in every platform's run: the number is the
    # progress measure for making the platforms one pipeline instead of eight.
    variants = len(steps) - len(assigned)
    print(
        f"{args.platform}: {len(composed)} groups from {len(steps)} manifest entries "
        f"({variants} platform-restricted variant(s))",
        file=sys.stderr,
    )
    report = coverage_report(assigned, known)
    if report:
        print(
            f"{len(report)} step(s) do not run on every platform yet:",
            file=sys.stderr,
        )
        for line in report:
            print(line, file=sys.stderr)

    print(json.dumps(composed, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
