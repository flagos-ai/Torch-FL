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

The composer uses only the standard library. The shared platform parser calls
it with decoded YAML data on a CPU runner; the JSON CLI remains useful for
local inspection and boundary tests. Vendor images need no YAML dependency.
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
    "profiles",
    "min_passed",
    "vendor_only",
    "boxing_variant",
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


def _has_scope(entry):
    """Whether an entry's id carries a ``.<platform>`` suffix."""
    return "." in _require_string(entry.get("id"), "id")


def _step_id(entry):
    """An entry's step id, with any ``.<platform>`` owner suffix removed.

    ``vendor_ops.dcu`` is step ``vendor_ops`` as DCU spells it -- see
    ``_step_owner``.
    """
    return _require_string(entry.get("id"), "id").split(".", 1)[0]


def _step_owner(entry, known):
    """The platform an id is scoped to, or None for a shared step.

    An id may end in ``.<platform>`` (``vendor_ops.dcu``). That says the entry
    is *that platform's own spelling* of the step: no other platform is
    expected to run it, and the coverage report does not count it against
    them. An id without a suffix is the shared step, where every platform that
    runs it counts.
    """
    step_id = _require_string(entry.get("id"), "id")
    if "." not in step_id:
        return None
    owner = step_id.split(".", 1)[1]
    if owner not in known:
        raise ManifestError(
            f"step {step_id!r} is scoped to {owner!r}, which is not a platform "
            f"config; known platforms are {', '.join(known)}"
        )
    return owner


def _base_platforms(entry, known):
    """The platforms an entry names, before a ``.<platform>`` scope is applied."""
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
    return list(platforms)


def _entry_platforms(entry, known):
    """The platforms an entry applies to when composed."""
    platforms = _base_platforms(entry, known)
    owner = _step_owner(entry, known)
    if owner is not None:
        return [platform for platform in platforms if platform != owner]
    return list(platforms)


def boxing_platforms(config_dir) -> list[str]:
    """The platforms whose config opts into a second, CUDA-boxing lane.

    Read from ``boxing_lane: true`` in each config. The JSON the loader hands
    this script is only this run's platform, so the rest of the directory is
    read for the coverage report's denominator.
    """
    directory = Path(config_dir)
    return sorted(
        path.stem for path in directory.glob("*.yml") if _config_opts_into_boxing(path)
    )


def _config_opts_into_boxing(path) -> bool:
    """Whether a config file states ``boxing_lane: true``.

    Read as text rather than parsed: this script is stdlib-only by design (see
    the module docstring), and the key has exactly one spelling in this
    repository. The manifest tests parse the keys properly and hold the two
    readings to each other.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return False
    return any(
        line.split("#", 1)[0].strip() == "boxing_lane: true"
        for line in text.splitlines()
    )


# The builds an entry's lane covers: a shared entry runs on both.
_BUILDS = {"shared": ("vendor", "boxing"), "vendor": ("vendor",), "boxing": ("boxing",)}


def _entry_lane(entry, boxing):
    """Which build an entry is composed into: "shared", "vendor" or "boxing".

    A shared entry runs on both builds of every platform that runs it. A
    ``.<platform>``-scoped id is that platform's boxing build -- it names the
    owner, and the purpose of the scope is exactly that build.
    ``vendor_only`` is the opposite: the vendor build of a boxing platform,
    where only that build can have what the step asserts.
    """
    if entry.get("vendor_only") and _has_scope(entry):
        raise ManifestError(
            f"step {entry.get('id')!r} is both vendor_only and scoped; a scope "
            "already says which build its entries are for"
        )
    if entry.get("vendor_only") or entry.get("boxing_variant"):
        return "vendor"
    if _has_scope(entry):
        return "boxing"
    return "shared"


def _check_boxing_variant(entry, already_scoped, boxing):
    """Refuse a ``boxing_variant`` that names a platform it cannot apply to.

    ``boxing_variant`` is the inline form of a ``.<platform>`` scope, for a step
    whose selection matches on a per-build mark (DCU's ``vendor_ops`` and its
    ``vendor_ops.hygon`` run the same command, so the second id adds a name but
    not a selection). It names boxing platforms whose boxing build runs this
    entry's command, and it is refused beside a scope: an entry is one build's
    entry or every build's, never both.
    """
    inline = entry.get("boxing_variant")
    if inline is None:
        return
    if _has_scope(entry):
        raise ManifestError(
            f"step {entry.get('id')!r} is both scoped and boxing_variant; a "
            "scope already says which build its entries are for"
        )
    if entry.get("vendor_only"):
        raise ManifestError(
            f"step {entry.get('id')!r} is both vendor_only and boxing_variant; "
            "boxing_variant already says which build it is for"
        )
    if not isinstance(inline, list) or not inline:
        raise ManifestError(
            f"step {entry.get('id')!r}: boxing_variant must be a non-empty list"
        )
    for platform in inline:
        if platform not in boxing:
            raise ManifestError(
                f"step {entry.get('id')!r}: boxing_variant names {platform!r}, "
                f"which has no boxing lane; boxing platforms are "
                f"{', '.join(boxing) or 'none'}"
            )
        if platform in already_scoped:
            raise ManifestError(
                f"step {entry.get('id')!r} claims {platform!r}'s boxing build "
                "twice: via this entry's boxing_variant and via a "
                f".{platform} scope"
            )


def validate_manifest(manifest, known, boxing=()):
    """Check the manifest's shape and return its entries.

    The invariants that matter: every entry is complete, a step's variants sit
    together and never overlap per build (a platform that got the same step
    twice on one build would run it twice, silently), and every entry names a
    platform that exists.

    Returns the entries, their ids in first-seen order, and for each id the
    platforms that run it on each build. ``boxing`` is the platforms that have
    a second build (``boxing_platforms``); a ``.<platform>``-scoped id belongs
    to that platform's boxing build.
    """
    if boxing:
        boxing = sorted(boxing)
    else:
        boxing = [platform for platform in known if _config_opts_into_boxing(platform)]
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest must be a mapping with a steps list")
    steps = manifest.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ManifestError("the manifest must define a non-empty steps list")

    seen_order: list[str] = []
    # step id -> build -> the platforms that run it there.
    assigned: dict[str, dict[str, set[str]]] = {}
    named: dict[str, dict[str, set[str]]] = {}
    owners_seen: dict[str, set[str]] = {}
    kinds_seen: dict[str, set[str]] = {}
    shared_named: dict[str, set[str]] = {}
    for index, entry in enumerate(steps):
        if not isinstance(entry, dict):
            raise ManifestError(f"steps[{index}] must be a mapping")
        unknown = set(entry) - _ENTRY_KEYS
        if unknown:
            raise ManifestError(
                f"steps[{index}] has unknown key(s): {', '.join(sorted(unknown))}"
            )
        step_id = _step_id(entry)
        platform_list = entry.get("platforms")
        if isinstance(platform_list, list) and len(set(platform_list)) != len(
            platform_list
        ):
            raise ManifestError(
                f"steps[{index}].platforms repeats {entry.get('id')!r}'s platform"
            )
        _require_string(entry.get("name"), f"steps[{index}].name")
        _require_string(entry.get("command"), f"steps[{index}].command")
        platforms = _entry_platforms(entry, known)
        minimum = entry.get("min_passed", 0)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise ManifestError(
                f"step {step_id!r}: min_passed must be a nonnegative integer"
            )
        profiles = entry.get("profiles", ["pr", "nightly"])
        if (
            not isinstance(profiles, list)
            or not profiles
            or any(profile not in ("pr", "nightly") for profile in profiles)
        ):
            raise ManifestError(f"step {step_id!r}: profiles must name pr or nightly")
        owner = _step_owner(entry, known)
        if owner is not None and owner not in boxing:
            raise ManifestError(
                f"step {entry.get('id')!r} is scoped to {owner!r}, which has no "
                f"boxing lane; boxing platforms are {', '.join(boxing) or 'none'}"
            )
        lane = _entry_lane(entry, boxing)
        if owner is not None and lane != "boxing":
            raise ManifestError(
                f"steps[{index}] ({entry.get('id')!r}) is scoped to {owner!r}, "
                "which has no boxing lane to be scoped to"
            )
        owners_seen.setdefault(step_id, set())
        if owner is not None:
            owners_seen[step_id].add(owner)
        # What the entry itself says about the builds it is for: an unscoped
        # shared entry covers both.
        kinds_seen.setdefault(step_id, set()).add(lane)
        if platform_list is None and lane == "shared":
            shared_named.setdefault(step_id, set()).update(known)

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

        inline = entry.get("boxing_variant")
        if step_id in assigned:
            previous = seen_order[-1] if seen_order else None
            if previous != step_id:
                raise ManifestError(
                    f"step {step_id!r} is not contiguous: all of a step's variants "
                    "belong together, so collapsing them into one is a local edit"
                )
            # A platform appears once per *lane*: a boxing platform runs the
            # vendor_only variant on its vendor build and its own scoped
            # variant on its boxing build, and every other platform runs one
            # entry on its one build. Two entries of one lane for one platform
            # would run that step twice on that build.
            overlap = named[step_id][lane] & set(platforms)
            if overlap:
                raise ManifestError(
                    f"step {step_id!r} lists {', '.join(sorted(overlap))} in more "
                    f"than one {lane}-lane variant, so those builds would run it "
                    "twice"
                )
            for build in _BUILDS.get(lane, (lane,)):
                assigned[step_id][build].update(platforms)
            named[step_id][lane].update(platforms or [owner])
        else:
            assigned[step_id] = dict.fromkeys(("vendor", "boxing"), set())
            named[step_id] = dict.fromkeys(("vendor", "boxing", "shared"), set())
            for build in _BUILDS.get(lane, (lane,)):
                assigned[step_id][build].update(platforms)
            named[step_id][lane].update(platforms or [owner])
            if inline is not None:
                owners_seen[step_id].update(inline)
            seen_order.append(step_id)

    for step_id in seen_order:
        scoped_owners = owners_seen.get(step_id, set())
        inline_owners = {
            platform
            for entry in steps
            if _step_id(entry) == step_id
            for platform in (entry.get("boxing_variant") or ())
        }
        for entry in steps:
            if _step_id(entry) == step_id:
                _check_boxing_variant(
                    entry,
                    scoped_owners - set(entry.get("boxing_variant") or ()),
                    boxing,
                )
        # A boxing platform runs one spelling of a step on one build:
        #
        #   * its own `.<platform>` variant and the shared entry cannot both
        #     exist -- only one of them can be its boxing build's entry;
        #   * a step that *is* spelled per build (some entry is scoped or
        #     vendor_only) must spell it for every boxing platform that runs
        #     it, or that platform's boxing build silently keeps the vendor
        #     build's selection.
        anchored = bool(kinds_seen[step_id] & {"vendor", "boxing"})
        both = sorted(
            platform
            for platform in boxing
            if anchored and platform in shared_named.get(step_id, set())
        )
        if both:
            raise ManifestError(
                f"step {step_id!r} has its own variant and a shared entry for "
                f"{', '.join(both)}; a boxing platform runs one spelling of a "
                "step on its boxing build"
            )
        unscoped = sorted(
            platform
            for platform in boxing
            if anchored
            and platform in assigned[step_id]["vendor"]
            and platform not in scoped_owners
            and platform not in inline_owners
            and not any(
                entry.get("vendor_only") and platform in _entry_platforms(entry, known)
                for entry in steps
                if _step_id(entry) == step_id
            )
        )
        if unscoped:
            raise ManifestError(
                f"step {step_id!r} runs on the boxing build of "
                f"{', '.join(unscoped)} with its vendor build's entry; give the "
                "platform its own `.<platform>` variant or mark the shared entry "
                "vendor_only"
            )

    return steps, seen_order, assigned, named


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


def compose(
    manifest,
    config,
    platform,
    known,
    boxing_lane=False,
    environment_check=None,
    boxing=(),
    profile="nightly",
):
    """The `integration_tests` list for one platform.

    Order is the manifest's, never the config's, so two platforms that run the
    same steps run them in the same sequence.

    ``boxing_lane`` composes the platform's CUDA-boxing build instead of its
    vendor build: a boxing platform runs every shared step on both, the
    ``vendor_only`` entries only on its vendor build, and the entries whose id
    is scoped to it (``vendor_ops.dcu``) only on its boxing build.

    ``environment_check`` replaces the config's ``environment_check`` block: the
    loader hands the boxing lane's ``boxing_environment_check`` here, because the
    two builds are held to different expectations from one config file.
    """
    if profile not in ("pr", "nightly"):
        raise ManifestError("weekly is reserved; supported profiles are pr and nightly")
    steps, _ids, _assigned, _named = validate_manifest(manifest, known, boxing)
    if platform not in known:
        raise ManifestError(
            f"unknown platform {platform!r}; known platforms are {', '.join(known)}"
        )
    if boxing_lane and platform not in boxing:
        raise ManifestError(
            f"{platform!r} has no boxing lane: its config states no boxing_lane: true"
        )
    if not isinstance(config, dict):
        raise ManifestError("the platform config must be a mapping")

    display_name = _require_string(config.get("display_name"), "display_name")

    consumed_from_config: set[str] = set()
    declared_from_config = {
        key
        for declared_step in steps
        for key in (declared_step.get("environment_from_config") or {})
    }

    def applies(entry):
        """Whether this build of ``platform`` runs ``entry``.

        A ``.<platform>``-scoped id is that platform's boxing build, and nobody
        else's. ``boxing_variant`` names the boxing builds that run this
        entry's command in place of the vendor one, so it is the vendor build's
        entry for them. Every other entry runs on both builds of a boxing
        platform unless it is ``vendor_only``, which is the vendor build's own;
        a platform without a boxing lane has only the vendor build, where that
        scope is inert.
        """
        owner = _step_owner(entry, known)
        runs_here = platform in _entry_platforms(entry, known)
        if owner is not None:
            return owner == platform and boxing_lane
        vendor_only = bool(entry.get("vendor_only"))
        if boxing_lane:
            return runs_here and not vendor_only
        if not runs_here:
            return False
        if vendor_only:
            return True
        inline = entry.get("boxing_variant") or ()
        return platform not in inline

    composed = []
    for entry in steps:
        if profile not in entry.get("profiles", ["pr", "nightly"]):
            continue
        if not applies(entry):
            continue
        environment = _environment(entry, f"step {entry['id']}.environment")
        for key, variable in (entry.get("environment_from_config") or {}).items():
            block = (
                environment_check
                if key == "environment_check" and environment_check is not None
                else config.get(key)
            )
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
            # Which manifest entry this group came from. A boxing platform runs
            # the same groups twice with a different entry selected for a
            # step, so the group list alone cannot say which build produced it.
            "manifest_id": entry["id"],
        }
        if entry.get("min_passed"):
            step["min_passed"] = entry["min_passed"]
        if environment:
            step["environment"] = environment
        composed.append(step)

    # The other direction: data a config states that no step of this platform
    # reads is data nobody checks. Left unchecked, a platform can carry a whole
    # expectation block and run against nothing.
    # Every block this build was actually handed counts as read: the boxing
    # build's expectations arrive as `boxing_environment_check`, so the vendor
    # block is not unread there -- it is simply not this build's.
    if environment_check is not None:
        consumed_from_config.add("boxing_environment_check")
        consumed_from_config.add("environment_check")
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
        raise ManifestError(
            f"no steps apply to {platform!r}{' boxing lane' if boxing_lane else ''}"
        )
    return composed


def coverage_report(assigned, named, known, boxing=()):
    """Lines describing how far the platforms still differ, worst first.

    The unit stays the platform. A step that has its own spelling for a boxing
    platform (``vendor_ops`` vs ``vendor_ops.dcu``) is reported over that
    platform alone: it is a per-platform step by construction, not a shared one
    most platforms are missing.
    """
    lines = []
    for step_id, by_lane in assigned.items():
        if any(named[step_id][lane] & set(boxing) for lane in ("vendor", "boxing")):
            covered = by_lane["vendor"] & named[step_id]["vendor"]
            covered |= by_lane["boxing"] & named[step_id]["boxing"]
        else:
            covered = by_lane["vendor"] | by_lane["boxing"]
        missing = [platform for platform in known if platform not in covered]
        if missing:
            lines.append(
                f"  {step_id}: {len(known) - len(missing)}/{len(known)} platforms "
                f"(no {', '.join(missing)})"
            )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--profile", default="pr")
    parser.add_argument("--manifest", required=True, help="manifest as JSON")
    parser.add_argument("--platform-config", required=True, help="config as JSON")
    parser.add_argument(
        "--config-dir",
        default=".github/configs",
        help="directory the platform list is read from",
    )
    parser.add_argument(
        "--boxing-lane",
        action="store_true",
        help=(
            "compose the platform's CUDA-boxing build (skips vendor_only steps, "
            "adds the ones scoped to this platform)"
        ),
    )
    parser.add_argument(
        "--environment-check",
        default=None,
        help="JSON block used in place of the config's environment_check",
    )
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(Path(args.manifest).read_text())
        config = json.loads(Path(args.platform_config).read_text())
        known = known_platforms(args.config_dir)
        boxing = boxing_platforms(args.config_dir)
        steps, _ids, assigned, named = validate_manifest(manifest, known, boxing)
        environment_check = (
            json.loads(args.environment_check)
            if args.environment_check is not None
            else None
        )
        composed = compose(
            manifest,
            config,
            args.platform,
            known,
            boxing_lane=args.boxing_lane,
            environment_check=environment_check,
            boxing=boxing,
            profile=args.profile,
        )
    except ManifestError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as error:
        print(f"::error::cannot read the manifest inputs: {error}", file=sys.stderr)
        return 1

    # Printed on every load, in every platform's run: the number is the
    # progress measure for making the platforms one pipeline instead of eight.
    variants = len(steps) - len(assigned)
    lane = f"{args.platform} (boxing)" if args.boxing_lane else args.platform
    print(
        f"{lane}: {len(composed)} groups from {len(steps)} manifest entries "
        f"({variants} platform-restricted variant(s))",
        file=sys.stderr,
    )
    report = coverage_report(assigned, named, known, boxing)
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
