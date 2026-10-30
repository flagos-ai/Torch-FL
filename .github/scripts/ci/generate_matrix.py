#!/usr/bin/env python3
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

"""
CI platform matrix generator with changed-file filtering and label routing.

For schedule / push: returns all configured platforms.
For workflow_dispatch: uses ``--dispatch-platforms`` (``all`` by default).
For pull_request:
  1. If a PR label matches ``platform:<name>``   → run only that platform.
  2. If a PR label matches ``platform:all``       → run all platforms.
  3. Otherwise match every changed file against per-platform watch paths.
     Shared or unrecognized code runs all platforms; documentation alone
     runs the platform-agnostic checks without hardware jobs.

Usage (called from a GitHub Actions step):
    python .github/scripts/ci/generate_matrix.py \\
        --event-name  pull_request \\
        --base-ref    main \\
        --configs-dir .github/configs \\
        --pr-labels   "platform:ascend,ci" \\
        --dispatch-platforms "ascend" \\
        >> "$GITHUB_OUTPUT"

A boxing platform (.github/configs/<p>.yml with ``boxing_lane: true``) is
scheduled twice: the bare platform name runs the vendor build, and the
``<p>-boxing`` lane runs the CUDA-boxing build of the same platform. The two
are selected together -- there is no way to ask for one and not the other.

Output written to GITHUB_OUTPUT:
    platforms=["ascend","cuda","dcu","dcu-boxing","hygon","hygon-boxing"]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import subprocess
import sys

import yaml

# Config file stems that are metadata, not platform names.
_NON_PLATFORM_STEMS: frozenset[str] = frozenset(
    ("template", "image_sources", "platforms")
)


def _load_platforms(configs_dir: pathlib.Path) -> list[str]:
    """Return sorted platform names derived from *.yml files in configs_dir."""
    return sorted(
        p.stem for p in configs_dir.glob("*.yml") if p.stem not in _NON_PLATFORM_STEMS
    )


def _platform_config(configs_dir: pathlib.Path, platform: str) -> dict:
    """One platform's config, or {} when it cannot be read."""
    try:
        cfg = yaml.safe_load((configs_dir / f"{platform}.yml").read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def boxing_platforms(configs_dir: pathlib.Path, platforms: list[str]) -> list[str]:
    """The platforms whose config opts into a second, CUDA-boxing lane.

    Read from ``boxing_lane: true`` in the platform's own config rather than
    from a list here: the flag is a statement about that platform's build, and
    it sits beside the ``environment_check`` that holds the lane to it.
    """
    return [
        platform
        for platform in platforms
        if _platform_config(configs_dir, platform).get("boxing_lane") is True
    ]


def _schedule(platforms: list[str], boxing: list[str]) -> list[str]:
    """Each boxing platform followed by its ``<p>-boxing`` lane."""
    lanes = []
    for platform in platforms:
        lanes.append(platform)
        if platform in boxing:
            lanes.append(f"{platform}-boxing")
    return lanes


def _platform_of(lane: str, boxing: list[str]) -> str:
    """The platform a lane name stands for (``dcu-boxing`` -> ``dcu``)."""
    for platform in boxing:
        if lane == f"{platform}-boxing":
            return platform
    return lane


def _changed_files(
    base_ref: str, base_sha: str = "", head_sha: str = "HEAD"
) -> list[str] | None:
    """Compare immutable PR commits, retaining both sides of a rename."""
    base = base_sha or f"origin/{base_ref}"
    cmd = [
        "git",
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        f"{base}...{head_sha}",
        "--",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [path for path in result.stdout.split("\0") if path]
    except subprocess.CalledProcessError as exc:
        print(
            f"::warning::git diff failed ({exc}); running all platforms",
            file=sys.stderr,
        )
        return None


def _matches(path: str, pattern: str) -> bool:
    """Watch paths are exact files, directory prefixes or shell-style globs."""
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return fnmatch.fnmatchcase(path, pattern)


def _platforms_from_files(
    changed: list[str] | None,
    all_platforms: list[str],
    configs_dir: pathlib.Path,
    boxing: list[str] | None = None,
) -> list[str]:
    """The lanes a pull request's changed files select."""
    platforms = _platforms_from_files_bare(changed, all_platforms, configs_dir)
    if boxing is None:
        boxing = boxing_platforms(configs_dir, all_platforms)
    return _schedule(platforms, boxing)


def _platforms_from_files_bare(
    changed: list[str] | None,
    all_platforms: list[str],
    configs_dir: pathlib.Path,
) -> list[str]:
    if changed is None:
        return all_platforms
    documentation = (".md", ".rst")
    code = [
        path
        for path in changed
        if not path.startswith("docs/") and not path.lower().endswith(documentation)
    ]
    if not code:
        return []
    watches = {}
    for platform in all_platforms:
        try:
            cfg = yaml.safe_load((configs_dir / f"{platform}.yml").read_text())
            watch = cfg.get("paths_watch")
            if (
                not isinstance(watch, list)
                or not watch
                or not all(isinstance(pattern, str) and pattern for pattern in watch)
            ):
                return all_platforms
            watches[platform] = watch + [
                f".github/configs/{platform}.yml",
                f".github/workflows/integration-test-{platform}.yml",
            ]
        except (OSError, yaml.YAMLError, AttributeError):
            print(
                f"::warning::Invalid watch paths for {platform}; running all platforms",
                file=sys.stderr,
            )
            return all_platforms
    selected = set()
    for path in code:
        affected = {
            platform
            for platform, watch in watches.items()
            if any(_matches(path, pattern) for pattern in watch)
        }
        if not affected:
            # Unknown/shared code must never disappear behind one vendor match.
            return all_platforms
        selected.update(affected)
    return [platform for platform in all_platforms if platform in selected]


def _platforms_from_labels(
    labels: list[str],
    all_platforms: list[str],
) -> list[str] | None:
    """
    Return a platform list derived from PR labels, or None if no routing label.

    Label conventions
    -----------------
    ``platform:all``      → run the full matrix (same as no filter).
    ``platform:<name>``   → run only the named platform; any unknown name
                            conservatively restores the full matrix.

    If multiple ``platform:`` labels are present, their union is returned.
    Returns None when no ``platform:`` label is found (caller falls through
    to the file-change-based filtering).
    """
    prefix = "platform:"
    routing = [lb[len(prefix) :].strip() for lb in labels if lb.startswith(prefix)]
    if not routing:
        return None
    if "all" in routing:
        return all_platforms
    if any(platform not in all_platforms for platform in routing):
        return all_platforms
    return [platform for platform in all_platforms if platform in set(routing)]


def _platforms_from_dispatch(
    requested: str,
    all_platforms: list[str],
    boxing: list[str] | None = None,
) -> list[str]:
    """Validate and return the lanes requested by workflow_dispatch.

    A platform name selects its lanes whole: naming ``dcu`` dispatches the DCU
    vendor build and ``dcu-boxing`` together, which is what a person asking for
    DCU means and keeps the two from ratcheting apart. Naming the lane itself
    (``dcu-boxing``) is accepted too, and also brings its platform's other
    build: there is no way to ask for one lane and not the other.
    """
    if boxing is None:
        boxing = []
    lanes = _schedule(all_platforms, boxing)
    values = [value.strip() for value in requested.split(",") if value.strip()]
    if not values or values == ["all"]:
        return lanes
    platforms = {_platform_of(value, boxing) for value in values}
    if "all" in platforms or any(value not in all_platforms for value in platforms):
        unknown = sorted(platforms - set(all_platforms) - {"all"})
        detail = f": {', '.join(unknown)}" if unknown else ""
        raise ValueError(f"invalid workflow_dispatch platform selection{detail}")
    # Every lane of a named platform is dispatched, the boxing lane included:
    # the two are one wheel built two ways, so asking for the platform is
    # asking for both.
    return [lane for lane in lanes if _platform_of(lane, boxing) in platforms]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True, help="GitHub event name")
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Base branch for git diff (PR base)",
    )
    parser.add_argument(
        "--configs-dir",
        default=".github/configs",
        help="Directory containing platform *.yml configs",
    )
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Newline-separated list of changed files (overrides git diff)",
    )
    parser.add_argument("--base-sha", default="", help="Immutable PR base SHA")
    parser.add_argument("--head-sha", default="HEAD", help="Immutable PR head SHA")
    parser.add_argument(
        "--pr-labels",
        default="",
        help="Comma-separated PR label names (used for triage routing)",
    )
    parser.add_argument(
        "--dispatch-platforms",
        default="",
        help="Comma-separated platforms for workflow_dispatch; empty means all",
    )
    args = parser.parse_args()

    configs_dir = pathlib.Path(args.configs_dir)
    if not configs_dir.is_dir():
        print(
            f"::error::configs directory not found: {configs_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_platforms = _load_platforms(configs_dir)
    boxing = boxing_platforms(configs_dir, all_platforms)
    if not all_platforms:
        print(
            f"::error::No platform configs found in {configs_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.event_name == "workflow_dispatch":
        try:
            selected = _platforms_from_dispatch(
                args.dispatch_platforms,
                all_platforms,
                boxing,
            )
        except ValueError as exc:
            print(f"::error::{exc}", file=sys.stderr)
            sys.exit(2)
        print(f"platforms={json.dumps(selected)}")
        return

    # Other non-PR events always run the full matrix.
    if args.event_name != "pull_request":
        print(f"platforms={json.dumps(_schedule(all_platforms, boxing))}")
        return

    # PR: label routing takes priority over file-change filtering.
    labels = [lb.strip() for lb in args.pr_labels.split(",") if lb.strip()]
    label_platforms = _platforms_from_labels(labels, all_platforms)
    if label_platforms is not None:
        print(f"platforms={json.dumps(_schedule(label_platforms, boxing))}")
        return

    # PR: compute reduced matrix from changed files.
    if args.changed_files is not None:
        changed = [f for f in args.changed_files.splitlines() if f.strip()]
    else:
        changed = _changed_files(args.base_ref, args.base_sha, args.head_sha)
    selected = _platforms_from_files(changed, all_platforms, configs_dir, boxing)
    print(f"platforms={json.dumps(selected)}")


if __name__ == "__main__":
    main()
