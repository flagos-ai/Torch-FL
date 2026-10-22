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

"""Turn the composer's inventory plan into the loader's workflow outputs.

The composer decides which shared inventory groups a platform's plan runs;
this helper is the one place that turns that decision into

* ``--inventory GROUP=KIND`` arguments for the hardware runner;
* ``--inventory GROUP=SOURCE`` arguments the loader validates from the source
  tree, without materializing any reports; and
* the exact ``{"cases", "aliases"}`` document of every shared inventory, so
  the hardware job can materialize each kind it runs.

It prints GitHub output heredocs, so the loader writes one well-formed output
per line group instead of joining shell fragments inside the workflow.
"""

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_plan(text, expected_kinds):
    """Validate the composer's ``inventory-plan`` line and return it."""
    lines = [
        line[len("inventory-plan=") :]
        for line in text.splitlines()
        if line.startswith("inventory-plan=")
    ]
    if len(lines) != 1:
        raise SystemExit(
            "the composer must print exactly one inventory-plan line, "
            f"found {len(lines)}"
        )
    try:
        plan = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"the composer's inventory plan is not JSON: {error}"
        ) from error
    if not isinstance(plan, list) or not plan:
        raise SystemExit("the composer's inventory plan must be a non-empty list")
    groups = []
    kinds = []
    for entry in plan:
        if not isinstance(entry, dict) or set(entry) != {
            "group",
            "kind",
            "requires_build_kernels",
            "source",
        }:
            raise SystemExit(f"malformed inventory plan entry: {entry!r}")
        group, kind = entry["group"], entry["kind"]
        source = entry["source"]
        if group in groups:
            raise SystemExit(f"the inventory plan repeats group {group!r}")
        if kind != expected_kinds.get(group):
            raise SystemExit(
                f"inventory group {group!r} binds {kind!r}, expected "
                f"{expected_kinds.get(group)!r}"
            )
        if (
            not isinstance(source, str)
            or not source
            or Path(source).is_absolute()
            or ".." in Path(source).parts
        ):
            raise SystemExit(
                f"inventory group {group!r} names no repository-relative source"
            )
        if not isinstance(entry["requires_build_kernels"], list):
            raise SystemExit(
                f"inventory group {group!r} must list its build kernel sets"
            )
        groups.append(group)
        kinds.append(kind)
    return plan, groups, kinds


def inventory_documents(kinds, expected_kinds):
    """The exact shared inventory document for each distinct kind."""
    if not kinds:
        raise SystemExit("the inventory plan runs no shared inventory")
    unknown = set(kinds) - set(expected_kinds.values())
    if unknown:
        raise SystemExit(f"unknown inventory kind(s): {', '.join(sorted(unknown))}")
    cases = _load(Path(__file__).with_name("flaggems_cases.py"), "flaggems_cases")
    documents = {}
    for kind in kinds:
        if kind in documents:
            continue
        try:
            documents[kind] = cases.inventory_document(kind)
        except cases.InventoryError as error:
            raise SystemExit(str(error)) from error
    return documents


def arguments(entries, value):
    """One shell-safe ``--inventory`` argument pair per plan entry."""
    return " ".join(
        f"--inventory {shlex.quote(f'{group}={value(entry)}')}"
        for group, entry in entries
    )


def emit(name, value):
    """One GitHub output, in heredoc form so multi-line JSON survives."""
    print(f"{name}<<__{name}__")
    if value:
        print(value)
    print(f"__{name}__")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composer-output", required=True)
    args = parser.parse_args(argv)

    composer = _load(
        Path(__file__).with_name("compose_integration_manifest.py"), "composer"
    )
    try:
        text = Path(args.composer_output).read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"cannot read the composer output: {error}") from error
    plan, groups, kinds = parse_plan(text, composer.INVENTORY_GROUPS)
    documents = inventory_documents(kinds, composer.INVENTORY_GROUPS)
    emit("integration_inventories", json.dumps(groups, separators=(",", ":")))
    emit(
        "loader_inventory_arguments",
        arguments(
            zip(groups, plan, strict=True),
            lambda entry: str(entry["source"]),
        ),
    )
    emit(
        "integration_inventory_arguments",
        arguments(zip(groups, kinds, strict=True), lambda kind: kind),
    )
    emit("test_inventory_kinds", " ".join(kinds))
    emit(
        "test_inventories",
        json.dumps(documents, separators=(",", ":"), sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
