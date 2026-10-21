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

"""Export exact pytest node IDs without importing torch or torch_fl.

The integration runner compares a native pytest report with these inventories.
Reading source tables in the CPU-only loader catches a newly declared case
before hardware is scheduled, and every hardware job receives the same plan.
Only inspectable literal tables are accepted; an unrecognized expression fails
the composition instead of silently shrinking the required case set.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path


_DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def source_root():
    """The checkout the inventory describes, or the tree containing this file."""
    value = None
    if __name__ == "__main__":
        value = os.environ.get("INTEGRATION_SOURCE_ROOT")
    return _DEFAULT_ROOT if not value else Path(value).resolve()


ROOT = source_root()
SHARED_SDPA = "tests/integration/shared_sdpa_caps.py"
VENDOR_SDPA = "tests/integration/vendor_lanes/sdpa_lane_caps.py"
VENDOR_CASES = "tests/integration/vendor_lanes/vendor_caps.py"
VENDOR_LANES = "tests/integration/vendor_lanes/test_sdpa_lanes.py"
FLAGGEMS_CASES = "tests/integration/flaggems/flaggems_caps.py"
FLAGGEMS_SDPA = "tests/integration/flaggems/sdpa_caps.py"
VENDOR_FILE = "tests/integration/vendor_lanes/test_sdpa_lanes.py"
SDPA_FILE = "tests/integration/flaggems/test_flaggems_sdpa_route.py"
FLAGGEMS_FILES = (
    "tests/integration/flaggems/device_operands_cases.py",
    "tests/integration/flaggems/pinned_cases.py",
    "tests/integration/flaggems/test_flaggems_device_identity.py",
    "tests/integration/flaggems/test_flaggems_exclusions.py",
    "tests/integration/flaggems/test_flaggems_execution.py",
    "tests/integration/flaggems/test_flaggems_route_table.py",
    "tests/integration/flaggems/test_flaggems_routing.py",
    SDPA_FILE,
)
CASE_FUNCTIONS = {"_case", "_plain_case", "_ascend_case", "_gcu_case"}


class InventoryError(ValueError):
    """A source table cannot be inspected as an exact test plan."""


def _tree(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise InventoryError(f"cannot read {path}: {error}") from error


def _location(path, node):
    try:
        location = path.relative_to(ROOT).as_posix()
    except ValueError:
        location = path.as_posix()
    return f"{location}:{node.lineno}"


def _path(value):
    """Resolve a repository-relative source path against the selected root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class Source:
    """Literal expressions in the test modules that declare case tables."""

    def path(self, value):
        """The parsed path for a repository-relative or absolute source name."""
        return _path(value)

    def __init__(self, paths):
        self.trees = {_path(path): _tree(_path(path)) for path in paths}
        self.assignments = {}
        self.imported_modules = {}
        self.imported_names = {}
        for path, tree in self.trees.items():
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    for alias in node.names:
                        self.imported_names[(path, alias.asname or alias.name)] = (
                            node.module,
                            alias.name,
                        )
                targets = node.targets if isinstance(node, ast.Assign) else ()
                if isinstance(node, ast.AnnAssign):
                    targets = (node.target,)
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    key = (path, target.id)
                    if key in self.assignments:
                        raise InventoryError(
                            f"{_location(path, target)}: {target.id} is assigned twice"
                        )
                    self.assignments[key] = (path, node.value)

    def module(self, path, expression):
        path = self.path(path)
        if not isinstance(expression, ast.Name):
            raise InventoryError(
                f"{_location(path, expression)}: imported table is not a name"
            )
        imported = self.imported_names.get((path, expression.id))
        if imported and imported[0] == "shared_sdpa_caps":
            return SHARED_SDPA, imported[1]
        if imported and imported[0] == "flaggems_caps":
            return FLAGGEMS_CASES, imported[1]
        if imported and imported[0] == "sdpa_caps":
            return FLAGGEMS_SDPA, imported[1]
        raise InventoryError(
            f"{_location(path, expression)}: cannot locate table {expression.id}"
        )

    def lookup(self, path, name):
        path = self.path(path)
        key = (path, name)
        if key in self.assignments:
            return self.assignments[key]
        if any(key[0] == path and key[1] == name for key in self.imported_names):
            source, imported = self.module(path, ast.Name(id=name, lineno=1))
            imported_key = (self.path(source), imported)
            if imported_key in self.assignments:
                return self.assignments[imported_key]
        raise InventoryError(f"{path.name}: unknown table {name}")

    def list(self, path, expression):
        path = self.path(path)
        """Return a flat list of literal strings from a case table."""
        if isinstance(expression, ast.ListComp):
            if (
                len(expression.generators) != 1
                or not isinstance(expression.elt, ast.Attribute)
                or expression.elt.attr != "tag"
                or not isinstance(expression.elt.value, ast.Name)
                or not isinstance(expression.generators[0].target, ast.Name)
                or expression.generators[0].target.id != expression.elt.value.id
                or expression.generators[0].ifs
                or expression.generators[0].is_async
            ):
                raise InventoryError(
                    f"{_location(path, expression)}: only a single case.tag "
                    "comprehension is a supported inventory table"
                )
            iterable = expression.generators[0].iter
            if not isinstance(iterable, ast.Name):
                raise InventoryError(
                    f"{_location(path, iterable)}: case.tag comprehension "
                    "iterates a table name"
                )
            source, assigned = self.lookup(path, iterable.id)
            return self.list(source, assigned)
        if isinstance(expression, ast.Name) and expression.id == "COMMON_TAGS":
            return self.common_tags(path, expression)
        if isinstance(expression, ast.Name):
            source, assigned = self.lookup(path, expression.id)
            return self.list(source, assigned)
        if isinstance(expression, ast.Constant):
            if not isinstance(expression.value, str):
                raise InventoryError(
                    f"{_location(path, expression)}: table item is not a string"
                )
            return [expression.value]
        if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
            if isinstance(expression, ast.Set):
                # A pytest ID has to follow sorted(order), not source order.
                values = []
                for element in expression.elts:
                    values.extend(self.list(path, element))
                return sorted(set(values))
            result = []
            for element in expression.elts:
                result.extend(self.list(path, element))
            return result
        if isinstance(expression, ast.Dict):
            result = []
            for key, value in zip(expression.keys, expression.values):
                result.extend(self.list(path, value if key is None else key))
            return result
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                name = expression.func.id
                if name in CASE_FUNCTIONS | {"_math_case"}:
                    return [self.case(path, name, expression)]
                if name == "Case" and expression.args:
                    tag = _constant_strings(expression.args[0])
                    if len(tag) == 1:
                        return tag
                if name == "tuple" and len(expression.args) == 1:
                    return self.list(path, expression.args[0])
                if name in {"set", "frozenset"} and len(expression.args) == 1:
                    return sorted(set(self.list(path, expression.args[0])))
                if name == "sorted" and len(expression.args) == 1:
                    return sorted(set(self.list(path, expression.args[0])))
                function = self.function(path, name)
                result = _literal_return(function)
                if (
                    result
                    and isinstance(result, ast.Call)
                    and not isinstance(result.func, ast.Name)
                ):
                    generated = _generated_case_tags(function, path)
                    if generated:
                        return generated
                if result and len(expression.args) == len(function.args.args):
                    return self.list(path, result)
            if (
                isinstance(expression.func, ast.Attribute)
                and expression.func.attr == "keys"
                and not expression.args
            ):
                return self.list(path, expression.func.value)
        raise InventoryError(
            f"{_location(path, expression)}: unsupported table expression "
            f"{ast.unparse(expression)}"
        )

    def size(self, path, expression):
        """Count a parametrized table whose values need a runtime platform."""
        if isinstance(expression, ast.Name):
            try:
                source, assigned = self.lookup(path, expression.id)
            except InventoryError:
                for node in self.trees[path].body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        for body_node in ast.walk(node):
                            if (
                                isinstance(body_node, ast.Assign)
                                and any(
                                    isinstance(target, ast.Name)
                                    and target.id == expression.id
                                    for target in body_node.targets
                                )
                                and isinstance(body_node.value, (ast.Dict, ast.Set))
                            ):
                                return -abs(self.size(path, body_node.value))
                raise
            return abs(self.size(source, assigned))
        if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
            return sum(max(0, self.size(path, item)) for item in expression.elts)
        if isinstance(expression, ast.Call):
            if (
                isinstance(expression.func, ast.Attribute)
                and isinstance(expression.func.value, ast.Name)
                and expression.func.value.id == "torch"
                and expression.func.attr == "device"
            ):
                return 1
            if isinstance(expression.func, ast.Name):
                if expression.func.id == "tuple" and len(expression.args) == 1:
                    return self.size(path, expression.args[0])
                if expression.func.id == "sorted" and len(expression.args) == 1:
                    return self.size(path, expression.args[0])
                function = self.function(path, expression.func.id)
                result = _literal_return(function)
                if result is not None:
                    if len(expression.args) == len(function.args.args) and all(
                        isinstance(argument, ast.Constant)
                        for argument in expression.args
                    ):
                        bindings = {
                            name.arg: argument.value
                            for name, argument in zip(
                                function.args.args, expression.args, strict=True
                            )
                        }
                        return abs(_literal_size(result, bindings, path))
                    return abs(self.size(path, result))
            raise InventoryError(
                f"{_location(path, expression)}: cannot count runtime table"
            )
        if isinstance(expression, ast.Dict):
            return len(expression.keys)
        if isinstance(expression, ast.Set):
            if all(isinstance(item, ast.Constant) for item in expression.elts):
                return -len({item.value for item in expression.elts})
        if isinstance(expression, ast.Constant):
            return 1
        raise InventoryError(
            f"{_location(path, expression)}: cannot count runtime table"
        )

    def common_tags(self, path, expression):
        """The tags generated from COMMON_CASES, with no runtime import."""
        path = self.path(path)
        source, assigned = self.lookup(path, "COMMON_TAGS")
        if (
            source == self.path(SHARED_SDPA)
            and isinstance(assigned, ast.Call)
            and isinstance(assigned.func, ast.Name)
            and assigned.func.id == "tuple"
            and assigned.args
            and isinstance(assigned.args[0], ast.GeneratorExp)
            and isinstance(assigned.args[0].elt, ast.Attribute)
            and assigned.args[0].elt.attr == "tag"
            and isinstance(assigned.args[0].generators[0].iter, ast.Name)
            and assigned.args[0].generators[0].iter.id == "COMMON_CASES"
        ):
            return self.list(
                self.path(SHARED_SDPA),
                ast.Name(id="COMMON_CASES", lineno=expression.lineno),
            )
        raise InventoryError(
            f"{_location(path, expression)}: COMMON_TAGS is not a tag table"
        )

    def table(self, path, name):
        path = self.path(path)
        source, assigned = self.lookup(path, name)
        return source, assigned

    def function(self, path, name):
        path = self.path(path)
        for node in self.trees[path].body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise InventoryError(f"{path.name}: helper {name} is missing")

    def case(self, path, name, call):
        path = self.path(path)
        if name == "_math_case":
            return "math"
        if len(call.args) < 2:
            raise InventoryError(f"{_location(path, call)}: case has no literal tag")
        if name == "_ascend_case":
            tag_index = 1
        else:
            tag_index = 0
        tags = self.list(path, call.args[tag_index])
        if name != "_ascend_case" and len(tags) != 1:
            raise InventoryError(f"{_location(path, call)}: case has no single tag")
        return tags[0]


def _literal_return(function):
    returns = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    if len(returns) == 1:
        return None if isinstance(returns[0], ast.Constant) else returns[0]
    result = [
        value
        for value in returns
        if isinstance(value, (ast.ListComp, ast.GeneratorExp))
    ]
    if len(result) == 1:
        return result[0]
    return None


def _loop_bindings(function):
    """Literal tuple rows assigned by a generated-case loop."""
    for node in ast.walk(function):
        if not isinstance(node, ast.For) or not isinstance(
            node.target, (ast.Tuple, ast.List)
        ):
            continue
        sequence = node.iter
        if not isinstance(sequence, (ast.Tuple, ast.List)):
            continue
        names = [name.id for name in node.target.elts if isinstance(name, ast.Name)]
        if len(names) != len(node.target.elts):
            continue
        for row in sequence.elts:
            values = row.elts if isinstance(row, (ast.Tuple, ast.List)) else [row]
            if len(values) == len(names):
                for name, value in zip(names, values, strict=True):
                    tags = _constant_strings(value)
                    if len(tags) == 1:
                        yield name, tags[0]


def _generated_case_tags(function, path):
    """Read tag expressions appended to a local list by a case helper."""
    added = []
    appended = []
    bindings = dict(_loop_bindings(function))
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "cases"
            for target in node.targets
        ):
            added.extend(_local_tags(node.value, bindings))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cases"
            and node.func.attr == "append"
            and node.args
        ):
            appended.extend(_local_tags(node.args[0], bindings))
    if added:
        return added
    return appended


def _local_tags(expression, bindings):
    """Tags from expressions built without a runtime import."""
    found = []
    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"_ascend_case", "_gcu_case"}
            and len(node.args) > 1
        ):
            if isinstance(node.args[0], ast.Constant):
                tags = _constant_strings(node.args[0], bindings)
            else:
                tags = _constant_strings(node.args[1], bindings)
            if len(tags) == 1:
                found.append(tags[0])
            continue
    return list(dict.fromkeys(found))


def _constant_strings(expression, bindings=None):
    """Evaluate f-strings of loop literals used to name generated cases."""
    bindings = {} if bindings is None else bindings
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return [expression.value]
    if isinstance(expression, ast.Name) and isinstance(
        bindings.get(expression.id), str
    ):
        return [bindings[expression.id]]
    if isinstance(expression, ast.JoinedStr):
        parts = []
        for value in expression.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif (
                isinstance(value, ast.FormattedValue)
                and isinstance(value.value, ast.Name)
                and isinstance(bindings.get(value.value.id), str)
                and value.conversion == -1
                and value.format_spec is None
            ):
                parts.append(bindings[value.value.id])
            else:
                return []
        return ["".join(parts)]
    return []


def _literal_size(expression, bindings, path):
    """Count rows generated by a literal expression for known parameters."""
    if isinstance(expression, (ast.Tuple, ast.List)):
        if any(
            isinstance(item, ast.Starred)
            and isinstance(item.value, (ast.Tuple, ast.List))
            for item in expression.elts
        ):
            return sum(
                _literal_size(item.value, bindings, path)
                if isinstance(item, ast.Starred)
                else 1
                for item in expression.elts
            )
        return len(expression.elts)
    if isinstance(expression, ast.Name) and expression.id in bindings:
        return _literal_size(
            ast.Constant(value=bindings[expression.id]), bindings, path
        )
    if isinstance(expression, ast.Constant) and isinstance(expression.value, bool):
        return 0 if expression.value else 1
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        if expression.func.id == "tuple" and len(expression.args) == 1:
            return _literal_size(expression.args[0], bindings, path)
    raise InventoryError(f"{_location(path, expression)}: cannot count generated cases")


def _unique(cases, name):
    cases = sorted(cases)
    if not cases:
        raise InventoryError(f"the {name} inventory is empty")
    if len(cases) != len(set(cases)):
        duplicates = sorted({case for case in cases if cases.count(case) > 1})
        raise InventoryError(f"the {name} inventory repeats {duplicates}")
    return cases


def unit_files():
    """All unit files, including test subpackages such as tests/unit/bpu."""
    return _unique(
        (
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tests/unit").rglob("test_*.py")
            if path.is_file()
        ),
        "unit file",
    )


def _shared_tags(source):
    path, expression = source.table(SHARED_SDPA, "COMMON_CASES")
    return tuple(source.list(path, expression))


def _lane_rows(source):
    """The fields of every literal ``LaneRow`` in the SDPA table."""
    path, expression = source.table(VENDOR_SDPA, "SDPA_LANE_ROWS")
    if not isinstance(expression, ast.Dict):
        raise InventoryError("SDPA_LANE_ROWS must be a dictionary literal")
    rows = []
    for value in expression.values:
        if (
            not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Name)
            or value.func.id != "LaneRow"
        ):
            raise InventoryError("every SDPA lane must be a LaneRow literal")
        rows.append({keyword.arg: keyword.value for keyword in value.keywords})
    return path, rows


def _row_cases(source, row_name, case_name):
    """Tags from ``row.<case_name>`` on every row of a literal table."""
    path, rows = _lane_rows(source)
    if row_name != "SDPA_LANE_ROWS":
        raise InventoryError(f"unknown SDPA table {row_name}")
    result = set()
    shared_tags = _shared_tags(source) if case_name == "cases" else None
    if case_name == "cases":
        for fields in rows:
            if list(source.list(path, fields["shared_cases"])) != list(shared_tags):
                raise InventoryError("a SDPA lane does not carry every shared case")
        result.update(shared_tags)
        case_name = "additive_cases"
    for fields in rows:
        if case_name not in fields:
            raise InventoryError(f"a SDPA lane does not state {case_name}")
        value = fields[case_name]
        if (
            case_name == "additive_cases"
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
        ):
            function = source.function(path, value.func.id)
            returned = _literal_return(function)
            if returned is not None and not isinstance(returned, (ast.Tuple, ast.List)):
                value = returned
            if returned is not None and any(
                isinstance(node, ast.Name) and node.id == "cases"
                for node in ast.walk(returned)
            ):
                tags = _generated_case_tags(function, path)
                if tags:
                    result.update(tags)
                    continue
        result.update(source.list(path, value))
    if shared_tags is not None:
        result.update(shared_tags)
    return result


def _vendor_tags(source):
    shared = set(_shared_tags(source))
    _, rows = _lane_rows(source)
    result = _row_cases(source, "SDPA_LANE_ROWS", "additive_cases")
    for fields in rows:
        declared = set(source.list(VENDOR_SDPA, fields["shared_cases"]))
        if declared != shared:
            raise InventoryError("a SDPA lane does not carry every shared case")
        result.update(declared)
    return result


def vendor_sdpa_cases():
    source = Source((SHARED_SDPA, VENDOR_CASES, VENDOR_SDPA, VENDOR_LANES))
    function = next(
        node
        for node in source.trees[source.path(VENDOR_LANES)].body
        if isinstance(node, ast.FunctionDef) and node.name == "_case_tags"
    )
    returned = _literal_return(function)
    if not (
        isinstance(returned, ast.ListComp)
        and len(returned.generators) == 1
        and not returned.generators[0].ifs
        and isinstance(returned.elt, ast.Attribute)
        and returned.elt.attr == "tag"
        and isinstance(returned.elt.value, ast.Name)
        and isinstance(returned.generators[0].target, ast.Name)
        and returned.generators[0].target.id == returned.elt.value.id
        and isinstance(returned.generators[0].iter, ast.Attribute)
        and returned.generators[0].iter.attr == "cases"
        and isinstance(returned.generators[0].iter.value, ast.Name)
        and returned.generators[0].iter.value.id == "row"
    ):
        raise InventoryError("the vendor SDPA case IDs must iterate a row's cases")
    tags = _row_cases(source, "SDPA_LANE_ROWS", "cases")
    cases = [f"{VENDOR_FILE}::test_the_vendor_lane_holds[{tag}]" for tag in tags]
    cases.append(f"{VENDOR_FILE}::test_the_platform_names_the_lane_it_carries")
    return _unique(cases, "vendor SDPA")


def _route_envelope(source):
    path, expression = source.table(FLAGGEMS_SDPA, "SDPA_CASES")
    if not isinstance(expression, ast.Tuple):
        raise InventoryError("SDPA_CASES must be a tuple literal")
    return [source.case(path, "_case", case) for case in expression.elts]


def _parametrize_ids(source, path, function):
    ids = []
    for decorator in function.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
        ):
            continue
        if len(decorator.args) != 2:
            raise InventoryError(
                f"{_location(path, decorator)}: parametrize is not a table and argument"
            )
        argument = decorator.args[0]
        if (
            not isinstance(argument, ast.Constant)
            or not isinstance(argument.value, str)
            or "," in argument.value
        ):
            raise InventoryError(
                f"{_location(path, decorator)}: only one parameter is supported"
            )
        table = decorator.args[1]
        explicit = next(
            (keyword.value for keyword in decorator.keywords if keyword.arg == "ids"),
            None,
        )
        if (
            isinstance(table, ast.ListComp)
            and isinstance(table.elt, ast.Attribute)
            and table.elt.attr == "tag"
            and isinstance(table.elt.value, ast.Name)
            and table.elt.value.id == "SDPA_CASES"
        ):
            values = _route_envelope(source)
        elif explicit is not None and any(
            isinstance(item, ast.Call)
            and isinstance(item.func, (ast.Name, ast.Attribute))
            for item in ast.walk(table)
        ):
            # torch.device values have stable explicit IDs. Importing torch to
            # derive values would make this CPU-only plan hardware-dependent.
            count = source.size(path, table)
            if count < 0:
                if not isinstance(table, (ast.Name, ast.Dict)):
                    raise InventoryError(
                        f"{_location(path, table)}: runtime IDs have no table size"
                    )
                # A dict or a name bound to one has keys as its case identities.
                values = [""] * abs(count)
            else:
                values = [""] * count
        elif explicit is not None:
            # The explicit pytest ID is the case identity. Count any table that
            # can be resolved without importing it, so a boolean or torch.device
            # value does not make this hardware-independent inventory depend on
            # the local runtime.
            count = source.size(path, table)
            if count < 0:
                raise InventoryError(
                    f"{_location(path, table)}: explicit IDs have no literal table size"
                )
            values = [""] * count
        else:
            values = source.list(path, table)
        if (
            explicit is not None
            and isinstance(table, ast.Attribute)
            and isinstance(table.value, ast.Call)
            and isinstance(table.value.func, ast.Name)
            and table.value.func.id == "sorted"
        ):
            table = table.value.args[0]
            explicit = None
        if explicit is None:
            ids.append(values)
        else:
            templates = source.list(path, explicit)
            if len(templates) != len(values):
                raise InventoryError(f"{path.name}: parametrize IDs have wrong length")
            ids.append(
                [
                    template.format(Tag=value)
                    if "{Tag}" in template and value
                    else template
                    for template, value in zip(templates, values, strict=True)
                ]
            )
    return ids


def _file_nodeids(source, path):
    """All node IDs in one module, in pytest's collection order."""
    nodeids = []
    path = _path(path)
    relative = path.relative_to(ROOT).as_posix()
    for node in source.trees[path].body:
        blocks = []
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            blocks = [(relative, node)]
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            blocks = [(f"{relative}::{node.name}", member) for member in node.body]
        for prefix, function in blocks:
            if not isinstance(
                function, ast.FunctionDef
            ) or not function.name.startswith("test_"):
                continue
            suffixes = [function.name]
            for ids in reversed(_parametrize_ids(source, path, function)):
                suffixes = [
                    f"{suffix}[{case_id}]" for suffix in suffixes for case_id in ids
                ]
            nodeids.extend(f"{prefix}::{suffix}" for suffix in suffixes)
    return nodeids


def _flaggems_nodeids():
    """One inventory for both passes, independent of per-case markers."""
    source = Source((SHARED_SDPA, FLAGGEMS_CASES, FLAGGEMS_SDPA, *FLAGGEMS_FILES))
    nodeids = []
    for path in FLAGGEMS_FILES:
        if path == SDPA_FILE:
            continue
        nodeids.extend(_file_nodeids(source, path))
    prefix = SDPA_FILE
    for node in source.trees[source.path(SDPA_FILE)].body:
        if not isinstance(node, ast.ClassDef):
            continue
        for function in node.body:
            if not isinstance(
                function, ast.FunctionDef
            ) or not function.name.startswith("test_"):
                continue
            name = f"{prefix}::{node.name}::{function.name}"
            if function.name == "test_route_follows_the_configured_arm":
                tags = _route_envelope(source)
            elif function.name == "test_shared_sdpa_calls_hold_on_the_configured_route":
                tags = _shared_tags(source)
            else:
                tags = ()
            nodeids.extend(f"{name}[{tag}]" for tag in tags)
            if not tags:
                nodeids.append(name)
    return _unique(nodeids, "FlagGems conformance")


def flaggems_cases():
    """The one ordered case set run by both native and kernel passes."""
    return _flaggems_nodeids()


def flaggems_kernel_cases():
    """The compatibility alias for the same shared FlagGems case set."""
    return flaggems_cases()


_INVENTORIES = {
    "unit": unit_files,
    "sdpa": vendor_sdpa_cases,
    "flaggems": flaggems_cases,
    "flaggems_kernels": flaggems_kernel_cases,
}
INVENTORY_KINDS = {
    "unit_tests_per_file": "unit",
    "sdpa_lanes": "sdpa",
    "flaggems_conformance": "flaggems",
    "flaggems_kernel_conformance": "flaggems_kernels",
}


def inventory(kind):
    if kind not in _INVENTORIES:
        raise InventoryError(f"unknown inventory {kind!r}")
    return _unique(_INVENTORIES[kind](), kind)


def inventory_document(kind):
    """Return the cases and pytest-native IDs that alias an explicit case ID."""
    cases = inventory(kind)
    aliases = {}
    document = {"cases": cases, "aliases": aliases}
    if kind == "flaggems":
        prefix = (
            "tests/integration/flaggems/test_flaggems_device_identity.py::"
            "test_device_properties_accepts_the_registered_name"
        )
        aliases = {
            f"{prefix}[device]": [f"{prefix}[flagos]"],
            f"{prefix}[device-index]": [f"{prefix}[flagos:0]"],
        }
    if kind == "unit":
        document["group"] = "unit_tests_per_file"
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=_INVENTORIES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    document = inventory_document(args.kind)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"{args.kind}: {len(document['cases'])} cases -> {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InventoryError as error:
        print(f"::error::{error}", file=sys.stderr)
        raise SystemExit(1) from error
