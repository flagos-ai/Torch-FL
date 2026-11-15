# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep scalar rebinding types distinct before entering the typed frontend.

TorchScript associates a type with a local name, unlike Python's bindings. For
type-changing scalars, assign fresh names and put the continuation in each
branch. All branches remain in the generated program; no examples are executed.
"""

import ast
import builtins
import copy

NUMERIC = {"bool", "int", "float"}
UNKNOWN = {"unknown"}


def annotation_types(node):
    if node is None:
        return UNKNOWN
    label = ast.unparse(node)
    if label in NUMERIC:
        return {label}
    if label.startswith("Optional[") and label[9:-1] in NUMERIC:
        return {label[9:-1], "None"}
    return {"other"}


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(self.names.get(node.id, node.id), node.ctx), node
        )


class ScalarBindings:
    def __init__(self, node, globals_):
        self.globals = globals_
        self.used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self.used.update(a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg))
        self.used.update(
            n.name
            for n in ast.walk(node)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        )
        self.shadowed = {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        } | {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
        self.shadowed.update(
            n.name
            for n in ast.walk(node)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        self.shadowed.update(
            n.asname or n.name.split(".")[0]
            for n in ast.walk(node)
            if isinstance(n, ast.alias)
        )
        self.used.update(self.shadowed)
        for item in ast.walk(node):
            if isinstance(item, (ast.Global, ast.Nonlocal)):
                self.used.update(item.names)
            elif isinstance(item, ast.ExceptHandler) and item.name:
                self.used.add(item.name)
        self.types = {
            a.arg: annotation_types(a.annotation)
            for a in node.args.args + node.args.kwonlyargs
        }
        self.assignments = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ]
        for _ in range(len(self.assignments) + 1):
            changed = False
            for statement in self.assignments:
                name = statement.targets[0].id
                old = self.types.get(name, set())
                new = old | self.kind(statement.value)
                if old != new:
                    self.types[name] = new
                    changed = True
            if not changed:
                break
        self.targets = {
            name for name, kinds in self.types.items() if len(kinds & NUMERIC) > 1
        }
        self.counter = 0
        # Limit source expansion, not supported runtime shapes or branches.
        self.limit = max(256, sum(1 for _ in ast.walk(node)) * 8)
        self.emitted = 0

    def builtin(self, node):
        if not isinstance(node, ast.Name) or node.id in self.shadowed:
            return None
        value = self.globals.get(node.id, getattr(builtins, node.id, None))
        return next(
            (name for name in NUMERIC if value is getattr(builtins, name)), None
        )

    def kind(self, node):
        if isinstance(node, ast.Constant):
            return {"None" if node.value is None else type(node.value).__name__}
        if isinstance(node, ast.Name):
            return self.types.get(node.id, UNKNOWN)
        if isinstance(node, ast.Call):
            kind = self.builtin(node.func)
            if kind:
                return {kind}
        if isinstance(node, ast.IfExp):
            return self.kind(node.body) | self.kind(node.orelse)
        if isinstance(node, ast.Compare) or (
            isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        ):
            return {"bool"}
        return UNKNOWN

    def writes(self, node):
        return {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
        } & self.targets

    def validate(self, node):
        if not self.targets:
            return
        if any(self.types[name] - (NUMERIC | {"None"}) for name in self.targets):
            raise NotImplementedError("scalar rebinding with an unknown value type")
        for item in ast.walk(node):
            if isinstance(item, (ast.Import, ast.ImportFrom)) and any(
                entry.name == "*"
                or (entry.asname or entry.name.split(".")[0]) in self.targets
                for entry in item.names
            ):
                raise NotImplementedError("scalar rebinding through an import")
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and item is not node
                and item.name in self.targets
            ):
                raise NotImplementedError("scalar rebinding through a definition")
            if isinstance(item, ast.IfExp) and len(self.kind(item) & NUMERIC) > 1:
                raise NotImplementedError("mixed scalar conditional expression")
            if (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name)
                and item.func.id in ("locals", "vars", "eval", "exec")
            ):
                raise NotImplementedError(
                    "scalar rebinding with namespace introspection"
                )
            if isinstance(item, (ast.Global, ast.Nonlocal)) and self.targets & set(
                item.names
            ):
                raise NotImplementedError("scalar rebinding of a shared binding")
            if (
                isinstance(
                    item,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.Lambda,
                        ast.GeneratorExp,
                        ast.ClassDef,
                    ),
                )
                and item is not node
            ):
                if self.targets & {
                    n.id for n in ast.walk(item) if isinstance(n, ast.Name)
                }:
                    raise NotImplementedError(
                        "scalar rebinding captured by a nested scope"
                    )
            if isinstance(item, ast.comprehension) and self.writes(item.target):
                raise NotImplementedError("scalar rebinding in a comprehension scope")
            if isinstance(item, ast.ExceptHandler) and item.name in self.targets:
                raise NotImplementedError("scalar rebinding in an exception handler")
            if isinstance(item, ast.Match):
                raise NotImplementedError("scalar rebinding with pattern matching")
            if isinstance(
                item,
                (ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith),
            ) and self.writes(item):
                raise NotImplementedError(
                    "scalar rebinding across a loop or protected scope"
                )
            if isinstance(
                item, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.Delete)
            ) and self.writes(item):
                raise NotImplementedError("unsupported scalar binding assignment")
            if isinstance(item, ast.Assign) and self.writes(item):
                if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                    raise NotImplementedError("scalar rebinding through destructuring")
                if (
                    isinstance(item.value, ast.Call)
                    and self.builtin(item.value.func) == "int"
                ):
                    if not item.value.args or self.kind(item.value.args[0]) - {
                        "bool",
                        "int",
                    }:
                        raise NotImplementedError(
                            "scalar rebinding needs general Python integer conversion"
                        )

    def fresh(self):
        while True:
            name = "_host_binding_" + str(self.counter)
            self.counter += 1
            if name not in self.used:
                self.used.add(name)
                return name

    def expression(self, node, names):
        return Rename(names).visit(copy.deepcopy(node))

    def block(self, statements, names):
        result = []
        for index, original in enumerate(statements):
            self.emitted += sum(1 for _ in ast.walk(original))
            if self.emitted > self.limit:
                raise NotImplementedError(
                    "scalar continuation expansion exceeds source budget"
                )
            if isinstance(original, ast.If) and self.writes(original):
                continuation = statements[index + 1 :]
                result.append(
                    ast.If(
                        self.expression(original.test, names),
                        self.block(original.body + continuation, dict(names)),
                        self.block(original.orelse + continuation, dict(names)),
                    )
                )
                return result
            if isinstance(original, ast.Assign) and self.writes(original):
                value = self.expression(original.value, names)
                logical = original.targets[0].id
                physical = self.fresh()
                names[logical] = physical
                result.append(ast.Assign([ast.Name(physical, ast.Store())], value))
            else:
                result.append(self.expression(original, names))
            if isinstance(original, (ast.Return, ast.Raise)):
                return result
        return result


def preserve_scalar_bindings(node, globals_):
    """Return an independent AST or decline before any source is executed."""
    normalizer = ScalarBindings(node, globals_)
    normalizer.validate(node)
    if not normalizer.targets:
        return node
    result = copy.deepcopy(node)
    result.body = normalizer.block(node.body, {})
    return ast.fix_missing_locations(result)
