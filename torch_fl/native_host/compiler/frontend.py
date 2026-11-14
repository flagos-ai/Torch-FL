# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Transform installed host source without executing tensor examples."""

import ast
import builtins
import copy
import inspect
import textwrap
import typing
import collections.abc
from collections import Counter
import hashlib
import sys
import types

from .constants import Evaluator


def type_text(value):
    origin = typing.get_origin(value)
    arguments = typing.get_args(value)
    if origin in (list, tuple, collections.abc.Sequence):
        if origin is tuple and arguments and arguments[-1] is not Ellipsis:
            return "Tuple[" + ", ".join(type_text(a) for a in arguments) + "]"
        return "List[" + type_text(arguments[0]) + "]"
    if origin is typing.Union:
        parts = [type_text(a) for a in arguments if a is not type(None)]
        if type(None) in arguments:
            return "Optional[" + parts[0] + "]"
        if len(set(parts)) == 1:
            return parts[0]
        return "Union[" + ", ".join(parts) + "]"
    if value.__module__ == "builtins":
        return value.__name__
    if value.__module__ == "torch":
        return "torch." + value.__name__
    raise NotImplementedError(str(value))


def returns_on_all_paths(statements):
    if not statements:
        return False
    last = statements[-1]
    return (
        isinstance(last, (ast.Return, ast.Raise))
        or isinstance(last, ast.If)
        and returns_on_all_paths(last.body)
        and returns_on_all_paths(last.orelse)
    )


def annotated_list(type_name, values):
    return ast.Call(
        ast.parse("torch.jit.annotate").body[0].value,
        [
            ast.parse("List[" + type_name + "]").body[0].value,
            ast.List(values, ast.Load()),
        ],
        [],
    )


def annotated_dict(values):
    return ast.Call(
        ast.parse("torch.jit.annotate").body[0].value,
        [
            ast.parse("Dict[str, Any]").body[0].value,
            ast.Dict([ast.Constant(k) for k, v in values], [v for k, v in values]),
        ],
        [],
    )


class Frontend:
    def __init__(self, hosts, launcher):
        self.hosts = hosts
        self.launcher = launcher
        self.functions = {}
        self.names = {}
        self.busy = set()
        self.grids = []
        self.kernel_slots = []
        self.containers = {}
        self.tagged = {}
        self.counter = 0
        self.descriptor_constructors = {}
        self.fallback_functions = set()
        metadata_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if isinstance(module, types.ModuleType)
            and name.split(".", 1)[0] in ("torch", "triton", "flag_gems")
        }
        metadata_modules.update(hosts.originals)
        self.constants = Evaluator(metadata_modules)
        self.module_literals = {}
        self.module_initializers = {}

    def literal_global(self, function, name):
        module_name = function.__module__
        if module_name not in self.module_literals:
            module = self.hosts.originals.get(module_name)
            if module is None:
                raise NotImplementedError("untracked global provider")
            tree = ast.parse(inspect.getsource(module))
            values = {}
            initialized = set()
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target, expression = statement.targets[0], statement.value
                elif isinstance(statement, ast.AnnAssign):
                    target, expression = statement.target, statement.value
                else:
                    continue
                if isinstance(target, ast.Name):
                    initialized.add(target.id)
                    try:
                        value = ast.literal_eval(expression)
                    except (ValueError, TypeError):
                        continue
                    if value is None or type(value) in (bool, int, float, str):
                        values[target.id] = value

            def module_scope(node):
                yield node
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    for child in ast.iter_child_nodes(node):
                        yield from module_scope(child)

            writes = Counter(
                node.id
                for node in module_scope(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
            mutable = {
                name
                for node in ast.walk(tree)
                if isinstance(node, ast.Global)
                for name in node.names
            }
            self.module_literals[module_name] = {
                key: value
                for key, value in values.items()
                if writes[key] == 1 and key not in mutable
            }
            self.module_initializers[module_name] = {
                key for key in initialized if writes[key] == 1 and key not in mutable
            }
        values = self.module_literals[module_name]
        current = function.__globals__[name]
        if name in values and current == values[name]:
            return ast.Constant(values[name])
        if name in self.module_initializers[module_name] and type(current) in (
            bool,
            int,
            float,
            str,
        ):
            kind = type(current).__name__
            slot = self.launcher.register_global(module_name, name, kind)
            expression = ast.Call(
                ast.parse("torch.ops.flagos_script_host.global_" + kind).body[0].value,
                [ast.Constant(slot)],
                [],
            )
            expression._type = kind
            return expression
        raise NotImplementedError(
            "runtime global requires compiled object semantics: " + name
        )

    def transform(
        self,
        function,
        argument_types=None,
        signature=None,
        return_hint=None,
        argument_containers=None,
        definition=None,
    ):
        key = (
            function,
            tuple(sorted((argument_types or {}).items())),
            return_hint,
            repr(argument_containers),
            ast.dump(definition, include_attributes=False) if definition else None,
        )
        if key in self.names and self.names[key] in self.functions:
            return self.names[key]
        name = "host_" + str(self.counter)
        self.counter += 1
        self.names[key] = name
        node = (
            copy.deepcopy(definition)
            if definition is not None
            else ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        )
        node.name = name
        node.decorator_list = []
        if signature:
            header = ast.parse(signature).body[0]
            node.args = header.args
            node.returns = header.returns
        for a in node.args.args + node.args.kwonlyargs:
            if argument_types and argument_types.get(a.arg):
                a.annotation = ast.parse(argument_types[a.arg]).body[0].value
        if return_hint:
            node.returns = ast.parse(return_hint).body[0].value
        transformer = Transform(self, function, node)
        transformer.containers.update(argument_containers or {})
        if node.returns is None:
            inferred = {
                transformer.annotation(n.value)
                for n in ast.walk(node)
                if isinstance(n, ast.Return)
            }
            if len(inferred) == 1 and "torch.Tensor" in inferred:
                node.returns = ast.parse("torch.Tensor").body[0].value
                transformer.return_type = "torch.Tensor"
        original = copy.deepcopy(node)
        try:
            node = transformer.visit(node)
        except NotImplementedError:
            if definition is not None:
                raise
            self.fallback_functions.add(function)
            returns = [n.value for n in ast.walk(original) if isinstance(n, ast.Return)]
            result = return_hint or (
                ast.unparse(original.returns) if original.returns else None
            )
            if (
                not result
                and original.body
                and isinstance(original.body[-1], ast.Return)
            ):
                result = transformer.annotation(original.body[-1].value)
            if not result:
                if (
                    original.args.args
                    or original.args.kwonlyargs
                    or not returns
                    or not all(
                        isinstance(v, (ast.Compare, ast.BoolOp))
                        or isinstance(v, ast.Constant)
                        and type(v.value) is bool
                        for v in returns
                    )
                ):
                    raise
                result = "bool"
            if any(v == "torch.dtype" for v in (argument_types or {}).values()):
                raise
            slot = self.launcher.register_foreign(self.hosts.resolve(function), ())
            node = original
            call = ast.Call(
                ast.parse("torch.ops.flagos_source_native.foreign_value").body[0].value,
                [
                    ast.Constant(slot),
                    annotated_list(
                        "Any", [ast.Name(a.arg, ast.Load()) for a in node.args.args]
                    ),
                    annotated_dict(
                        [
                            (a.arg, ast.Name(a.arg, ast.Load()))
                            for a in node.args.kwonlyargs
                        ]
                    ),
                ],
                [],
            )
            node.body = [
                ast.Assign([ast.Name("_foreign", ast.Store())], call),
                ast.Assert(
                    ast.Call(
                        ast.parse("torch.jit.isinstance").body[0].value,
                        [
                            ast.Name("_foreign", ast.Load()),
                            ast.parse(result).body[0].value,
                        ],
                        [],
                    )
                ),
                ast.Return(ast.Name("_foreign", ast.Load())),
            ]
            node.returns = ast.parse(result).body[0].value
        returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        tagged = {}
        if returns and all(
            isinstance(r.value, ast.Tuple)
            and len(r.value.elts) == 2
            and isinstance(r.value.elts[0], ast.Constant)
            and type(r.value.elts[0].value) is str
            for r in returns
        ):
            tagged = {
                r.value.elts[0].value: transformer.annotation(r.value.elts[1])
                for r in returns
            }
            if all(tagged.values()) and len(set(tagged.values())) > 1:
                for r in returns:
                    r.value.elts[1] = ast.Call(
                        ast.parse("torch.jit.annotate").body[0].value,
                        [ast.Name("Any", ast.Load()), r.value.elts[1]],
                        [],
                    )
                node.returns = ast.parse("Tuple[str, Any]").body[0].value
                self.tagged[name] = tuple(sorted(tagged.items()))
        if (
            len(returns) == 1
            and isinstance(returns[0].value, ast.Tuple)
            and len(returns[0].value.elts) >= 2
        ):
            values = returns[0].value.elts
            if isinstance(values[0], ast.Name) and isinstance(values[1], ast.Name):
                inherited = transformer.tagged_values.get(values[1].id)
                if inherited and inherited[0] == values[0].id:
                    self.tagged[name] = inherited[1]
        if node.returns is None:
            annotations = {
                transformer.annotation(n.value)
                for n in ast.walk(node)
                if isinstance(n, ast.Return)
            }
            if None not in annotations and annotations:
                nullable = "None" in annotations
                concrete = {a for a in annotations if a != "None"}
                for item in list(concrete):
                    if item.startswith("Optional["):
                        concrete.remove(item)
                        concrete.add(item[9:-1])
                        nullable = True
                result = (
                    next(iter(concrete))
                    if len(concrete) == 1
                    else "None"
                    if not concrete
                    else None
                )
                if result and nullable and result != "None":
                    result = "Optional[" + result + "]"
                if result is None:
                    result = "Any"
                if result != "None" and not returns_on_all_paths(node.body):
                    if not result.startswith("Optional["):
                        result = "Optional[" + result + "]"
                    node.body.append(ast.Return(ast.Constant(None)))
                node.returns = ast.parse(result).body[0].value
        if node.args.kwonlyargs:
            required = [
                (a, d)
                for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if d is None
            ]
            optional = [
                (a, d)
                for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if d is not None
            ]
            optional_count = len(node.args.defaults)
            fixed = (
                node.args.args[:-optional_count] if optional_count else node.args.args
            )
            trailing = node.args.args[-optional_count:] if optional_count else []
            node.args.args = (
                fixed + [a for a, d in required] + trailing + [a for a, d in optional]
            )
            node.args.defaults.extend(d for a, d in optional)
            node.args.kwonlyargs = []
            node.args.kw_defaults = []
        self.functions[name] = ast.fix_missing_locations(node)
        containers = [
            transformer.container(n.value)
            for n in ast.walk(node)
            if isinstance(n, ast.Return)
        ]
        if containers and all(c == containers[0] for c in containers):
            self.containers[name] = containers[0]
        return name

    def descriptor_constructor(self, owner):
        if owner in self.descriptor_constructors:
            return self.descriptor_constructors[owner]
        name = "descriptor_ctor_" + str(self.counter)
        self.counter += 1
        self.descriptor_constructors[owner] = name
        signature = (
            "def "
            + name
            + '(base: torch.Tensor, shape: List[int], strides: List[int], block_shape: List[int], padding: str="zero") -> Any: pass'
        )
        node = ast.parse(signature).body[0]
        original = ast.parse(
            textwrap.dedent(inspect.getsource(owner.__post_init__))
        ).body[0]
        fields = ("base", "shape", "strides", "block_shape", "padding")
        slot = self.launcher.register_foreign(self.hosts.resolve(owner), ())
        factory = ast.Call(
            ast.parse("torch.ops.flagos_source_native.foreign_value").body[0].value,
            [
                ast.Constant(slot),
                annotated_list("Any", [ast.Name(n, ast.Load()) for n in fields]),
                annotated_dict([]),
            ],
            [],
        )

        class Flatten(ast.NodeTransformer):
            def visit_Attribute(inner, value):
                if (
                    isinstance(value.value, ast.Name)
                    and value.value.id == "self"
                    and value.attr in fields
                ):
                    return ast.Name(value.attr, value.ctx)
                return inner.generic_visit(value)

            def visit_Assert(inner, value):
                return ast.If(
                    ast.UnaryOp(ast.Not(), inner.visit(value.test)),
                    [ast.Return(copy.deepcopy(factory))],
                    [],
                )

        node.body = [Flatten().visit(n) for n in original.body]
        node.body.append(
            ast.Return(
                ast.Call(
                    ast.parse("torch.ops.flagos_script_host.descriptor").body[0].value,
                    [ast.Name(n, ast.Load()) for n in fields],
                    [],
                )
            )
        )
        transformer = Transform(self, owner.__post_init__, node)
        try:
            node = transformer.visit(node)
        except NotImplementedError:
            node.body = [ast.Return(factory)]
        self.functions[name] = ast.fix_missing_locations(node)
        return name

    def source(self, function, signature):
        name = self.transform(function, signature=signature)
        node = self.functions[name]
        text = "import torch\nimport math\nfrom typing import Any, List, Dict, Tuple, Optional\n\n"
        text += "def _reduce_mul(values: List[int], initial: int) -> int:\n    for value in values:\n        initial *= value\n    return initial\n\n"
        text += "def _bit_length(value: int) -> int:\n    value = abs(value)\n    result = 0\n    while value != 0:\n        value = value >> 1\n        result += 1\n    return result\n\n"
        text += "def _next_power_of_2(value: int) -> int:\n    result = 1\n    while result < value:\n        result = result * 2\n    return result\n\n"
        text += "\n\n".join(self.grids) + "\n\n"
        text += "\n\n".join(ast.unparse(n) for n in self.functions.values())
        arguments = [a.arg for a in node.args.args]
        parameters = ast.unparse(node.args)
        text += (
            "\n\nclass Host(torch.nn.Module):\n    def forward(_host_instance, "
            + parameters
            + ") -> torch.Tensor:\n        return "
            + name
            + "("
            + ", ".join(arguments)
            + ")\n"
        )
        return text


class Transform(ast.NodeTransformer):
    def __init__(self, frontend, function, node):
        self.frontend = frontend
        self.function = function
        self.root = node
        self.local_functions = {}
        self.globals = function.__globals__
        self.inputs = {a.arg for a in node.args.args + node.args.kwonlyargs}
        self.tensor_inputs = {
            a.arg
            for a in node.args.args
            if a.annotation and ast.unparse(a.annotation) == "torch.Tensor"
        }
        self.locals = set(self.inputs)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                self.locals.add(n.id)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n is not node:
                self.locals.add(n.name)
        self.overloads = {}
        self.types = {
            a.arg: ast.unparse(a.annotation)
            for a in node.args.args + node.args.kwonlyargs
            if a.annotation
        }
        defaults = (
            dict(
                zip(
                    [a.arg for a in node.args.args[-len(node.args.defaults) :]],
                    node.args.defaults,
                )
            )
            if node.args.defaults
            else {}
        )
        defaults.update(
            (a.arg, d)
            for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
            if d is not None
        )
        for a in node.args.args + node.args.kwonlyargs:
            default = defaults.get(a.arg)
            if (
                a.annotation is None
                and isinstance(default, ast.Constant)
                and type(default.value) in (bool, int, float, str)
            ):
                a.annotation = ast.Name(type(default.value).__name__, ast.Load())
            if (
                isinstance(default, ast.Constant)
                and default.value is None
                and a.annotation
                and not ast.unparse(a.annotation).startswith("Optional[")
            ):
                a.annotation = (
                    ast.parse("Optional[" + ast.unparse(a.annotation) + "]")
                    .body[0]
                    .value
                )
            if a.annotation:
                self.types[a.arg] = ast.unparse(a.annotation)

        self.grids = {}
        self.return_type = ast.unparse(node.returns) if node.returns else None
        self.containers = {}
        self.objects = {}
        self.grid_names = {
            n.func.slice.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Subscript)
            and isinstance(n.func.slice, ast.Name)
        }
        self.assignments = Counter(
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        )
        self.literal_values = {}
        self.defined = set(self.inputs)
        self.tagged_values = {}
        for a in node.args.args + node.args.kwonlyargs:
            default = defaults.get(a.arg)
            if (
                a.annotation is not None
                or not isinstance(default, ast.Constant)
                or default.value is not None
            ):
                continue
            candidates = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == a.arg for t in n.targets
                ):
                    candidates.add(self.annotation(n.value))
                if isinstance(n, ast.BinOp):
                    if isinstance(n.left, ast.Name) and n.left.id == a.arg:
                        candidates.add(self.annotation(n.right))
                    if isinstance(n.right, ast.Name) and n.right.id == a.arg:
                        candidates.add(self.annotation(n.left))
            candidates.discard(None)
            candidates.discard("None")
            if len(candidates) == 1:
                kind = next(iter(candidates))
                kind = (
                    kind if kind.startswith("Optional[") else "Optional[" + kind + "]"
                )
                a.annotation = ast.parse(kind).body[0].value
                self.types[a.arg] = kind
        self.usage_hints = dict(self.types)
        for _ in range(4):

            def known(n):
                if isinstance(n, ast.Name):
                    return self.usage_hints.get(n.id)
                return self.annotation(n)

            for n in ast.walk(node):
                if isinstance(n, ast.Call) and ast.unparse(n.func) == "triton.cdiv":
                    for a in n.args:
                        if isinstance(a, ast.Name):
                            self.usage_hints.setdefault(a.id, "int")
                if isinstance(n, ast.BinOp):
                    for target, other in ((n.left, n.right), (n.right, n.left)):
                        if isinstance(target, ast.Name) and known(other) in (
                            "int",
                            "float",
                        ):
                            self.usage_hints.setdefault(target.id, known(other))
                if isinstance(n, ast.Assign) and len(n.targets) == 1:
                    target = n.targets[0]
                    if isinstance(target, ast.Name) and known(n.value):
                        self.usage_hints.setdefault(target.id, known(n.value))
                    if isinstance(target, ast.Tuple) and isinstance(n.value, ast.Name):
                        members = [known(v) for v in target.elts]
                        if all(members):
                            self.usage_hints[n.value.id] = (
                                "Tuple[" + ", ".join(members) + "]"
                            )
        nullable = {
            n.left.id
            for n in ast.walk(node)
            if isinstance(n, ast.Compare)
            and isinstance(n.left, ast.Name)
            and any(
                isinstance(c, ast.Constant) and c.value is None for c in n.comparators
            )
        }
        for name in nullable:
            if name in self.usage_hints and not self.usage_hints[name].startswith(
                "Optional["
            ):
                self.usage_hints[name] = "Optional[" + self.usage_hints[name] + "]"

    def assert_type(self, name, kind):
        return ast.Assert(
            ast.Call(
                ast.parse("torch.jit.isinstance").body[0].value,
                [ast.Name(name, ast.Load()), ast.parse(kind).body[0].value],
                [],
            )
        )

    def visit_statements(self, items):
        result = []

        def append(value):
            if isinstance(value, list):
                for child in value:
                    append(child)
            elif value is not None:
                result.append(value)

        index = 0
        while index < len(items):
            if (
                index + 1 < len(items)
                and isinstance(items[index], ast.FunctionDef)
                and isinstance(items[index + 1], ast.Expr)
            ):
                block = ast.Module(body=items[index : index + 2], type_ignores=[])
                digest = hashlib.sha256(
                    (
                        self.function.__name__
                        + "\n"
                        + ast.dump(block, include_attributes=False)
                    ).encode()
                ).hexdigest()[:16]
                helper_name = "_host_closure_" + digest
                record = next(
                    (
                        r
                        for r in self.frontend.hosts.manifest["modules"]
                        .get(self.function.__module__, {})
                        .get("regions", [])
                        if r["name"] == helper_name
                    ),
                    None,
                )
                if record:
                    helper = getattr(
                        self.frontend.hosts.modules[self.function.__module__],
                        helper_name,
                    )
                    slot = self.frontend.launcher.register_foreign(helper, ())
                    call = ast.Call(
                        ast.parse("torch.ops.flagos_source_native.foreign_value")
                        .body[0]
                        .value,
                        [
                            ast.Constant(slot),
                            annotated_list(
                                "Any",
                                [ast.Name(n, ast.Load()) for n in record["inputs"]],
                            ),
                            annotated_dict([]),
                        ],
                        [],
                    )
                    result.append(ast.Expr(call))
                    index += 2
                    continue
            append(self.visit(items[index]))
            index += 1
        return result

    def container(self, node):
        if hasattr(node, "_container"):
            return node._container
        if isinstance(node, ast.Name):
            return self.containers.get(node.id)
        if isinstance(node, ast.Tuple):
            return tuple(self.container(n) for n in node.elts)
        if isinstance(node, ast.ListComp):
            return ("list", self.container(node.elt))
        if isinstance(node, ast.Subscript):
            owner = self.container(node.value)
            if isinstance(owner, tuple) and len(owner) == 2 and owner[0] == "list":
                return owner[1]
        if isinstance(node, ast.Call):
            label = ast.unparse(node.func)
            if label in self.frontend.containers:
                return self.frontend.containers[label]
            if label in ("tuple", "torch.Size"):
                return "tuple"
            if (
                isinstance(node.func, ast.Attribute)
                and self.annotation(node.func.value) == "torch.Tensor"
                and node.func.attr in ("size", "stride")
                and not node.args
            ):
                return "tuple"
        if (
            isinstance(node, ast.Attribute)
            and self.annotation(node.value) == "torch.Tensor"
            and node.attr == "shape"
        ):
            return "tuple"
        if isinstance(node, ast.IfExp):
            first, second = self.container(node.body), self.container(node.orelse)
            if first == second:
                return first
            if first == "tuple" and isinstance(node.orelse, ast.List):
                return first
        return None

    def kernel_value(self, node):
        value = self.visit(node)
        if self.container(value) == "tuple":
            return ast.Call(
                ast.parse("torch.ops.flagos_source_native.tuple_value").body[0].value,
                [value],
                [],
            )
        return value

    def annotation(self, node):
        if node is None:
            return "None"
        if hasattr(node, "_type"):
            return node._type
        if isinstance(node, ast.Name):
            return self.types.get(
                node.id, getattr(self, "usage_hints", {}).get(node.id)
            )
        if isinstance(node, ast.Constant):
            return {
                int: "int",
                float: "float",
                bool: "bool",
                str: "str",
                type(None): "None",
            }.get(type(node.value))
        if isinstance(node, ast.Attribute):
            try:
                if type(self.resolve(node)).__name__ == "dtype":
                    return "torch.dtype"
            except (AttributeError, ValueError):
                pass
            owner = self.annotation(node.value)
            if owner == "torch.Tensor":
                return {
                    "shape": "List[int]",
                    "ndim": "int",
                    "device": "torch.device",
                    "dtype": "torch.dtype",
                }.get(node.attr)
        if isinstance(node, ast.Subscript):
            owner = self.annotation(node.value)
            if owner and owner.startswith("List["):
                return owner if isinstance(node.slice, ast.Slice) else owner[5:-1]
            if owner == "torch.Tensor":
                return owner
        if isinstance(node, ast.BinOp):
            left, right = self.annotation(node.left), self.annotation(node.right)
            if left == right:
                return left
        if isinstance(node, ast.IfExp):
            left, right = self.annotation(node.body), self.annotation(node.orelse)
            if left == right:
                return left
        if isinstance(node, ast.ListComp):
            element = self.annotation(node.elt)
            if element:
                return "List[" + element + "]"
        if isinstance(node, ast.Compare):
            return "bool"
        if isinstance(node, ast.BoolOp):
            annotations = {self.annotation(n) for n in node.values}
            return annotations.pop() if len(annotations) == 1 else None
        if isinstance(node, (ast.Tuple, ast.List)):
            annotations = [self.annotation(n) for n in node.elts]
            if annotations and all(annotations):
                if isinstance(node, ast.Tuple):
                    return "Tuple[" + ", ".join(annotations) + "]"
                if len(set(annotations)) == 1:
                    return "List[" + annotations[0] + "]"
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in ("tuple", "list", "torch.Size"):
                return self.annotation(node.args[0])
            if (
                isinstance(node.func, ast.Attribute)
                and self.annotation(node.func.value) == "torch.Tensor"
            ):
                if node.func.attr in (
                    "is_complex",
                    "is_contiguous",
                    "is_conj",
                    "is_neg",
                    "is_floating_point",
                ):
                    return "bool"
                if node.func.attr in ("numel", "dim", "storage_offset", "element_size"):
                    return "int"
                if node.func.attr in ("size", "stride"):
                    return "int" if node.args else "List[int]"
                if node.func.attr in (
                    "contiguous",
                    "to",
                    "expand",
                    "new_empty",
                    "view",
                    "reshape",
                    "permute",
                    "squeeze",
                ):
                    return "torch.Tensor"
            if name in ("torch.empty", "torch.empty_like", "torch.tensor"):
                return "torch.Tensor"
            if name in ("torch.result_type", "torch.promote_types"):
                return "torch.dtype"
            if name in (
                "torch.ops.flagos_source_native.broadcast_shapes",
                "torch.broadcast_shapes",
            ):
                return "List[int]"
            if name in (
                "torch.view_as_real",
                "torch.view_as_complex",
                "torch.real",
                "torch.imag",
                "torch.complex",
            ):
                return "torch.Tensor"
            if name in (
                "_reduce_mul",
                "_next_power_of_2",
                "int",
                "len",
                "sum",
                "math.ceil",
                "triton.cdiv",
            ):
                return "int"
            if name in ("min", "max"):
                if len(node.args) == 1:
                    owner = self.annotation(node.args[0])
                    return owner[5:-1] if owner and owner.startswith("List[") else None
                kinds = {self.annotation(a) for a in node.args}
                return next(iter(kinds)) if len(kinds) == 1 else None
            if name in ("float", "math.sqrt"):
                return "float"
            if name == "torch.jit.annotate":
                return ast.unparse(node.args[0])
            if name == "torch.device":
                return "torch.device"
            if isinstance(node.func, ast.Name) and node.func.id in self.globals:
                value = self.globals[node.func.id]
                if inspect.isfunction(value):
                    result = inspect.signature(value).return_annotation
                    if result is not inspect.Signature.empty:
                        try:
                            return type_text(result)
                        except (AttributeError, NotImplementedError):
                            pass
            if name in self.frontend.functions:
                function = self.frontend.functions[name]
                if function.returns:
                    return ast.unparse(function.returns)
        return None

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Tuple):
            kinds = [self.annotation(n) for n in node.iter.elts]
            if all(kinds) and len(set(kinds)) > 1:
                if any(
                    isinstance(n, (ast.Break, ast.Continue)) for n in ast.walk(node)
                ):
                    raise NotImplementedError("heterogeneous loop control")
                result = []
                names = []
                for index, value in enumerate(node.iter.elts):
                    name = (
                        "_tuple_iteration_" + str(len(self.defined)) + "_" + str(index)
                    )
                    names.append(name)
                    result.extend(
                        self.visit_statements(
                            [ast.Assign([ast.Name(name, ast.Store())], value)]
                        )
                    )
                    self.locals.add(name)

                class Rename(ast.NodeTransformer):
                    def visit_Name(inner, value):
                        return (
                            ast.copy_location(ast.Name(inner.new, value.ctx), value)
                            if value.id == node.target.id
                            else value
                        )

                for name in names:
                    rename = Rename()
                    rename.new = name
                    result.extend(
                        self.visit_statements(
                            [rename.visit(copy.deepcopy(n)) for n in node.body]
                        )
                    )
                result.extend(self.visit_statements(node.orelse))
                return result
        if isinstance(node.target, ast.Name) and isinstance(
            node.iter, (ast.Tuple, ast.List)
        ):
            annotations = {self.annotation(n) for n in node.iter.elts}
            if len(annotations) == 1 and None not in annotations:
                self.types[node.target.id] = annotations.pop()
        if (
            isinstance(node.iter, ast.Call)
            and ast.unparse(node.iter.func) == "enumerate"
        ):
            owner = self.annotation(node.iter.args[0])
            if (
                owner
                and owner.startswith("List[")
                and isinstance(node.target, ast.Tuple)
            ):
                self.types[node.target.elts[0].id] = "int"
                self.types[node.target.elts[1].id] = owner[5:-1]
        return self.generic_visit(node)

    def visit_ListComp(self, node):
        for generator in node.generators:
            owner = self.annotation(generator.iter)
            if (
                owner
                and owner.startswith("List[")
                and isinstance(generator.target, ast.Name)
            ):
                self.types[generator.target.id] = owner[5:-1]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node is not self.root:
            if node not in self.root.body:
                raise NotImplementedError("conditional local helper definition")
            if node.decorator_list or node.args.vararg or node.args.kwarg:
                raise NotImplementedError("decorated or variadic local helper")
            if any(
                isinstance(n, (ast.Global, ast.Nonlocal, ast.FunctionDef))
                for statement in node.body
                for n in ast.walk(statement)
            ):
                raise NotImplementedError("local helper with nested scope")
            bound = {
                argument.arg
                for argument in ast.walk(node.args)
                if isinstance(argument, ast.arg)
            }
            bound.update(
                n.id
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            )
            reads = {
                n.id
                for statement in node.body
                for n in ast.walk(statement)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            if node.name in reads or (reads - bound) & self.locals:
                raise NotImplementedError("recursive or captured local helper")
            defaults = node.args.defaults + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default in defaults:
                try:
                    value = ast.literal_eval(default)
                except (ValueError, TypeError):
                    raise NotImplementedError(
                        "nonliteral local helper default"
                    ) from None
                if value is not None and type(value) not in (bool, int, float, str):
                    raise NotImplementedError("local helper container default")
            uses = {
                id(n)
                for n in ast.walk(self.root)
                if isinstance(n, ast.Name) and n.id == node.name
            }
            calls = {
                id(n.func)
                for n in ast.walk(self.root)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == node.name
            }
            if uses != calls:
                raise NotImplementedError("escaping local helper")
            self.local_functions[node.name] = copy.deepcopy(node)
            return None
        node.args = self.visit(node.args)
        node.body = self.visit_statements(node.body)
        for statement in ast.walk(node):
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.List)
                and not statement.value.elts
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                name = statement.targets[0].id
                elements = [
                    self.annotation(call.args[0])
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and ast.unparse(call.func.value) == name
                    and call.func.attr == "append"
                ]
                for call in ast.walk(node):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and ast.unparse(call.func.value) == name
                        and call.func.attr == "extend"
                        and isinstance(call.args[0], ast.List)
                    ):
                        elements.extend(self.annotation(e) for e in call.args[0].elts)
                elements = set(elements)
                elements.discard(None)
                if len(elements) > 1:
                    elements = {"Any"}
                if len(elements) == 1:
                    element = elements.pop()
                    self.types[name] = "List[" + element + "]"
                    statement.value = annotated_list(element, [])
                    if element == "Any":
                        for call in ast.walk(node):
                            if (
                                isinstance(call, ast.Call)
                                and isinstance(call.func, ast.Attribute)
                                and ast.unparse(call.func.value) == name
                                and call.func.attr == "extend"
                                and isinstance(call.args[0], ast.List)
                            ):
                                call.args[0] = annotated_list("Any", call.args[0].elts)
        return node

    def visit_Tuple(self, node):
        if any(isinstance(value, ast.Starred) for value in node.elts):
            elements = []
            kinds = set()
            for value in node.elts:
                if isinstance(value, ast.Starred):
                    kind = self.annotation(value.value)
                    if kind is None or not kind.startswith("List["):
                        raise NotImplementedError(
                            "starred tuple requires a typed sequence"
                        )
                    kinds.add(kind[5:-1])
                    elements.append((True, value.value))
                else:
                    kinds.add(self.annotation(value))
                    elements.append((False, value))
            if len(kinds) != 1 or None in kinds:
                raise NotImplementedError("heterogeneous starred tuple")
            kind = kinds.pop()
            result = annotated_list(kind, [])
            for starred, value in elements:
                part = self.visit(value)
                if not starred:
                    part = annotated_list(kind, [part])
                result = ast.BinOp(result, ast.Add(), part)
            result._container = "tuple"
            result._type = "List[" + kind + "]"
            return result
        if not node.elts and isinstance(node.ctx, ast.Load):
            result = annotated_list("int", [])
            result._container = "tuple"
            return result
        return self.generic_visit(node)

    def visit_Return(self, node):
        if isinstance(node.value, ast.Call):
            node.value._return_hint = self.return_type
        node = self.generic_visit(node)
        kind = self.annotation(node.value)
        if node.value is not None and kind:
            node.value._type = kind
        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.objects:
            value = getattr(self.objects[node.value.id], node.attr, None)
            if type(value) in (str, int, float, bool):
                return ast.Constant(value)
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "dtype"
            and self.annotation(node.value.value) == "torch.Tensor"
        ):
            if node.attr == "itemsize":
                return self.visit(
                    ast.Call(
                        ast.Attribute(node.value.value, "element_size", ast.Load()),
                        [],
                        [],
                    )
                )
            if node.attr == "is_floating_point":
                return self.visit(
                    ast.Call(
                        ast.Attribute(
                            node.value.value, "is_floating_point", ast.Load()
                        ),
                        [],
                        [],
                    )
                )
        hint = getattr(node, "_return_hint", None)
        if (
            hint
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            function = self.resolve(node.value.func)
            if inspect.isfunction(function) and function.__module__.startswith(
                "flag_gems"
            ):
                slot = self.frontend.launcher.register_foreign_projection(
                    self.frontend.hosts.resolve(function), node.attr
                )
                name = "host_projection_" + str(slot)
                helper = ast.parse(
                    "def "
                    + name
                    + "(args: List[Any], kwargs: Dict[str, Any]) -> "
                    + hint
                    + ":\n    value = torch.ops.flagos_source_native.foreign_value("
                    + str(slot)
                    + ", args, kwargs)\n    assert torch.jit.isinstance(value, "
                    + hint
                    + ")\n    return value"
                ).body[0]
                self.frontend.functions[name] = helper
                return ast.Call(
                    ast.Name(name, ast.Load()),
                    [
                        annotated_list("Any", [self.visit(a) for a in node.value.args]),
                        annotated_dict(
                            [(k.arg, self.visit(k.value)) for k in node.value.keywords]
                        ),
                    ],
                    [],
                )
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        left, right = self.annotation(node.left), self.annotation(node.right)
        if isinstance(node.left, ast.Call) and right in ("int", "float"):
            node.left._return_hint = right
        if isinstance(node.right, ast.Call) and left in ("int", "float"):
            node.right._return_hint = left
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            try:
                first, second = ast.literal_eval(left), ast.literal_eval(right)
            except (ValueError, TypeError):
                pass
            else:
                return ast.Constant(
                    first in second
                    if isinstance(node.ops[0], ast.In)
                    else first not in second
                )
            node.left = left
            node.comparators = [right]
        if (
            len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(node.left, ast.Name)
            and not getattr(node, "_optional_guarded", False)
        ):
            kind = self.annotation(node.left)
            other = node.comparators[0]
            if (
                kind
                and kind.startswith("Optional[List[")
                and isinstance(other, (ast.List, ast.Tuple))
                and not other.elts
            ):
                equal = isinstance(node.ops[0], ast.Eq)
                if isinstance(other, ast.Tuple):
                    return ast.Constant(not equal)
                node._optional_guarded = True
                test = ast.Compare(
                    copy.deepcopy(node.left),
                    [ast.IsNot() if equal else ast.Is()],
                    [ast.Constant(None)],
                )
                return ast.BoolOp(
                    ast.And() if equal else ast.Or(), [test, self.generic_visit(node)]
                )
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and self.annotation(node.target) == "int":
            self.literal_values.pop(node.target.id, None)
            return self.visit(
                ast.Assign(
                    [node.target],
                    ast.BinOp(
                        ast.Name(node.target.id, ast.Load()), node.op, node.value
                    ),
                )
            )
        return self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        raise NotImplementedError("generator expression")

    def visit_ImportFrom(self, node):
        if node.level or node.module not in self.frontend.hosts.originals:
            raise NotImplementedError("local import " + node.module)
        module = self.frontend.hosts.originals[node.module]
        for entry in node.names:
            if entry.name == "*":
                raise NotImplementedError("wildcard local import")
            self.objects[entry.asname or entry.name] = getattr(module, entry.name)
        return None

    def visit_With(self, node):
        # The native launch boundary installs a current-device guard.
        if all(
            ast.unparse(i.context_expr.func) == "torch_device_fn.device"
            for i in node.items
        ):
            return self.visit_statements(node.body)
        raise NotImplementedError("context manager " + ast.unparse(node))

    def resolve(self, node):
        if isinstance(node, ast.Name):
            return self.objects.get(
                node.id, self.globals.get(node.id, getattr(builtins, node.id, None))
            )
        if isinstance(node, ast.Attribute):
            return getattr(self.resolve(node.value), node.attr)
        raise ValueError(ast.unparse(node))

    def visit_arg(self, node):
        if node.annotation and ast.unparse(node.annotation) == "torch.Size":
            node.annotation = ast.parse("List[int]").body[0].value
        elif (
            isinstance(node.annotation, ast.Name) and node.annotation.id in self.globals
        ):
            node.annotation = (
                ast.parse(type_text(self.globals[node.annotation.id])).body[0].value
            )
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.literal_values:
            return ast.Constant(self.literal_values[node.id])
        if node.id not in self.locals and node.id in self.globals:
            value = self.globals[node.id]
            if value is None or isinstance(value, (int, float, bool, str)):
                return self.frontend.literal_global(self.function, node.id)
            if type(value).__name__ == "dtype":
                return ast.parse(str(value)).body[0].value
        return node

    def visit_Expr(self, node):
        if (
            isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "logger.debug"
        ):
            return None
        return self.generic_visit(node)

    def visit_Try(self, node):
        if (
            len(node.handlers) == 1
            and len(node.handlers[0].body) == 1
            and isinstance(node.handlers[0].body[0], ast.Raise)
        ):
            handler = node.handlers[0]
            raised = handler.body[0].exc
            if (
                handler.name
                and isinstance(raised, ast.Call)
                and ast.unparse(raised.func) == ast.unparse(handler.type)
                and len(raised.args) == 1
                and ast.unparse(raised.args[0]) == "str(" + handler.name + ")"
            ):
                return self.visit_statements(node.body)
        digest = hashlib.sha256(
            (
                self.function.__name__ + "\n" + ast.dump(node, include_attributes=False)
            ).encode()
        ).hexdigest()[:16]
        region_name = "_host_region_" + digest
        record = next(
            (
                r
                for r in self.frontend.hosts.manifest["modules"]
                .get(self.function.__module__, {})
                .get("regions", [])
                if r["name"] == region_name
            ),
            None,
        )
        if record and len(record["outputs"]) == 1:
            output = record["outputs"][0]
            assignment = next(
                (
                    statement
                    for statement in node.body
                    if isinstance(statement, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == output
                        for t in statement.targets
                    )
                ),
                None,
            )
            kind = self.annotation(assignment.value) if assignment else None
            if (
                assignment
                and isinstance(assignment.value, ast.Call)
                and ast.unparse(assignment.value.func) == "list"
                and kind
                and kind.startswith("Optional[List[")
            ):
                kind = kind[9:-1]
            if kind and kind.startswith("List["):
                helper = getattr(
                    self.frontend.hosts.modules[self.function.__module__], region_name
                )
                slot = self.frontend.launcher.register_foreign(helper, ())
                temporary = "_region_" + digest
                values = annotated_list(
                    "Any", [ast.Name(name, ast.Load()) for name in record["inputs"]]
                )
                call = ast.Call(
                    ast.parse("torch.ops.flagos_source_native.foreign_value")
                    .body[0]
                    .value,
                    [ast.Constant(slot), values, annotated_dict([])],
                    [],
                )
                original_input_kind = (
                    self.annotation(assignment.value.args[0])
                    if isinstance(assignment.value, ast.Call) and assignment.value.args
                    else None
                )
                self.types[output] = kind
                self.defined.add(output)
                foreign = [
                    ast.Assign([ast.Name(temporary, ast.Store())], call),
                    ast.Assert(
                        ast.Call(
                            ast.parse("torch.jit.isinstance").body[0].value,
                            [
                                ast.Name(temporary, ast.Load()),
                                ast.parse(kind).body[0].value,
                            ],
                            [],
                        )
                    ),
                    ast.Assign(
                        [ast.Name(output, ast.Store())], ast.Name(temporary, ast.Load())
                    ),
                ]
                expression = assignment.value
                if (
                    len(record["outputs"]) == 1
                    and isinstance(expression, ast.Call)
                    and ast.unparse(expression.func) == "list"
                    and len(expression.args) == 1
                    and isinstance(expression.args[0], ast.Name)
                ):
                    source = expression.args[0]
                    source_type = original_input_kind
                    # The original helper remains responsible for exceptional coercions.
                    if source_type == kind:
                        return [
                            ast.Assign(
                                [ast.Name(output, ast.Store())],
                                ast.Call(ast.Name("list", ast.Load()), [source], []),
                            )
                        ]
                    if source_type == "Optional[" + kind + "]":
                        return [
                            ast.If(
                                ast.Compare(
                                    copy.deepcopy(source),
                                    [ast.Is()],
                                    [ast.Constant(None)],
                                ),
                                foreign,
                                [
                                    ast.Assign(
                                        [ast.Name(output, ast.Store())],
                                        ast.Call(
                                            ast.Name("list", ast.Load()), [source], []
                                        ),
                                    )
                                ],
                            )
                        ]
                return foreign
        # Catch-and-rethrow with the same exception type/message changes no data flow.
        if (
            len(node.handlers) == 1
            and len(node.handlers[0].body) == 1
            and isinstance(node.handlers[0].body[0], ast.Raise)
        ):
            handler = node.handlers[0]
            raised = handler.body[0].exc
            if (
                handler.name
                and isinstance(raised, ast.Call)
                and ast.unparse(raised.func) == ast.unparse(handler.type)
                and len(raised.args) == 1
                and ast.unparse(raised.args[0]) == "str(" + handler.name + ")"
            ):
                return self.visit_statements(node.body)
        raise NotImplementedError(
            "exception handler in "
            + self.function.__module__
            + "."
            + self.function.__qualname__
        )

    def visit_If(self, node):
        if isinstance(node.test, ast.Call):
            node.test._return_hint = "bool"
        test = self.visit(node.test)
        if isinstance(test, ast.Constant) and isinstance(test.value, bool):
            return self.visit_statements(node.body if test.value else node.orelse)
        node.test = test
        narrowed = None
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.left, ast.Name)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            name = test.left.id
            annotation = self.types.get(name, "")
            if annotation.startswith("Optional[") and isinstance(
                test.ops[0], (ast.Is, ast.IsNot)
            ):
                narrowed = (name, annotation[9:-1], isinstance(test.ops[0], ast.Is))
        before = {
            key: dict(getattr(self, key))
            for key in (
                "grids",
                "types",
                "containers",
                "literal_values",
                "tagged_values",
            )
        }
        defined = set(self.defined)
        bool_name = (
            test.id
            if isinstance(test, ast.Name) and self.types.get(test.id) == "bool"
            else None
        )
        narrowed_tags = []
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Name)
            and isinstance(test.comparators[0], ast.Constant)
        ):
            tag = test.comparators[0].value
            for payload, (mode, variants) in self.tagged_values.items():
                choices = dict(variants)
                if mode == test.left.id and tag in choices:
                    narrowed_tags.append((payload, mode, choices, tag))

        def statements(items):
            return self.visit_statements(items)

        if narrowed and not narrowed[2]:
            self.types[narrowed[0]] = narrowed[1]
        if bool_name:
            self.literal_values[bool_name] = True
        body_assertions = []
        for payload, mode, choices, tag in narrowed_tags:
            self.types[payload] = choices[tag]
            body_assertions.append(self.assert_type(payload, choices[tag]))
        node.body = statements(node.body)
        node.body = body_assertions + node.body
        first = {key: dict(getattr(self, key)) for key in before}
        first_defined = set(self.defined)
        for key, value in before.items():
            setattr(self, key, dict(value))
        self.defined = set(defined)
        if narrowed and narrowed[2]:
            self.types[narrowed[0]] = narrowed[1]
        if bool_name:
            self.literal_values[bool_name] = False
        else_assertions = []
        for payload, mode, choices, tag in narrowed_tags:
            remaining = {k: v for k, v in choices.items() if k != tag}
            self.tagged_values[payload] = (mode, tuple(sorted(remaining.items())))
            if len(set(remaining.values())) == 1:
                kind = next(iter(remaining.values()))
                self.types[payload] = kind
                else_assertions.append(self.assert_type(payload, kind))
        node.orelse = statements(node.orelse)
        node.orelse = else_assertions + node.orelse
        second = {key: dict(getattr(self, key)) for key in before}
        second_defined = set(self.defined)
        body_ends = bool(node.body) and isinstance(
            node.body[-1], (ast.Return, ast.Raise)
        )
        else_ends = bool(node.orelse) and isinstance(
            node.orelse[-1], (ast.Return, ast.Raise)
        )
        for key in before:
            if body_ends and not else_ends:
                merged = second[key]
            elif else_ends and not body_ends:
                merged = first[key]
            else:
                merged = {}
                for name, value in first[key].items():
                    other = second[key].get(name)
                    same = (
                        ast.dump(value) == ast.dump(other)
                        if isinstance(value, ast.AST) and isinstance(other, ast.AST)
                        else value == other
                    )
                    if same:
                        merged[name] = value
                    elif key == "types" and value and other:
                        if value == "Optional[" + other + "]":
                            merged[name] = value
                        elif other == "Optional[" + value + "]":
                            merged[name] = other
            setattr(self, key, merged)
        self.defined = (
            second_defined
            if body_ends and not else_ends
            else first_defined
            if else_ends and not body_ends
            else first_defined & second_defined
        )
        if (
            narrowed
            and narrowed[2]
            and node.body
            and isinstance(node.body[-1], (ast.Return, ast.Raise))
        ):
            self.types[narrowed[0]] = narrowed[1]
        return node

    def visit_BoolOp(self, node):
        previous = self.types
        self.types = dict(previous)
        try:
            return self.visit_boolean_values(node)
        finally:
            self.types = previous

    def visit_boolean_values(self, node):
        values = []
        for original in node.values:
            value = self.visit(original)
            if isinstance(value, ast.Constant) and type(value.value) is bool:
                stops = (
                    (not value.value) if isinstance(node.op, ast.And) else value.value
                )
                if stops:
                    if all(
                        isinstance(v, ast.Constant)
                        or isinstance(v, ast.Name)
                        and v.id in self.defined
                        for v in values
                    ):
                        return value
                    values.append(value)
                    break
                if not values:
                    continue
                if all(self.annotation(v) == "bool" for v in values):
                    continue
            values.append(value)
            if (
                isinstance(value, ast.Compare)
                and len(value.ops) == 1
                and isinstance(value.left, ast.Name)
                and isinstance(value.comparators[0], ast.Constant)
                and value.comparators[0].value is None
            ):
                kind = self.types.get(value.left.id, "")
                if kind.startswith("Optional[") and (
                    isinstance(node.op, ast.Or)
                    and isinstance(value.ops[0], ast.Is)
                    or isinstance(node.op, ast.And)
                    and isinstance(value.ops[0], ast.IsNot)
                ):
                    self.types[value.left.id] = kind[9:-1]
        if not values:
            return ast.Constant(isinstance(node.op, ast.And))
        if len(values) == 1:
            return values[0]
        node.values = values
        if all(
            isinstance(n, ast.Constant) and type(n.value) is bool for n in node.values
        ):
            return ast.Constant(
                all(n.value for n in node.values)
                if isinstance(node.op, ast.And)
                else any(n.value for n in node.values)
            )
        if len(node.values) == 2:
            a, b = node.values
            kind = self.annotation(a)
            if kind and kind.startswith("List["):
                if isinstance(b, ast.Tuple):
                    b = ast.List(b.elts, ast.Load())
                if not isinstance(a, ast.Name):
                    raise NotImplementedError(
                        "repeated evaluation in boolean value expression"
                    )
                return (
                    ast.IfExp(copy.deepcopy(a), a, b)
                    if isinstance(node.op, ast.Or)
                    else ast.IfExp(copy.deepcopy(a), b, a)
                )
        return node

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Call):
            node.operand._return_hint = "bool"
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node

    def visit_Assign(self, node):
        value = node.value
        if (
            isinstance(node.targets[0], ast.Name)
            and isinstance(value, ast.Call)
            and ast.unparse(value.func) == "type"
            and len(value.args) == 1
            and self.annotation(value.args[0]) == "torch.Tensor"
        ):
            self.objects[node.targets[0].id] = self.globals.get(
                "torch", __import__("torch")
            ).Tensor
            return None
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(value, ast.Call)
        ):
            value._return_hint = self.usage_hints.get(node.targets[0].id)
        if (
            isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.grid_names
        ):
            self.grids[node.targets[0].id] = value
            return None
        if isinstance(value, ast.Attribute) and isinstance(node.targets[0], ast.Name):
            try:
                constant = self.resolve(value)
            except (AttributeError, ValueError):
                constant = None
            if (
                constant is not None
                and hasattr(constant, "run")
                and hasattr(constant, "arg_names")
            ):
                self.objects[node.targets[0].id] = constant
                return None
        if isinstance(value, ast.Lambda):
            self.grids[node.targets[0].id] = value
            return None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "instantiate"
        ):
            owner = self.resolve(value.func.value)
            slot = self.frontend.launcher.register_pointwise(
                self.frontend.hosts.resolve(owner)
            )
            target = node.targets[0].id
            self.overloads[target] = (
                slot,
                target + "_rank",
                owner.fx.num_output_tensors(),
            )
            return ast.Assign(
                [ast.Name(target + "_rank", ast.Store())], self.visit(value.args[0])
            )
        node = self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            variants = self.frontend.tagged.get(node.value.func.id)
            targets = node.targets[0].elts
            if (
                variants
                and len(targets) >= 2
                and all(isinstance(n, ast.Name) for n in targets[:2])
            ):
                self.tagged_values[targets[1].id] = (targets[0].id, variants)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined.add(target.id)
                self.literal_values.pop(target.id, None)
                if self.assignments[target.id] == 1 and isinstance(
                    node.value, ast.Constant
                ):
                    self.literal_values[target.id] = node.value.value
        annotation = self.annotation(node.value)
        container = self.container(node.value)
        if len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and container is not None:
                self.containers[node.targets[0].id] = container
            elif isinstance(node.targets[0], ast.Tuple) and isinstance(
                container, tuple
            ):
                for target, kind in zip(node.targets[0].elts, container):
                    if isinstance(target, ast.Name):
                        self.containers[target.id] = kind
        if annotation and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                self.types[node.targets[0].id] = annotation
            elif (
                isinstance(node.targets[0], (ast.Tuple, ast.List))
                and annotation == "List[int]"
            ):
                for element in node.targets[0].elts:
                    if isinstance(element, ast.Name):
                        self.types[element.id] = "int"
            elif isinstance(
                node.targets[0], (ast.Tuple, ast.List)
            ) and annotation.startswith("Tuple["):
                members = ast.parse(annotation).body[0].value.slice.elts
                for element, member in zip(node.targets[0].elts, members):
                    if isinstance(element, ast.Name):
                        self.types[element.id] = ast.unparse(member)
        return node

    def visit_Call(self, node):
        label = ast.unparse(node.func)
        if isinstance(node.func, ast.Name) and node.func.id in self.local_functions:
            definition = self.local_functions[node.func.id]
            parameters = [argument.arg for argument in definition.args.args]
            kinds = {
                p: self.annotation(value) for p, value in zip(parameters, node.args)
            }
            kinds.update({k.arg: self.annotation(k.value) for k in node.keywords})
            if any(kind is None for kind in kinds.values()):
                raise NotImplementedError("untyped local helper argument")
            name = self.frontend.transform(
                self.function,
                kinds,
                definition=definition,
            )
            return ast.Call(
                ast.Name(name, ast.Load()),
                [],
                [
                    ast.keyword(p, self.visit(value))
                    for p, value in zip(parameters, node.args)
                ]
                + [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords],
            )
        try:
            function = self.resolve(node.func)
            if (
                inspect.isfunction(function)
                and function.__module__ in self.frontend.hosts.originals
            ):
                values = [ast.literal_eval(v) for v in node.args]
                options = {
                    k.arg: ast.literal_eval(k.value)
                    for k in node.keywords
                    if k.arg is not None
                }
                if len(options) == len(node.keywords):
                    result = self.frontend.constants.call(function, values, options)
                    if result is None or type(result) in (bool, int, float, str):
                        return ast.Constant(result)
        except (Exception,):
            pass
        if (
            label == "hasattr"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Constant)
        ):
            try:
                owner = self.resolve(node.args[0])
            except (ValueError, AttributeError):
                owner = None
            if inspect.ismodule(owner):
                return ast.Constant(hasattr(owner, node.args[1].value))
        if label == "torch.are_deterministic_algorithms_enabled":
            return ast.Call(
                ast.parse("torch.ops.flagos_script_host.deterministic").body[0].value,
                [],
                [],
            )

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "data_ptr"
            and self.annotation(node.func.value) == "torch.Tensor"
        ):
            return ast.Call(
                ast.parse("torch.ops.flagos_script_host.data_ptr").body[0].value,
                [self.visit(node.func.value)],
                [],
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "bit_length":
            return ast.Call(
                ast.Name("_bit_length", ast.Load()), [self.visit(node.func.value)], []
            )
        if label in ("min", "max"):
            concrete = {self.annotation(a) for a in node.args} - {None}
            if len(concrete) == 1:
                for a in node.args:
                    if isinstance(a, (ast.Call, ast.Attribute)):
                        a._return_hint = next(iter(concrete))
        if label == "triton.next_power_of_2":
            return ast.Call(
                ast.Name("_next_power_of_2", ast.Load()),
                [self.visit(a) for a in node.args],
                [],
            )
        if (
            label == "torch.tensor"
            and node.args
            and isinstance(node.args[0], ast.List)
            and not node.args[0].elts
        ):
            node.args[0] = annotated_list("float", [])
        if label == "sorted" and node.keywords:
            raise NotImplementedError("sorted with key")
        if label == "triton.cdiv":
            return self.visit(GridCapture([]).visit(node))
        if label in ("functools.reduce", "reduce"):
            reducer = node.args[0]
            multiply = ast.unparse(reducer) == "operator.mul"
            if (
                isinstance(reducer, ast.Lambda)
                and len(reducer.args.args) == 2
                and isinstance(reducer.body, ast.BinOp)
                and isinstance(reducer.body.op, ast.Mult)
            ):
                multiply = (
                    isinstance(reducer.body.left, ast.Name)
                    and isinstance(reducer.body.right, ast.Name)
                    and [reducer.body.left.id, reducer.body.right.id]
                    == [a.arg for a in reducer.args.args]
                )
            if multiply:
                return ast.Call(
                    ast.Name("_reduce_mul", ast.Load()),
                    [self.visit(n) for n in node.args[1:]],
                    [],
                )
        if label == "str" and self.annotation(node.args[0]) == "torch.dtype":
            return ast.Call(
                ast.parse("torch.ops.flagos_source_native.dtype_str").body[0].value,
                [self.visit(node.args[0])],
                [],
            )
        if isinstance(node.func, ast.Subscript):
            original_kernel = self.resolve(node.func.value)
            if original_kernel is None:
                raise NotImplementedError(
                    "dynamic kernel " + ast.unparse(node.func.value)
                )
            kernel = self.frontend.hosts.resolve(original_kernel)
            slot = self.frontend.launcher.register_script_kernel(kernel)
            grid = self.grids.get(ast.unparse(node.func.slice), node.func.slice)
            args = annotated_list("Any", [self.kernel_value(a) for a in node.args])
            kwargs = annotated_dict(
                [(k.arg, self.kernel_value(k.value)) for k in node.keywords]
            )
            if isinstance(grid, ast.Tuple):
                return ast.Call(
                    ast.parse("torch.ops.flagos_source_native.launch").body[0].value,
                    [
                        ast.Constant(slot),
                        args,
                        kwargs,
                        ast.List([self.visit(a) for a in grid.elts], ast.Load()),
                    ],
                    [],
                )
            if not isinstance(grid, ast.Lambda):
                raise NotImplementedError("grid " + ast.unparse(grid))
            index = len(self.frontend.grids)
            captures = sorted(
                {n.id for n in ast.walk(grid.body) if isinstance(n, ast.Name)}
                - {a.arg for a in grid.args.args}
                - {"triton", "min", "max", "int", "len"}
            )
            body = GridCapture(captures, grid.args.args[0].arg).visit(
                copy.deepcopy(grid.body)
            )
            if isinstance(body, ast.Tuple):
                body = ast.List(body.elts, ast.Load())
            self.frontend.grids.append(
                "def grid_"
                + str(index)
                + "(meta: Dict[str, int], captures: List[int]) -> List[int]:\n    return "
                + ast.unparse(ast.fix_missing_locations(body))
            )
            native_id = len(self.frontend.kernel_slots)
            self.frontend.kernel_slots.append(slot)
            return ast.Call(
                ast.parse("torch.ops.flagos_script_host.launch").body[0].value,
                [
                    ast.Constant(native_id),
                    args,
                    kwargs,
                    ast.Constant(index),
                    ast.List([ast.Name(n, ast.Load()) for n in captures], ast.Load()),
                ],
                [],
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_tensor":
            try:
                owner = self.resolve(node.func.value)
            except (ValueError, AttributeError):
                owner = None
            if (
                isinstance(owner, type)
                and owner.__module__ == "triton.tools.tensor_descriptor"
                and owner.__name__ == "TensorDescriptor"
            ):
                function = getattr(owner, "from_tensor")
                parameters = list(inspect.signature(function).parameters)
                kinds = {p: self.annotation(v) for p, v in zip(parameters, node.args)}
                kinds.update({k.arg: self.annotation(k.value) for k in node.keywords})
                target = self.frontend.transform(
                    function, {k: v for k, v in kinds.items() if v}, return_hint="Any"
                )
                return ast.Call(
                    ast.Name(target, ast.Load()),
                    [],
                    [
                        ast.keyword(p, self.visit(v))
                        for p, v in zip(parameters, node.args)
                    ]
                    + [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords],
                )
        try:
            constructor = self.resolve(node.func)
        except (ValueError, AttributeError):
            constructor = None
        if (
            isinstance(constructor, type)
            and constructor.__module__ == "triton.tools.tensor_descriptor"
            and constructor.__name__ == "TensorDescriptor"
        ):
            name = self.frontend.descriptor_constructor(constructor)
            return ast.Call(
                ast.Name(name, ast.Load()),
                [self.visit(v) for v in node.args],
                [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords],
            )
        if label == "isinstance":
            annotation = self.annotation(node.args[0])
            if annotation and (
                annotation.startswith("List[")
                or annotation.startswith("Optional[List[")
            ):
                alternatives = (
                    node.args[1].elts
                    if isinstance(node.args[1], ast.Tuple)
                    else [node.args[1]]
                )
                targets = [self.resolve(t) for t in alternatives]
                if all(
                    t in (int, float, bool, str, tuple)
                    or getattr(t, "__name__", None) == "Tensor"
                    for t in targets
                ):
                    return ast.Constant(False)
            if annotation in ("torch.Tensor", "int", "float", "bool", "str"):
                declared = self.resolve(ast.parse(annotation).body[0].value)
                alternatives = (
                    node.args[1].elts
                    if isinstance(node.args[1], ast.Tuple)
                    else [node.args[1]]
                )
                return ast.Constant(
                    any(issubclass(declared, self.resolve(t)) for t in alternatives)
                )
        if label in ("tuple", "torch.Size"):
            value = self.visit(node.args[0])
            value._container = "tuple"
            return value
        if label == "torch._C._is_alias_of":
            node.func = (
                ast.parse("torch.ops.flagos_source_native.is_alias").body[0].value
            )
            return self.generic_visit(node)
        if label == "torch.broadcast_shapes":
            return ast.Call(
                ast.parse("torch.ops.flagos_source_native.broadcast_shapes")
                .body[0]
                .value,
                [annotated_list("List[int]", [self.visit(a) for a in node.args])],
                [],
            )
        if isinstance(node.func, ast.Name) and node.func.id in self.overloads:
            slot, rank, count = self.overloads[node.func.id]
            kwargs = {k.arg: k.value for k in node.keywords}
            return ast.Call(
                ast.parse("torch.ops.flagos_source_native.pointwise_launch")
                .body[0]
                .value,
                [
                    ast.Constant(slot),
                    ast.Name(rank, ast.Load()),
                    annotated_list("Any", [self.visit(a) for a in node.args]),
                    annotated_list(
                        "torch.Tensor",
                        [self.visit(kwargs["out" + str(i)]) for i in range(count)],
                    ),
                ],
                [],
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "redispatch":
            function = self.resolve(node.func)
            prefix = []
            remaining = []
            for item in node.args:
                if (
                    not remaining
                    and isinstance(item, ast.Name)
                    and item.id not in self.locals
                ):
                    prefix.append(self.resolve(item))
                else:
                    remaining.append(self.visit(item))
            slot = self.frontend.launcher.register_foreign(function, tuple(prefix))
            return ast.Call(
                ast.parse("torch.ops.flagos_source_native.foreign_tensor")
                .body[0]
                .value,
                [
                    ast.Constant(slot),
                    annotated_list("Any", remaining),
                    annotated_dict(
                        [(k.arg, self.visit(k.value)) for k in node.keywords]
                    ),
                ],
                [],
            )
        if isinstance(node.func, ast.Name) and node.func.id in self.globals:
            function = self.globals[node.func.id]
            if type(function).__name__ == "PointwiseDynamicFunction":
                slot = self.frontend.launcher.register_pointwise(
                    self.frontend.hosts.resolve(function)
                )
                call = ast.Call(
                    ast.parse("torch.ops.flagos_source_native.pointwise_apply")
                    .body[0]
                    .value,
                    [
                        ast.Constant(slot),
                        annotated_list("Any", [self.visit(a) for a in node.args]),
                        annotated_dict(
                            [(k.arg, self.visit(k.value)) for k in node.keywords]
                        ),
                    ],
                    [],
                )
                if function.fx.num_output_tensors() != 1:
                    raise NotImplementedError("multiple pointwise outputs")
                result = ast.Subscript(call, ast.Constant(0), ast.Load())
                result._type = "torch.Tensor"
                return result
            if (
                inspect.isfunction(function)
                and function.__module__ in self.frontend.hosts.originals
            ):
                parameters = list(inspect.signature(function).parameters)
                types = {p: self.annotation(a) for p, a in zip(parameters, node.args)}
                types.update({k.arg: self.annotation(k.value) for k in node.keywords})
                containers = {
                    p: self.container(a) for p, a in zip(parameters, node.args)
                }
                containers.update(
                    {k.arg: self.container(k.value) for k in node.keywords}
                )
                node.func = ast.Name(
                    self.frontend.transform(
                        function,
                        {k: v for k, v in types.items() if v},
                        return_hint=getattr(node, "_return_hint", None),
                        argument_containers=containers,
                    ),
                    ast.Load(),
                )
                node.keywords = [
                    ast.keyword(p, a) for p, a in zip(parameters, node.args)
                ] + node.keywords
                node.args = []
        return self.generic_visit(node)


class GridCapture(ast.NodeTransformer):
    def __init__(self, names, meta="meta"):
        self.names = names
        self.meta = meta

    def visit_Name(self, node):
        if node.id == self.meta:
            return ast.Name("meta", ast.Load())
        if node.id in self.names:
            return ast.Subscript(
                ast.Name("captures", ast.Load()),
                ast.Constant(self.names.index(node.id)),
                ast.Load(),
            )
        return node

    def visit_Call(self, node):
        if ast.unparse(node.func) == "triton.cdiv":
            a, b = map(self.visit, node.args)
            return ast.BinOp(
                ast.BinOp(ast.BinOp(a, ast.Add(), b), ast.Sub(), ast.Constant(1)),
                ast.FloorDiv(),
                copy.deepcopy(b),
            )
        return self.generic_visit(node)
