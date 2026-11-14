# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Lower original metadata-only descriptor hooks to current-call C++ code."""

import ast
import inspect
import json
import textwrap


def compile_hook(function, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    if len(tree.args.args) != 1 or tree.args.vararg or tree.args.kwarg:
        raise NotImplementedError("hook signature")
    parameter = tree.args.args[0].arg
    locals_ = set()
    writes = []

    def argument(node):
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == parameter
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            raise NotImplementedError("hook argument")
        return node.slice.value

    def expression(node):
        if isinstance(node, ast.Name) and node.id in locals_:
            return "local_" + node.id
        if isinstance(node, ast.Subscript):
            return "values.at(" + json.dumps(argument(node)) + ")"
        if isinstance(node, ast.Constant) and type(node.value) in (bool, int):
            value = (
                ("true" if node.value else "false")
                if type(node.value) is bool
                else "int64_t(" + str(node.value) + ")"
            )
            return "c10::IValue(" + value + ")"
        if isinstance(node, ast.List):
            return (
                "c10::IValue(c10::List<int64_t>{"
                + ", ".join("(" + expression(v) + ").toInt()" for v in node.elts)
                + "})"
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return "c10::IValue(!HookTruth(" + expression(node.operand) + "))"
        raise NotImplementedError("hook expression " + ast.dump(node))

    def statements(nodes, depth):
        lines = []
        for node in nodes:
            prefix = "  " * depth
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = expression(node.value)
                if isinstance(target, ast.Name):
                    declaration = "" if target.id in locals_ else "c10::IValue "
                    locals_.add(target.id)
                    lines.append(
                        prefix
                        + declaration
                        + "local_"
                        + target.id
                        + " = "
                        + value
                        + ";"
                    )
                elif isinstance(target, ast.Attribute) and target.attr == "block_shape":
                    key = argument(target.value)
                    writes.append(key)
                    lines.append(
                        prefix
                        + "flagos::native_host::Descriptor(values.at("
                        + json.dumps(key)
                        + ')).insert_or_assign("block_shape", '
                        + value
                        + ");"
                    )
                else:
                    raise NotImplementedError("hook assignment")
            elif isinstance(node, ast.If):
                before = set(locals_)
                lines.append(prefix + "if (HookTruth(" + expression(node.test) + ")) {")
                lines.extend(statements(node.body, depth + 1))
                if locals_ != before:
                    raise NotImplementedError("hook branch-local bindings")
                lines.append(prefix + "} else {")
                lines.extend(statements(node.orelse, depth + 1))
                if locals_ != before:
                    raise NotImplementedError("hook branch-local bindings")
                lines.append(prefix + "}")
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            else:
                raise NotImplementedError("hook statement " + type(node).__name__)
        return lines

    body = statements(tree.body, 1)
    if not writes:
        raise NotImplementedError("no descriptor mutation")
    source = (
        "void "
        + name
        + "(const flagos::native_host::NamedArguments& values) {\n"
        + "\n".join(body)
        + "\n}"
    )
    return source, sorted(set(writes))
