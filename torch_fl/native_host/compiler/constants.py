# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate admitted immutable metadata expressions without executing host calls."""

import ast
import builtins
import inspect
import operator
import textwrap
import types


class NotConstant(Exception):
    pass


class Returned(BaseException):
    def __init__(self, value):
        self.value = value


_ALLOWED_EXCEPTIONS = {
    name: value
    for name, value in vars(__import__("builtins")).items()
    if isinstance(value, type) and issubclass(value, Exception)
}


def literal(value):
    return (
        value is None
        or type(value) in (bool, int, float, str)
        or (type(value) in (tuple, list) and all(literal(item) for item in value))
    )


class Evaluator:
    def __init__(self, modules):
        self.modules = modules
        self.active = set()
        self.dependencies = set()

    def call(self, function, args, kwargs):
        if (
            type(function) is not types.FunctionType
            or function.__module__ not in self.modules
        ):
            raise NotConstant()
        if function in self.active:
            raise NotConstant()
        if not all(literal(v) for v in (*args, *kwargs.values())):
            raise NotConstant()
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        self.dependencies.add(function.__module__)
        self.active.add(function)
        try:
            self.statements(node.body, dict(bound.arguments), function.__globals__)
        except Returned as result:
            if not literal(result.value):
                raise NotConstant()
            return result.value
        finally:
            self.active.remove(function)
        return None

    def attribute(self, value, name, default=None, has_default=False):
        if type(value) is types.ModuleType and value.__name__ in self.modules:
            self.dependencies.add(value.__name__)
            if name in vars(value):
                result = vars(value)[name]
                if (
                    type(result) in (types.ModuleType, type)
                    or type(result) is types.FunctionType
                ):
                    return result
                # The wheel cohort pins its version, while runtime global flags
                # and mutable configuration are not build-time constants.
                if name == "__version__" and type(result) is str:
                    return result
                raise NotConstant()
            if "__getattr__" in vars(value):
                raise NotConstant()
            if has_default:
                return default
            raise AttributeError(name)
        if type(value) is type and value.__module__ in self.modules:
            self.dependencies.add(value.__module__)
            try:
                result = inspect.getattr_static(value, name)
                if inspect.getattr_static(
                    type(result), "__get__", None
                ) is not None and not (
                    type(result) is types.FunctionType
                    or type(result) in (staticmethod, classmethod)
                    and type(result.__func__) is types.FunctionType
                ):
                    raise NotConstant()
                return result
            except AttributeError:
                if has_default:
                    return default
                raise
        raise NotConstant()

    def expression(self, node, local, global_):
        def ev(item):
            return self.expression(item, local, global_)

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in local:
                return local[node.id]
            value = global_.get(node.id)
            if type(value) is types.ModuleType and value.__name__ in self.modules:
                return value
            if type(value) is types.FunctionType and value.__module__ in self.modules:
                return value
            if node.id in _ALLOWED_EXCEPTIONS:
                return _ALLOWED_EXCEPTIONS[node.id]
            raise NotConstant()
        if isinstance(node, ast.Attribute):
            return self.attribute(ev(node.value), node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            return (
                tuple(ev(v) for v in node.elts)
                if isinstance(node, ast.Tuple)
                else [ev(v) for v in node.elts]
            )
        if isinstance(node, ast.Subscript):
            value, index = ev(node.value), ev(node.slice)
            if not literal(value) or not literal(index):
                raise NotConstant()
            return value[index]
        if isinstance(node, ast.BoolOp):
            value = ev(node.values[0])
            for item in node.values[1:]:
                if (isinstance(node.op, ast.And) and not value) or (
                    isinstance(node.op, ast.Or) and value
                ):
                    return value
                value = ev(item)
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not ev(node.operand)
        if isinstance(node, ast.Compare):
            a = ev(node.left)
            operations = {
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Is: operator.is_,
                ast.IsNot: operator.is_not,
            }
            for op, right in zip(node.ops, node.comparators):
                b = ev(right)
                if type(op) not in operations:
                    raise NotConstant()
                if not operations[type(op)](a, b):
                    return False
                a = b
            return True
        if isinstance(node, ast.Call):
            if any(k.arg is None for k in node.keywords):
                raise NotConstant()
            args = [ev(v) for v in node.args]
            kwargs = {k.arg: ev(k.value) for k in node.keywords}
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in (
                    "getattr",
                    "hasattr",
                    "int",
                    "str",
                    "bool",
                    "tuple",
                    "list",
                    "len",
                ) and (
                    name in local
                    or global_.get(name, getattr(builtins, name))
                    is not getattr(builtins, name)
                ):
                    raise NotConstant()
                if (
                    name in ("getattr", "hasattr")
                    and not kwargs
                    and 2 <= len(args) <= 3
                ):
                    if name == "getattr":
                        return self.attribute(
                            args[0],
                            args[1],
                            args[2] if len(args) == 3 else None,
                            len(args) == 3,
                        )
                    if len(args) != 2:
                        raise NotConstant()
                    try:
                        self.attribute(args[0], args[1])
                        return True
                    except AttributeError:
                        return False
                if (
                    name in ("int", "str", "bool", "tuple", "list", "len")
                    and all(literal(v) for v in args)
                    and not kwargs
                ):
                    return {
                        "int": int,
                        "str": str,
                        "bool": bool,
                        "tuple": tuple,
                        "list": list,
                        "len": len,
                    }[name](*args)
            if isinstance(node.func, ast.Attribute):
                value = ev(node.func.value)
                if (
                    type(value) is str
                    and node.func.attr in ("split", "startswith", "endswith")
                    and all(literal(v) for v in args)
                    and not kwargs
                ):
                    return getattr(value, node.func.attr)(*args)
            return self.call(ev(node.func), args, kwargs)
        raise NotConstant()

    def statements(self, nodes, local, global_):
        def ev(node):
            return self.expression(node, local, global_)

        for node in nodes:
            if isinstance(node, ast.Return):
                raise Returned(ev(node.value) if node.value else None)
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                local[node.targets[0].id] = ev(node.value)
            elif isinstance(node, ast.If):
                self.statements(
                    node.body if ev(node.test) else node.orelse, local, global_
                )
            elif isinstance(node, ast.Try) and not node.finalbody:
                try:
                    self.statements(node.body, local, global_)
                except NotConstant:
                    raise
                except Exception as error:
                    for handler in node.handlers:
                        if handler.type is None or isinstance(error, ev(handler.type)):
                            if handler.name:
                                raise NotConstant()
                            self.statements(handler.body, local, global_)
                            break
                    else:
                        raise
                else:
                    self.statements(node.orelse, local, global_)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    module = self.modules.get(name.name)
                    if module is None:
                        raise NotConstant()
                    self.dependencies.add(name.name)
                    if name.asname:
                        local[name.asname] = module
                    else:
                        local[name.name.split(".")[0]] = self.modules[
                            name.name.split(".")[0]
                        ]
            elif (
                isinstance(node, ast.Pass)
                or isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
            ):
                continue
            else:
                raise NotConstant()
