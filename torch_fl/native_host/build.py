# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build-time source compilation experiment; never imported by the runtime."""

import argparse
import ast
import copy
import dataclasses
from enum import Enum
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import types
import textwrap


def module_name(name):
    return "m_" + hashlib.sha256(name.encode()).hexdigest()[:16]


def function_identity(function):
    code = function.__code__
    return f"{function.__module__}:{function.__qualname__}:{Path(code.co_filename).name}:{code.co_firstlineno}"


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def metadata_regions(definition):
    """Export straight-line exception regions without changing their semantics."""
    regions = []
    if not isinstance(definition, ast.FunctionDef):
        return regions

    def scope_nodes(node):
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                yield child
            else:
                yield from scope_nodes(child)

    nodes = list(scope_nodes(definition))
    locals_ = {
        node.id
        for node in nodes
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    locals_.update(
        argument.arg
        for argument in ast.walk(definition.args)
        if isinstance(argument, ast.arg)
    )
    locals_.update(
        node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    )
    for region in nodes:
        if not isinstance(region, ast.Try) or region.orelse or region.finalbody:
            continue
        if not all(isinstance(statement, ast.Assign) for statement in region.body):
            continue
        if any(
            isinstance(
                node,
                (
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.Lambda,
                    ast.NamedExpr,
                ),
            )
            for node in ast.walk(region)
        ):
            continue
        if not all(
            len(handler.body) == 1 and isinstance(handler.body[0], ast.Raise)
            for handler in region.handlers
        ):
            continue
        written = set()
        inputs = set()
        for statement in region.body:
            inputs.update(
                node.id
                for node in ast.walk(statement.value)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in locals_
                and node.id not in written
            )
            written.update(
                node.id
                for target in statement.targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
        if not written:
            continue
        handler_reads = {
            node.id
            for handler in region.handlers
            for node in ast.walk(handler)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in locals_
        }
        if not handler_reads.issubset(inputs):
            continue
        digest = hashlib.sha256(
            (
                definition.name + "\n" + ast.dump(region, include_attributes=False)
            ).encode()
        ).hexdigest()[:16]
        name = "_host_region_" + digest
        inputs, outputs = sorted(inputs), sorted(written)
        function = ast.parse(f"def {name}({', '.join(inputs)}): pass").body[0]
        result = [ast.Name(output, ast.Load()) for output in outputs]
        function.body = [
            copy.deepcopy(region),
            ast.Return(
                result[0] if len(result) == 1 else ast.Tuple(result, ast.Load())
            ),
        ]
        regions.append((function, {"name": name, "inputs": inputs, "outputs": outputs}))
    return regions


def continuation_regions(definition):
    """Keep guarded prefixes native while an exception-bearing tail stays Python."""
    if not isinstance(definition, ast.FunctionDef):
        return []
    nodes = list(ast.walk(definition))
    scoped = (
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.NamedExpr,
        ast.Delete,
        ast.Match,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
    )
    if any(isinstance(n, scoped) for n in nodes) or any(
        isinstance(n, ast.FunctionDef) and n is not definition for n in nodes
    ):
        return []
    if any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"locals", "globals", "vars", "eval", "exec", "dir"}
        for n in nodes
    ):
        return []

    def written(items):
        result = set()
        for item in items:
            for node in ast.walk(item):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    result.add(node.id)
                elif isinstance(node, ast.alias):
                    result.add(node.asname or node.name.split(".")[0])
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    result.add(node.name)
        return result

    arguments = {a.arg for a in ast.walk(definition.args) if isinstance(a, ast.arg)}
    # A small existing region can keep the remainder native. Do not replace it
    # with a coarser continuation that moves all subsequent launches to Cython.
    metadata = {
        ast.dump(helper.body[0], include_attributes=False)
        for helper, _ in metadata_regions(definition)
    }
    known = set(arguments)
    prefix_writes = set()
    guarded = False
    regions = []
    for index, statement in enumerate(definition.body):
        if (
            isinstance(statement, ast.Try)
            and guarded
            and ast.dump(statement, include_attributes=False) not in metadata
        ):
            tail = ast.Module(body=definition.body[index:], type_ignores=[])
            reads = {
                n.id
                for n in ast.walk(tail)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            # Passing a conditionally bound local eagerly could introduce an
            # UnboundLocalError on a path that never reads it in the original.
            if reads & (prefix_writes - known):
                return regions
            inputs = sorted(reads & known)
            digest = hashlib.sha256(
                (
                    definition.name + "\n" + ast.dump(tail, include_attributes=False)
                ).encode()
            ).hexdigest()[:16]
            name = "_host_continuation_" + digest
            helper = ast.parse(f"def {name}({', '.join(inputs)}): pass").body[0]
            helper.body = copy.deepcopy(tail.body)
            regions.append(
                (
                    helper,
                    {
                        "name": name,
                        "inputs": inputs,
                        "outputs": [],
                        "kind": "continuation",
                    },
                )
            )
        prefix_writes.update(written([statement]))
        if isinstance(statement, ast.Assign):
            known.update(written(statement.targets))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            known.update(written([statement.target]))
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            known.update(written([statement]))
        if isinstance(statement, ast.If) and any(
            isinstance(n, ast.Return) for n in ast.walk(statement)
        ):
            guarded = True
    return regions


def closure_regions(definition):
    """Export local callbacks handed directly to a consumer, with live captures."""
    if not isinstance(definition, ast.FunctionDef):
        return []

    def outer_nodes(node):
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            yield from outer_nodes(child)

    outer = list(outer_nodes(definition))
    locals_ = {
        n.id for n in outer if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    locals_.update(a.arg for a in ast.walk(definition.args) if isinstance(a, ast.arg))
    regions = []
    for parent in outer:
        for _, statements in ast.iter_fields(parent):
            if not isinstance(statements, list):
                continue
            for first, second in zip(statements, statements[1:]):
                if not isinstance(first, ast.FunctionDef) or not isinstance(
                    second, ast.Expr
                ):
                    continue
                if not isinstance(second.value, ast.Call):
                    continue
                uses = [
                    n
                    for n in ast.walk(second)
                    if isinstance(n, ast.Name) and n.id == first.name
                ]
                outside_uses = [
                    n for n in outer if isinstance(n, ast.Name) and n.id == first.name
                ]
                if len(uses) != 1 or outside_uses != uses:
                    continue
                scoped = (
                    ast.Nonlocal,
                    ast.Global,
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.ClassDef,
                )
                if any(isinstance(n, scoped) for n in ast.walk(first)) or any(
                    isinstance(n, ast.FunctionDef) and n is not first
                    for n in ast.walk(first)
                ):
                    continue
                bound = {first.name}
                bound.update(
                    a.arg for a in ast.walk(first.args) if isinstance(a, ast.arg)
                )
                bound.update(
                    n.id
                    for n in ast.walk(first)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                )
                block = ast.Module(body=[first, second], type_ignores=[])

                def reads(node):
                    return {
                        n.id
                        for n in ast.walk(node)
                        if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Load)
                        and n.id in locals_
                    }

                body_reads = set().union(*(reads(n) for n in first.body)) - bound
                evaluated_reads = reads(first.args) | reads(second)
                for decorator in first.decorator_list:
                    evaluated_reads.update(reads(decorator))
                inputs = sorted(body_reads | (evaluated_reads - {first.name}))
                digest = hashlib.sha256(
                    (
                        definition.name
                        + "\n"
                        + ast.dump(block, include_attributes=False)
                    ).encode()
                ).hexdigest()[:16]
                name = "_host_closure_" + digest
                helper = ast.parse(f"def {name}({', '.join(inputs)}): pass").body[0]
                helper.body = [copy.deepcopy(first), copy.deepcopy(second)]
                regions.append(
                    (
                        helper,
                        {
                            "name": name,
                            "inputs": inputs,
                            "outputs": [],
                            "kind": "closure",
                        },
                    )
                )
    return regions


class HostSource(ast.NodeTransformer):
    """Retain control flow, discarding annotations which imply Cython C types."""

    def __init__(self, modules=()):
        self.modules = modules

    def visit_ImportFrom(self, node):
        if node.level or node.module not in self.modules:
            return node
        assignments = []
        for name in node.names:
            if name.name == "*":
                raise ValueError("wildcard imports cannot bind a compiled host")
            assignments.extend(
                ast.parse(
                    f"{name.asname or name.name} = _runtime.import_attribute({node.module!r}, {name.name!r})"
                ).body
            )
        return assignments

    def visit_FunctionDef(self, node):
        node.returns = None
        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg):
                argument.annotation = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign([node.target], node.value), node)


def source_module(module, package, modules=()):
    path = Path(inspect.getsourcefile(module))
    source = path.read_text()
    tree = ast.parse(source)
    definitions = []
    classes = {"PointwiseDynamicFunction", "FunctionSchema", "StridedBuffer"}
    for node in tree.body:
        value = vars(module).get(getattr(node, "name", ""))
        function = value.fn if type(value).__name__ == "ConstexprFunction" else value
        if isinstance(node, ast.FunctionDef) and isinstance(
            function, types.FunctionType
        ):
            if function.__module__ == module.__name__:
                if type(value).__name__ == "ConstexprFunction":
                    node.decorator_list = []
                definitions.append(node)
        elif isinstance(node, ast.ClassDef) and (
            node.name in classes
            or (isinstance(value, type) and dataclasses.is_dataclass(value))
            or (
                isinstance(value, type)
                and value.__module__ == module.__name__
                and module.__name__.startswith(("flag_gems", "hopper"))
                and not issubclass(value, (Enum, tuple))
            )
        ):
            for index, method in enumerate(node.body):
                if not isinstance(method, ast.FunctionDef):
                    continue
                member = inspect.getattr_static(value, method.name, None)
                if type(member).__name__ in {
                    "JITFunction",
                    "LibEntry",
                    "Heuristics",
                    "Autotuner",
                }:
                    node.body[index] = ast.parse(
                        f"{method.name} = _original.{node.name}.__dict__[{method.name!r}]"
                    ).body[0]
            if dataclasses.is_dataclass(value):
                fields = [field for field in dataclasses.fields(value) if field.init]
                parameters = []
                assignments = []
                for field in fields:
                    reference = (
                        f"_original.{node.name}.__dataclass_fields__[{field.name!r}]"
                    )
                    if field.default_factory is not dataclasses.MISSING:
                        default = "=_dataclass_missing"
                        assigned = f"{reference}.default_factory() if {field.name} is _dataclass_missing else {field.name}"
                    else:
                        default = (
                            ""
                            if field.default is dataclasses.MISSING
                            else f"={reference}.default"
                        )
                        assigned = field.name
                    parameters.append(field.name + default)
                    if value.__dataclass_params__.frozen:
                        assignments.append(
                            f"    object.__setattr__(self, {field.name!r}, {assigned})"
                        )
                    else:
                        assignments.append(f"    self.{field.name} = {assigned}")
                lines = [f"def __init__(self, {', '.join(parameters)}):"]
                lines.extend(assignments or ["    pass"])
                if hasattr(value, "__post_init__"):
                    lines.append("    self.__post_init__()")
                node.body.insert(0, ast.parse("\n".join(lines)).body[0])
                node.bases = [
                    ast.Attribute(
                        ast.Name("_original", ast.Load()), node.name, ast.Load()
                    )
                ]
                node.decorator_list = []
            definitions.append(node)
    regions = [
        region
        for definition in definitions
        for region in (
            metadata_regions(definition)
            + closure_regions(definition)
            + continuation_regions(definition)
        )
    ]
    definitions.extend(function for function, _ in regions)
    definitions = [HostSource(modules).visit(node) for node in definitions]
    for definition in definitions:
        if (
            isinstance(definition, ast.ClassDef)
            and definition.name == "PointwiseDynamicFunction"
        ):
            for method in definition.body:
                if (
                    isinstance(method, ast.FunctionDef)
                    and method.name == "_call_real_impl"
                ):
                    method.body[:0] = ast.parse("""
_native_result = _runtime.pointwise_prepare(self, args, kwargs, execute=True)
if _native_result is not None:
    return _native_result
""").body
                elif (
                    isinstance(method, ast.FunctionDef) and method.name == "instantiate"
                ):
                    method.body = ast.parse(
                        "return _runtime.pointwise_instantiate(self, ndim)"
                    ).body
    names = {node.name for node in definitions}
    used = {
        node.id
        for definition in definitions
        for node in ast.walk(definition)
        if isinstance(node, ast.Name)
    }
    globals_used = sorted(used.intersection(vars(module)) - names)
    header = [
        "# Generated from the installed wheel; do not edit.",
        "# Source license: Apache-2.0 (FlagGems) or the original module license.",
        "from importlib import import_module as _import_module",
        f"from {package} import runtime as _runtime",
        "_dataclass_missing = object()",
        f"_original = _import_module({module.__name__!r})",
    ]
    header.extend(f"{name} = _original.__dict__[{name!r}]" for name in globals_used)
    # Binding is performed after every module has loaded, so import cycles do
    # not return a mixture of Python and compiled host functions.
    body = ast.unparse(ast.fix_missing_locations(ast.Module(definitions, [])))
    binder = ["def _bind(resolve):"]
    if globals_used:
        binder.append("    global " + ", ".join(globals_used))
        binder.extend(
            f"    {name} = resolve(_original.__dict__[{name!r}])"
            for name in globals_used
        )
    else:
        binder.append("    pass")
    return "\n".join(header) + "\n\n" + body + "\n\n" + "\n".join(binder) + "\n", {
        "module": module.__name__,
        "source": str(path),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "functions": sorted(names),
        "globals": globals_used,
        "regions": [record for _, record in regions],
    }


def collect_modules(functions):
    modules = {}
    pending = [importlib.import_module(fn.__module__) for fn in functions]
    pending.append(importlib.import_module("flag_gems.utils.pointwise_dynamic"))
    pending.append(importlib.import_module("triton"))
    prefixes = (
        "flag_gems.utils",
        "flag_gems.ops",
        "hopper.ops",
        "torch._prims_common",
        "triton.tools.tensor_descriptor",
        "triton._utils",
    )
    while pending:
        module = pending.pop()
        if module.__name__ in modules:
            continue
        modules[module.__name__] = module
        for value in vars(module).values():
            if isinstance(value, types.FunctionType):
                value = inspect.unwrap(value)
            owner = getattr(value, "__module__", "")
            if isinstance(value, (types.FunctionType, type)) and owner.startswith(
                prefixes
            ):
                pending.append(importlib.import_module(owner))
        for node in ast.walk(
            ast.parse(Path(inspect.getsourcefile(module)).read_text())
        ):
            if (
                isinstance(node, ast.ImportFrom)
                and not node.level
                and node.module
                and node.module.startswith(prefixes)
            ):
                pending.append(importlib.import_module(node.module))
    return modules


def closure_source(function, provider, attribute):
    node = HostSource().visit(
        ast.parse(textwrap.dedent(inspect.getsource(function.__code__))).body[0]
    )
    freevars = function.__code__.co_freevars
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    globals_used = sorted(used.intersection(function.__globals__) - set(freevars))
    header = [
        "from importlib import import_module",
        f"_function = import_module({provider!r}).__dict__[{attribute!r}]",
    ]
    header.extend(f"{name} = _function.__globals__[{name!r}]" for name in globals_used)
    factory = ["def make(bindings):"]
    factory.extend(f"    {name} = bindings[{name!r}]" for name in freevars)
    factory.append(
        textwrap.indent(ast.unparse(ast.fix_missing_locations(node)), "    ")
    )
    factory.append(f"    return {node.name}")
    binder = ["def _bind(resolve):"]
    if globals_used:
        binder.append("    global " + ", ".join(globals_used))
        binder.extend(
            f"    {name} = resolve(_function.__globals__[{name!r}])"
            for name in globals_used
        )
    else:
        binder.append("    pass")
    return "\n".join(header + factory + binder) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--functions", nargs="+", help="Restrict seed functions for development probes"
    )
    parser.add_argument("--package", default="fg_aot")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--build-temp", type=Path)
    parser.add_argument("--emit-only", action="store_true")
    parser.add_argument("--launcher-only", action="store_true")
    args = parser.parse_args()
    # The vendor core must be selected before Torch loads its native libraries.
    import torch_fl  # noqa: F401
    import torch
    import triton
    from setuptools import Distribution, Extension

    output = args.output.resolve()
    namespace = args.package
    package = output.joinpath(*namespace.split("."))
    build_temp = args.build_temp.resolve() if args.build_temp else output / "build"
    source_dir = build_temp / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    root = Path(__file__).resolve().parent
    from torch_fl import _build_config
    from torch_fl.native_host.backend import build_record, get_backend

    backend = get_backend(_build_config.ACCELERATOR)
    triton_root = Path(triton.__file__).parent
    backend_record = build_record(backend, triton_root)
    extensions = backend.extensions(namespace, root, torch, backend_record, triton_root)
    manifest = {
        "package": namespace,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "backend": backend_record,
        "modules": {},
        "closures": {},
    }
    if not args.launcher_only:
        # Install platform policies and optional-dependency shims before the
        # published operators import Triton extensions at module scope.
        torch_fl.flagos.init()
        import flag_gems
        from torch_fl import _build_config
        from torch_fl.native_host import source_variants
        from torch_fl.native_host.vendor_cohort import source_policy_for_build
        from Cython.Build import cythonize
        from Cython.Compiler import Options

        # Version-gated FlagGems branches may name APIs absent from this Torch
        # wheel. Preserve Python's NameError only if that branch is executed.
        Options.error_on_unknown_names = False

        if args.functions is None:
            args.functions = sorted(
                {
                    function.__name__
                    for _, function, *_ in flag_gems._FULL_CONFIG
                    if isinstance(function, types.FunctionType)
                    and hasattr(flag_gems, function.__name__)
                }
            )
        functions = [getattr(flag_gems, name) for name in args.functions]
        registry_functions = [
            function
            for _, function, *_ in flag_gems._FULL_CONFIG
            if isinstance(function, types.FunctionType)
        ]
        policy = source_policy_for_build(_build_config, args.source_root)
        alternatives = source_variants.load(policy)
        if policy is not None:
            manifest["source_variants"] = policy
        modules = collect_modules(
            functions + registry_functions + list(alternatives.values())
        )
        cython_extensions = []
        runtime = source_dir / "runtime.pyx"
        write_changed(
            runtime,
            (root / "runtime.py")
            .read_text()
            .replace(
                "from fg_aot import _launcher", f"from {namespace} import _launcher"
            ),
        )
        cython_extensions.append(Extension(namespace + ".runtime", [str(runtime)]))
        for name, module in sorted(modules.items()):
            code, record = source_module(module, namespace, modules)
            generated = source_dir / (module_name(name) + ".pyx")
            write_changed(generated, code)
            manifest["modules"][name] = record
            cython_extensions.append(
                Extension(namespace + "." + generated.stem, [str(generated)])
            )
            for attribute, value in vars(module).items():
                if not isinstance(value, types.FunctionType) or not value.__closure__:
                    continue
                identity = function_identity(value)
                if identity in manifest["closures"]:
                    continue
                closure_name = module_name(identity)
                closure = source_dir / (closure_name + ".pyx")
                write_changed(closure, closure_source(value, name, attribute))
                manifest["closures"][identity] = {
                    "module": closure_name,
                    "provider": name,
                    "attribute": attribute,
                }
                cython_extensions.append(
                    Extension(namespace + "." + closure_name, [str(closure)])
                )
        # torch-fl adapters can export closures under a public FlagGems name;
        # their private __name__ is not an attribute of their defining module.
        for public_name, function in zip(args.functions, functions):
            if not isinstance(function, types.FunctionType) or not function.__closure__:
                continue
            identity = function_identity(function)
            if identity in manifest["closures"]:
                continue
            closure_name = module_name(identity)
            closure = source_dir / (closure_name + ".pyx")
            write_changed(closure, closure_source(function, "flag_gems", public_name))
            manifest["closures"][identity] = {
                "module": closure_name,
                "provider": "flag_gems",
                "attribute": public_name,
            }
            cython_extensions.append(
                Extension(namespace + "." + closure_name, [str(closure)])
            )
        manifest["ops"] = {
            name: [fn.__module__, fn.__name__]
            for name, fn in zip(args.functions, functions)
        }
        # Publish every registered host already covered by this source closure.
        # Selecting seed operators must not leave helper/factory siblings in Python.
        for _, function, *_ in flag_gems._FULL_CONFIG:
            if not isinstance(function, types.FunctionType):
                continue
            name = function.__name__
            resolved = getattr(flag_gems, name, function)
            record = manifest["modules"].get(resolved.__module__)
            if record and resolved.__name__ in record["functions"]:
                manifest["ops"][name] = [resolved.__module__, resolved.__name__]
        extensions.extend(
            cythonize(
                cython_extensions,
                compiler_directives={"language_level": 3, "annotation_typing": False},
                nthreads=args.jobs,
            )
        )
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    license_dir = package / "licenses"
    license_dir.mkdir(exist_ok=True)
    for provider in ("torch", "flag_gems", "flagtree"):
        distribution = importlib.metadata.distribution(provider)
        for source in distribution.files or ():
            if source.name.lower().startswith(
                ("license", "licence", "notice", "copying")
            ) and any(part.endswith(".dist-info") for part in source.parts):
                (license_dir / (provider + "-" + source.name)).write_bytes(
                    Path(distribution.locate_file(source)).read_bytes()
                )
    if args.emit_only:
        print(
            json.dumps(
                {
                    "host_modules": len(extensions),
                    "ops": len(manifest.get("ops", {})),
                    "compiled": False,
                }
            )
        )
        return
    distribution = Distribution(
        {"name": "torch-fl-host-probe", "ext_modules": extensions}
    )
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(output)
    command.build_temp = str(build_temp / "objects")
    command.parallel = args.jobs
    command.ensure_finalized()
    command.run()
    expected = {
        Path(command.get_ext_fullpath(extension.name)).resolve()
        for extension in extensions
    }
    for old in package.glob("*.so"):
        if old.resolve() not in expected:
            old.unlink()
    if not args.launcher_only:
        from torch_fl.native_host.compiler.bundle import build as build_typed
        from torch_fl.native_host.load import Hosts, install_launcher

        # This process is the wheel builder. Its final C++ compilation is still
        # build-time work; installed wheels always enable the compilation guard.
        install_launcher(namespace, enforce_native_compilation=False)
        hosts = Hosts(package)
        hosts.activate()
        typed = build_typed(
            package / "typed",
            hosts,
            set(manifest["ops"]),
            source_root=root.parents[1],
            build_temp=build_temp / "typed",
        )
        manifest["typed_hosts"] = {
            "compiled": len(typed["compiled"]),
            "fallback": len(typed["fallback"]),
            "coverage": typed["coverage"],
        }
        (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "host_modules": len(extensions)}))


if __name__ == "__main__":
    main()
