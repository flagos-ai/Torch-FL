# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build-time source compilation experiment; never imported by the runtime."""

import argparse
import ast
import dataclasses
from enum import Enum
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
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
    definitions = [HostSource(modules).visit(node) for node in definitions]
    for definition in definitions:
        if (
            isinstance(definition, ast.ClassDef)
            and definition.name == "PointwiseDynamicFunction"
        ):
            for method in definition.body:
                if isinstance(method, ast.FunctionDef) and method.name == "instantiate":
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
        "--functions", nargs="+", help="Restrict seed functions for development probes"
    )
    parser.add_argument("--package", default="fg_aot")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--build-temp", type=Path)
    parser.add_argument("--emit-only", action="store_true")
    parser.add_argument("--launcher-only", action="store_true")
    args = parser.parse_args()
    import torch
    import torch_fl  # noqa: F401
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
    cuda = Path(
        os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    )
    torch_include = Path(torch.__file__).parent / "include"
    driver = Path(triton.__file__).parent / "backends/nvidia/driver.c"
    extensions = [
        Extension(
            namespace + "._launcher",
            [str(root / "csrc/launcher.cc")],
            include_dirs=[str(torch_include), str(cuda / "include")],
            library_dirs=[str(cuda / "lib64/stubs"), str(cuda / "lib/stubs")],
            libraries=["cuda"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        Extension(
            namespace + ".cuda_utils",
            [str(driver)],
            include_dirs=[str(cuda / "include")],
            library_dirs=[str(cuda / "lib64/stubs"), str(cuda / "lib/stubs")],
            libraries=["cuda"],
            extra_compile_args=["-O3"],
        ),
    ]
    manifest = {
        "package": namespace,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "modules": {},
        "closures": {},
    }
    if not args.launcher_only:
        import flag_gems
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
        modules = collect_modules(functions + registry_functions)
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
    print(json.dumps({"output": str(output), "host_modules": len(extensions)}))


if __name__ == "__main__":
    main()
