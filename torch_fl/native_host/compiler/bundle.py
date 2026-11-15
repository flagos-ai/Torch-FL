# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile the installed FlagGems registry into shape-independent host programs."""

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import torch
import flag_gems
from .frontend import Frontend
from .ir import emit, HEADER
from .hooks import compile_hook
from .coverage import discover_variants, write_coverage


class Recorder:
    def __init__(self, hosts):
        self.hosts = hosts
        self.records = {"kernel": [], "pointwise": [], "foreign": [], "global": []}
        self.references = {}
        for module_name, module in sorted(hosts.originals.items()):
            for name, value in vars(module).items():
                path = [name]
                for _ in range(5):
                    self.references.setdefault(
                        id(value), {"module": module_name, "path": path[:]}
                    )
                    if not hasattr(value, "arg_names") or not hasattr(value, "fn"):
                        break
                    value = value.fn
                    path.append("fn")
        for module_name, module in sorted(hosts.modules.items()):
            for name, value in vars(hosts.originals[module_name]).items():
                if (
                    isinstance(value, type)
                    and value.__module__ == module_name
                    and hasattr(module, name)
                ):
                    resolved = getattr(module, name)
                    for attribute in vars(value):
                        if attribute.startswith("__") or not hasattr(
                            resolved, attribute
                        ):
                            continue
                        member = getattr(resolved, attribute)
                        self.references.setdefault(
                            id(member),
                            {"generated": module_name, "path": [name, attribute]},
                        )
            for record in hosts.manifest["modules"][module_name].get("regions", []):
                value = getattr(module, record["name"])
                self.references[id(value)] = {
                    "generated": module_name,
                    "path": [record["name"]],
                }

    def symbol(self, value):
        if (
            inspect.isfunction(value)
            and value.__module__ == "torch"
            and getattr(torch, value.__name__, None) is value
        ):
            return {"builtin_module": "torch", "path": [value.__name__]}
        if inspect.ismethod(value) and hasattr(value.__self__, "_schema"):
            schema = value.__self__._schema
            return {
                "operator": schema.name,
                "overload": schema.overload_name or "default",
                "method": value.__name__,
            }
        if id(value) in self.references:
            return self.references[id(value)]
        for original, resolved in self.hosts.objects.items():
            if resolved is value and original in self.references:
                return self.references[original]
        raise NotImplementedError("unbound source symbol " + repr(value))

    def add(self, kind, record):
        index = len(self.records[kind])
        self.records[kind].append(record)
        return index

    def register_script_kernel(self, entry):
        return self.add("kernel", {"symbol": self.symbol(entry.original)})

    def register_pointwise(self, owner):
        return self.add("pointwise", {"symbol": self.symbol(owner)})

    def register_foreign_projection(self, function, attribute):
        return self.add(
            "foreign",
            {"symbol": self.symbol(function), "prefix": [], "attribute": attribute},
        )

    def register_foreign(self, function, prefix):
        return self.add(
            "foreign",
            {
                "symbol": self.symbol(function),
                "prefix": [self.symbol(v) for v in prefix],
            },
        )

    def register_global(self, module, name, kind):
        return self.add(
            "global", {"symbol": {"module": module, "path": [name]}, "type": kind}
        )


def cpp_type(t):
    kind = t.kind()
    if kind == "OptionalType":
        return "std::optional<" + cpp_type(t.getElementType()) + ">"
    if kind == "ListType":
        return "std::vector<" + cpp_type(t.getElementType()) + ">"
    return {
        "TensorType": "at::Tensor",
        "IntType": "int64_t",
        "FloatType": "double",
        "BoolType": "bool",
        "StringType": "std::string",
        "DeviceObjType": "c10::Device",
    }[kind]


def argument_check(t, value):
    if t.kind() == "OptionalType":
        return (
            "("
            + value
            + ".isNone() || "
            + argument_check(t.getElementType(), value)
            + ")"
        )
    if t.kind() == "ListType":
        return (
            value
            + "."
            + {"TensorType": "isTensorList", "IntType": "isIntList"}[
                t.getElementType().kind()
            ]
            + "()"
        )
    return (
        value
        + "."
        + {
            "TensorType": "isTensor",
            "IntType": "isInt",
            "FloatType": "isDouble",
            "BoolType": "isBool",
            "StringType": "isString",
            "DeviceObjType": "isDevice",
        }[t.kind()]
        + "()"
    )


NATIVE_HEADER = r"""
#include "source_host.h"
using SourceKwarg=at::native::flagos::PyKwarg;
inline bool HookTruth(const c10::IValue& value) {
 if(value.isBool())return value.toBool();if(value.isInt())return value.toInt()!=0;
 if(value.isDouble())return value.toDouble()!=0;if(value.isNone())return false;
 if(value.isString())return !value.toStringRef().empty();
 TORCH_CHECK(false,"unsupported native metadata-hook condition");
}
inline c10::IValue SourceValue(c10::IValue value) {
 if(value.isTensor()) {
  const auto& tensor=value.toTensor();
  if(!tensor.defined())return c10::IValue();
  if(tensor.device().is_cpu() && tensor.unsafeGetTensorImpl()->is_wrapped_number())return c10::IValue(tensor.item());
 }
 return value;
}
"""


def build(directory, hosts, names, *, source_root, build_temp, emit_only=False):
    from setuptools import Extension, Distribution
    from torch_fl import _build_config

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_root)
    build_temp = Path(build_temp)
    build_temp.mkdir(parents=True, exist_ok=True)
    variants, entries = discover_variants(
        flag_gems._FULL_CONFIG,
        hosts.original_ops,
        names,
        torch._C._jit_get_schemas_for_operator,
    )
    cohort = {
        "vendor": _build_config.ACCELERATOR,
        "torch": hosts.manifest["torch"],
        "triton": hosts.manifest["triton"],
        "flagtree": importlib.metadata.version("flagtree"),
        "flag_gems": importlib.metadata.version("flag_gems"),
        "host_manifest_sha256": hashlib.sha256(
            json.dumps(hosts.manifest, sort_keys=True).encode()
        ).hexdigest(),
        "selected_hosts": sorted(names),
    }
    write_coverage(directory, entries, [], [], cohort=cohort)
    recorder = Recorder(hosts)
    library = torch.library.Library("flagos_script_host", "DEF")
    library.define(
        "launch(int slot, Any[] args, Dict(str, Any) kwargs, int grid_id, int[] captures) -> ()",
        alias_analysis="CONSERVATIVE",
    )
    library.define("deterministic() -> bool", alias_analysis="CONSERVATIVE")
    for kind in ("bool", "int", "float", "str"):
        library.define(f"global_{kind}(int slot) -> {kind}")
    library.define("data_ptr(Tensor tensor) -> int")
    library.define(
        "descriptor(Tensor base, int[] shape, int[] strides, int[] block_shape, str padding) -> Any",
        alias_analysis="CONSERVATIVE",
    )
    declarations = []
    successful = []
    failed = []
    sample = None
    native_groups = {}
    hook_records = []
    hook_declarations = []
    seen_hooks = set()
    metadata_sources = {}
    for original in hosts.originals.values():
        for function in vars(original).values():
            if (
                not inspect.isfunction(function)
                or id(function) in seen_hooks
                or function.__module__ not in hosts.originals
            ):
                continue
            seen_hooks.add(id(function))
            name = "MetadataHook" + str(len(hook_records))
            try:
                code, writes = compile_hook(function, name)
            except (NotImplementedError, OSError, TypeError, SyntaxError):
                continue
            hook_declarations.append(code)
            hook_records.append(
                {
                    "symbol": recorder.symbol(function),
                    "name": name,
                    "writes": writes,
                    "sha256": hashlib.sha256(
                        inspect.getsource(function).encode()
                    ).hexdigest(),
                }
            )
    for index, (name, function, signature, public) in enumerate(variants):
        namespace = "Host" + str(index)
        checkpoints = {kind: len(records) for kind, records in recorder.records.items()}
        try:
            frontend = Frontend(hosts, recorder)
            source = frontend.source(function, signature)
            if function in frontend.fallback_functions:
                raise NotImplementedError(
                    "root host requires compiled object semantics"
                )
            path = build_temp / (namespace + ".py")
            path.write_text(source)
            spec = importlib.util.spec_from_file_location("bundle_" + namespace, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            scripted = torch.jit.script(module.Host())
            graph = scripted.forward.graph.copy()
            torch._C._jit_pass_inline(graph)
            torch._C._jit_pass_peephole(graph)
            torch._C._jit_pass_cse(graph)
            torch._C._jit_pass_dce(graph)
            grids = [
                torch.jit.script(getattr(module, "grid_" + str(i)))
                for i in range(len(frontend.grids))
            ]
            code = "\n".join(
                emit(f.graph, function_name="Grid" + str(i), binding=False)
                for i, f in enumerate(grids)
            )
            code += "\n" + emit(
                graph, binding=False, grid_slots=frontend.kernel_slots, slot_tables=True
            )
            graph_args = list(scripted.forward.schema.arguments)[1:]
            graph_values = list(graph.inputs())[1:]
            types = {
                a.name: cpp_type(v.type()) for a, v in zip(graph_args, graph_values)
            }
            jit_types = {a.name: v.type() for a, v in zip(graph_args, graph_values)}
            for dependency in frontend.constants.dependencies:
                dependency_module = frontend.constants.modules[dependency]
                dependency_path = inspect.getsourcefile(dependency_module)
                if dependency_path:
                    metadata_sources[dependency] = hashlib.sha256(
                        Path(dependency_path).read_bytes()
                    ).hexdigest()
            declarations.append("namespace " + namespace + " {\n" + code + "\n}")
            successful.append(
                {"name": name, "signature": signature, "namespace": namespace}
            )
            native_groups.setdefault(name, []).append(
                (namespace, public, graph_args, types, jit_types)
            )
            if sample is None:
                sample = emit(graph, grid_slots=frontend.kernel_slots)
            print("COMPILED", name, signature, flush=True)
        except Exception as error:
            for kind, length in checkpoints.items():
                del recorder.records[kind][length:]
            failed.append({"name": name, "signature": signature, "error": str(error)})
            print("FALLBACK", name, str(error).splitlines()[-1][:180], flush=True)
    coverage = write_coverage(
        directory, entries, successful, failed, cohort=cohort, complete=True
    )
    if not successful:
        raise RuntimeError("no native roots compiled")
    native_code = []
    callback_bindings = []
    for number, (name, group) in enumerate(native_groups.items()):
        callback = "NativeEntry" + str(number)
        public = group[0][1]
        names = [p["name"] for p in public]
        positional = sum(not p["keyword"] for p in public)
        lines = [
            "bool "
            + callback
            + "(c10::ArrayRef<c10::IValue> args,c10::ArrayRef<SourceKwarg> kwargs,at::Tensor& output) {",
            f"if(args.size()>{positional})return false;",
        ]
        for i, p in enumerate(public):
            value = p["default"]
            initial = (
                "c10::IValue()"
                if value is None
                else "c10::IValue("
                + (
                    "true"
                    if value is True
                    else "false"
                    if value is False
                    else "int64_t(" + str(value) + ")"
                    if type(value) is int
                    else json.dumps(value)
                )
                + ")"
            )
            lines.append(
                f"c10::IValue a{i}={initial};bool present{i}={str(p['has_default']).lower()};"
            )
            if not p["keyword"]:
                lines.append(
                    f"if(args.size()>{i}){{a{i}=SourceValue(args[{i}]);present{i}=true;}}"
                )
        lines.append("for(const auto& kw:kwargs) {")
        for i, p in enumerate(public):
            condition = (
                ("if" if i == 0 else "else if")
                + "(std::string_view(kw.name)=="
                + json.dumps(p["name"])
                + ") {"
            )
            lines.append(condition)
            if not p["keyword"]:
                lines.append(f"if(args.size()>{i})return false;")
            if "torch.dtype" in p["type"]:
                lines.append(
                    "if(!kw.is_none && !kw.value.isNone() && !kw.is_dtype)return false;"
                )
            lines.append(
                f"a{i}=kw.is_none?c10::IValue():SourceValue(kw.value);present{i}=true;}}"
            )
        lines.append("else return false; }")
        lines.append(
            "if(!("
            + " && ".join("present" + str(i) for i in range(len(public)))
            + "))return false;"
        )
        for namespace, _, graph_args, types, jit_types in group:
            checks = [
                argument_check(jit_types[p["name"]], "a" + str(i))
                for i, p in enumerate(public)
            ]
            values = {
                p["name"]: "a"
                + str(i)
                + ".to<"
                + types[p["name"]].replace("std::vector<", "c10::List<")
                + ">()"
                for i, p in enumerate(public)
            }
            lines.append(
                "if("
                + " && ".join(checks)
                + ") { output="
                + namespace
                + "::Forward("
                + ", ".join(values[a.name] for a in graph_args)
                + ");return true;}"
            )
        lines.append("return false; }")
        native_code.append("\n".join(lines))
        callback_bindings.append(
            "callbacks["
            + json.dumps(name)
            + "]=py::int_(reinterpret_cast<uintptr_t>(&"
            + callback
            + "));"
        )
    init = sample[sample.index("PYBIND11_MODULE(") :].replace(
        "source_host_ir,", "_source_hosts,"
    )
    init = init.replace(
        'module.def("forward", &Forward, py::call_guard<py::gil_scoped_release>());', ""
    )
    init = init.replace(
        '  module.def("bind_api",',
        '  module.def("bind_slots", [](std::vector<int64_t> k,std::vector<int64_t> p,std::vector<int64_t> f) { kernel_slots=std::move(k);pointwise_slots=std::move(p);foreign_slots=std::move(f); });\n  module.def("bind_api",',
    )
    init = (
        init[:-2]
        + "\npy::dict metadata_hooks;\n"
        + "\n".join(
            "metadata_hooks["
            + json.dumps(h["name"])
            + "]=py::int_(reinterpret_cast<uintptr_t>(&"
            + h["name"]
            + "));"
            for h in hook_records
        )
        + '\nmodule.attr("metadata_hooks")=metadata_hooks;\n}\n'
    )
    init = (
        init[:-2]
        + "\npy::dict callbacks;\n"
        + "\n".join(callback_bindings)
        + '\nmodule.attr("callbacks")=callbacks;\n}\n'
    )
    path = build_temp / "source_hosts.cpp"
    path.write_text(
        HEADER
        + "\n"
        + NATIVE_HEADER
        + "\n"
        + "\n".join(hook_declarations + declarations + native_code)
        + "\n"
        + init
    )
    if emit_only:
        return {
            "compiled": successful,
            "fallback": failed,
            "source": str(path),
            "coverage": coverage["summary"],
        }
    torch_dir = Path(torch.__file__).parent
    extension = Extension(
        "_source_hosts",
        [str(path)],
        include_dirs=[
            str(torch_dir / "include"),
            str(torch_dir / "include/torch/csrc/api/include"),
            str(source_root / "csrc/aten/backends/flagos"),
            str(source_root / "torch_fl/native_host/csrc"),
        ],
        library_dirs=[str(torch_dir / "lib")],
        libraries=["torch_python", "torch_cpu", "c10"],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        ],
    )
    distribution = Distribution(
        {"name": "source-host-bundle-probe", "ext_modules": [extension]}
    )
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(directory)
    command.build_temp = str(build_temp / "objects")
    command.ensure_finalized()
    command.run()
    expected = Path(command.get_ext_fullpath(extension.name)).resolve()
    for old in directory.glob("*.so"):
        if old.resolve() != expected:
            old.unlink()
    coverage = write_coverage(
        directory,
        entries,
        successful,
        failed,
        cohort=cohort,
        complete=True,
        artifact_built=True,
    )
    manifest = {
        "source_abi": 2,
        "global_abi": 1,
        "foreign_abi": 1,
        "metadata_sources": metadata_sources,
        "pointwise_launch_abi": 1,
        "pointwise_at_abi": 1,
        "hook_abi": 2,
        "native_hooks": hook_records,
        "records": recorder.records,
        "compiled": successful,
        "fallback": failed,
        "coverage": coverage["summary"],
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "abi": 2,
    }
    artifacts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*.so")
    }
    manifest["artifacts"] = artifacts
    (directory / "__init__.py").write_text("")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
