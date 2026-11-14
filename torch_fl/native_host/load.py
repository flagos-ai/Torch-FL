# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Load prebuilt host artifacts and bind original FlagGems device functions."""

import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
from pathlib import Path
import shlex
import sys
import types

_native_commands = []
_admitted_kernels = {}
_launched_kernels = {}
_audit_installed = False
_launcher_module = None


def audit_native_compilers(event, values):
    if event != "subprocess.Popen":
        return
    frame = sys._getframe(1)
    while frame is not None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if any(
            part in path
            for part in ("/flag_gems/", "/triton/", "/torch_fl/native_host/")
        ):
            break
        frame = frame.f_back
    if frame is None:
        return
    command = values[1]
    words = command if isinstance(command, (list, tuple)) else shlex.split(str(command))
    command = list(map(str, words))
    _native_commands.append(command)
    forbidden = {
        "cc",
        "c++",
        "gcc",
        "g++",
        "clang",
        "clang++",
        "nvcc",
        "ninja",
        "cmake",
        "make",
        "ld",
        "lld",
    }
    if any(
        Path(word).name in forbidden
        for argument in command
        for word in shlex.split(argument)
    ):
        raise RuntimeError(f"runtime native compilation is forbidden: {command}")


def kernel_launcher(source, metadata):
    provenance = _admitted_kernels.get(id(source.fn))
    if provenance is None:
        raise RuntimeError(f"kernel has no admitted FlagGems source: {source.fn}")
    if hashlib.sha256(source.fn.src.encode()).hexdigest() != provenance["jit_sha256"]:
        raise RuntimeError("FlagGems JIT source changed after admission")
    _launched_kernels[metadata.name] = provenance
    return _launcher_module.Launcher(source, metadata)


def module_name(name):
    return "m_" + hashlib.sha256(name.encode()).hexdigest()[:16]


def function_identity(function):
    code = function.__code__
    return f"{function.__module__}:{function.__qualname__}:{Path(code.co_filename).name}:{code.co_firstlineno}"


def install_launcher(package="fg_aot"):
    global _audit_installed, _launcher_module
    import triton.backends.nvidia.driver as driver

    _launcher_module = importlib.import_module(package + "._launcher")
    cuda_utils = importlib.import_module(package + ".cuda_utils")

    driver.CudaUtils = lambda: cuda_utils
    driver.PyCUtensorMap = cuda_utils.PyCUtensorMap
    driver.CudaLauncher = kernel_launcher
    if not _audit_installed:
        sys.addaudithook(audit_native_compilers)
        _audit_installed = True

    # Refuse an accidental return to either runtime-generated native module.
    def reject(*args, **kwargs):
        raise RuntimeError("runtime C/C++ launcher compilation is forbidden")

    driver.compile_module_from_src = reject
    from triton.runtime import driver as runtime_driver

    for active in (
        getattr(runtime_driver, "_active", None),
        getattr(runtime_driver, "_default", None),
    ):
        if active is not None:
            active.utils = cuda_utils
            active.launcher_cls = kernel_launcher


class Hosts:
    def __init__(self, path):
        import torch_fl
        import flag_gems
        import torch
        import triton
        from flag_gems.utils.libentry import LibEntry
        from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction

        self.path = Path(path)
        self.artifacts = self._artifacts()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.original_ops = {
            name: getattr(flag_gems, name)
            for name in self.manifest["ops"]
            if hasattr(flag_gems, name)
        }
        if (
            self.manifest["torch"] != torch.__version__
            or self.manifest["triton"] != triton.__version__
        ):
            raise RuntimeError(
                "native hosts require their build-time Torch/FlagTree cohort"
            )
        driver_source = Path(triton.__file__).parent / "backends/nvidia/driver.c"
        if (
            hashlib.sha256(driver_source.read_bytes()).hexdigest()
            != self.manifest["driver_sha256"]
        ):
            raise RuntimeError(
                "FlagTree CUDA driver ABI changed since the native-host build"
            )
        self.package = self.manifest.get("package", "fg_aot")
        _launcher = importlib.import_module(self.package + "._launcher")
        runtime = importlib.import_module(self.package + ".runtime")
        self.modules = {}
        self.originals = {}
        self.objects = {}
        self.closures = {
            identity: importlib.import_module(self.package + "." + record["module"])
            for identity, record in self.manifest.get("closures", {}).items()
        }
        self.provenance = []
        self.LibEntry = LibEntry
        self.Pointwise = PointwiseDynamicFunction
        self.runtime = runtime
        _launcher.configure_runtime(
            str(Path(torch_fl.__file__).parent / "lib/libflagos.so")
        )
        for name, record in self.manifest["modules"].items():
            original = importlib.import_module(name)
            source = Path(inspect.getsourcefile(original))
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"FlagGems host source changed: {name}")
            self.originals[name] = original
            self.modules[name] = importlib.import_module(
                self.package + "." + module_name(name)
            )
        for name, (module, function) in self.manifest["ops"].items():
            if name not in self.original_ops:
                self.original_ops[name] = getattr(self.originals[module], function)
        for module in self.modules.values():
            module._bind(self.resolve)
        for module in self.closures.values():
            module._bind(self.resolve)
        shape = self.modules["flag_gems.utils.shape_utils"]
        pointwise = self.modules["flag_gems.utils.pointwise_dynamic"]
        runtime.configure(
            self.resolve,
            self.pointwise_kernel,
            (
                shape.heuristics_for_tile_size,
                shape.heuristics_for_num_warps,
                shape.stride_order,
                pointwise._balanced_grid_partition,
                pointwise._tensor_inputs_all_complex,
                flag_gems.runtime.torch_device_fn.get_device_capability()[0],
            ),
            self.modules,
        )

    def resolve(self, value):
        import triton
        from flag_gems.runtime import torch_device_fn

        found = self.objects.get(id(value))
        if found is not None:
            return found
        if value is torch_device_fn:
            found = self.runtime.DeviceAPI(value)
        elif type(value) is logging.Logger:
            found = self.runtime.Logger(value)
        elif isinstance(value, types.ModuleType) and value.__name__ == "triton":
            found = types.SimpleNamespace(**vars(value))
            self.objects[id(value)] = found
            compiled = self.modules["triton"]
            for name in self.manifest["modules"]["triton"]["functions"]:
                setattr(found, name, getattr(compiled, name))
        elif isinstance(value, dict):
            found = {}
            self.objects[id(value)] = found
            found.update((k, self.resolve(v)) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            found = type(value)(self.resolve(v) for v in value)
        elif isinstance(value, triton.KernelInterface):
            entry = value if isinstance(value, self.LibEntry) else self.LibEntry(value)
            jit = entry.jit_function
            source = Path(inspect.getsourcefile(jit.fn)).resolve()
            if "flag_gems" not in source.parts:
                raise RuntimeError(f"device kernel is not from FlagGems: {source}")
            self.provenance.append(
                {
                    "kernel": jit.__name__,
                    "source": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "jit_sha256": hashlib.sha256(jit.src.encode()).hexdigest(),
                }
            )
            _admitted_kernels[id(jit)] = self.provenance[-1]
            found = self.runtime.KernelEntry(value, entry)
        elif isinstance(value, types.FunctionType):
            identity = function_identity(value)
            if identity in self.closures:
                found = self.closures[identity].make(
                    {
                        name: self.resolve(cell.cell_contents)
                        for name, cell in zip(
                            value.__code__.co_freevars, value.__closure__
                        )
                    }
                )
            else:
                module = self.modules.get(value.__module__)
                found = (
                    getattr(module, value.__name__, value)
                    if module is not None
                    else value
                )
        elif isinstance(value, type):
            module = self.modules.get(value.__module__)
            found = (
                getattr(module, value.__name__, value) if module is not None else value
            )
        else:
            kind = type(value)
            module = self.modules.get(kind.__module__)
            compiled = (
                getattr(module, kind.__name__, None) if module is not None else None
            )
            if (
                compiled is not None
                and compiled is not kind
                and hasattr(value, "__dict__")
            ):
                found = compiled.__new__(compiled)
                self.objects[id(value)] = found
                found.__dict__.update(
                    {
                        k: v if k == "_scalar_fn" else self.resolve(v)
                        for k, v in value.__dict__.items()
                    }
                )
                if isinstance(value, self.Pointwise):
                    found.overloads = {}
                    found._kernel_info_cache = {}
                    found._capture_output_pool = {}
            else:
                return value
        self.objects[id(value)] = found
        return found

    def pointwise_kernel(self, owner, ndim):
        # The upstream generator emits only its original device kernel here.
        # The rank-independent host adapter was compiled during the build.
        pointwise = self.originals["flag_gems.utils.pointwise_dynamic"]

        name = f"{owner._scalar_fn.__name__}_kernel_rank_{ndim}"
        generator = pointwise.ModuleGenerator(
            owner.fx, owner._scalar_fn, ndim, name, "unused_host", owner.config
        )
        code = pointwise.IndentedBuffer()
        generator.generate_imports(code)
        if owner.config.prefer_1d_tile:
            generator.kernel_gen.codegen_1d_tile(code)
        else:
            generator.kernel_gen.codegen_nd_tile(code)
        text = code.getvalue()
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = pointwise.code_cache_dir() / f"native_host_kernel_{digest}.py"
        pointwise.write_atomic(str(path), text)
        spec = importlib.util.spec_from_file_location(f"_fg_device_{digest}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__dict__.update(owner._scalar_fn.__globals__)
        module.__dict__[owner._scalar_fn.__name__] = owner._scalar_fn
        entry = getattr(module, name)
        self.provenance.append(
            {
                "kernel": name,
                "source": str(path),
                "sha256": digest,
                "generator": "flag_gems.utils.pointwise_dynamic.KernelGenerator",
                "scalar_kernel": owner._scalar_fn.__name__,
                "jit_sha256": hashlib.sha256(
                    entry.jit_function.src.encode()
                ).hexdigest(),
            }
        )
        _admitted_kernels[id(entry.jit_function)] = self.provenance[-1]
        # Generated device code is admitted only through the wheel's generator.
        return self.runtime.KernelEntry(entry, entry)

    def op(self, name):
        return self.resolve(self.original_ops[name])

    def _artifacts(self):
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.path.glob("*.so")
        }

    def audit(self):
        if self._artifacts() != self.artifacts:
            raise RuntimeError("host artifacts changed during execution")
        return {
            "host_artifacts": self.artifacts,
            "subprocess_commands": _native_commands,
            "launched_kernels": _launched_kernels,
        }

    def activate(self):
        import flag_gems
        import flag_gems.ops
        import torch
        from torch_fl import _build_config

        if torch.version.cuda is not None or "boxing" in _build_config.KERNELS:
            raise RuntimeError(
                "native host experiment requires the CPU-Torch vendor build"
            )
        if getattr(_build_config, "NATIVE_HOSTS", False):
            torch.ops.flagos.strict_host_routes(True)
        for name in self.manifest["ops"]:
            function = self.op(name)
            setattr(flag_gems, name, function)
            if hasattr(flag_gems.ops, name):
                setattr(flag_gems.ops, name, function)
