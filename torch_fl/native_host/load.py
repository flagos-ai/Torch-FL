# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Load prebuilt host artifacts and bind original FlagGems device functions."""

import ast
from contextvars import ContextVar
from functools import cache, wraps
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
from pathlib import Path
import shlex
import sys
import types

_COMPILE_BEFORE_ENQUEUE = frozenset(
    {
        "ascend",
        "cambricon",
        "enflame",
        "hygon",
        "iluvatar",
        "kunlunxin",
        "metax",
        "mthreads",
        "nvidia",
        "thead",
        "tsingmicro",
    }
)
_SINGLETON_TUNER_SOURCES = {
    "flag_gems.flagtune.cost_model": "e50a33b2950f5170041ab03a54a19d355a7ac3b4610254e9f15b04f57e280a62",
    "flag_gems.utils.libentry": "a6bcda6d615748d4ec1be4119f245e62002d5347a4b5e983a21cb6dff933d720",
}

# These configurations are shared by the published kernels. The launcher
# accepts them only when the source hash and active tuning domain agree.
_MUL_SOURCE = "df1d9068fbe35596b74bad406446b39901deb6777e2439cc09bb2b6d1de0f348"
_MUL_CONFIG = dict(BLOCK_SIZE=1024, num_warps=4, num_stages=3, num_ctas=1)
_METAX_LOG_SOFTMAX_SOURCE = (
    "77fcf532bc130edd67f5b452316a87e212fbab47cd585f65ea942ecd35c229fe"
)
_METAX_LOG_SOFTMAX_CONFIG = dict(BLOCK_M=4, num_warps=4, num_stages=2, num_ctas=1)
_PUBLISHED_EAGER_CONFIGS = {
    ("mul_scalar_kernel", _MUL_SOURCE): _MUL_CONFIG,
    ("mul_broadcast_2d_kernel", _MUL_SOURCE): _MUL_CONFIG,
    ("mul_generic_nd_kernel", _MUL_SOURCE): _MUL_CONFIG,
    (
        "mm_kernel_general",
        "d37eb3ca6511b79340f2c199428873baf7d1e704f34146e154c61c1801aa8d89",
    ): dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4, num_ctas=1),
    (
        "bmm_kernel",
        "e07964330ae4b2a4fffea5144d5382e275bc5f4b11deab9cea714705b5e7e509",
    ): dict(
        TILE_M=64,
        TILE_N=64,
        TILE_K=32,
        GROUP_M=2,
        num_warps=4,
        num_stages=2,
        num_ctas=1,
    ),
    (
        "bmm_kernel",
        "ccc70322242e8917809ec128cd5128f3275f921085196792480fb671bffc354e",
    ): dict(
        TILE_M=64,
        TILE_N=64,
        TILE_K=64,
        GROUP_M=1,
        num_warps=4,
        num_stages=2,
        num_ctas=1,
    ),
    (
        "mm_kernel",
        "8480dcaecfb6961653bc7663fc0a86d0ae32b7387af378adf2d23cba4c9aeb12",
    ): dict(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
        SPLIT_K=1,
        num_warps=4,
        num_stages=2,
        num_ctas=1,
    ),
    (
        "sum_dim_kernel",
        "5259b9f60445ef4a99e85387bc1e4c7a2264681170c695c6fc9b94378fa6348b",
    ): dict(BLOCK_M=4, BLOCK_N=1024, num_warps=4, num_stages=2, num_ctas=1),
    (
        "softmax_backward_kernel_inner",
        "d7c6eef628fba956403c1f43f4a2c94dea1142ec8fa532359de5016e3bf9488d",
    ): dict(TILE_N=128, num_warps=4, num_stages=2, num_ctas=1),
    (
        "log_softmax_backward_kernel",
        "eb890f9b6d9caec2555d9d26a1c1b2ab091f65f3edc16f7c566e10a9c224d948",
    ): dict(BLOCK_M=16, BLOCK_N=512, num_warps=8, num_stages=2, num_ctas=1),
    ("log_softmax_kernel", _METAX_LOG_SOFTMAX_SOURCE): _METAX_LOG_SOFTMAX_CONFIG,
    ("log_softmax_backward_kernel", _METAX_LOG_SOFTMAX_SOURCE): (
        _METAX_LOG_SOFTMAX_CONFIG
    ),
}

# This published kernel uses TILE_N as an arange extent. The vendor heuristic
# can return a short reduction length directly (for example 3 instead of 4).
_POWER_OF_TWO_HEURISTICS = {
    (
        "mean_dim_kernel_non_inner",
        "c968ad14d3ba76ed87c29e5e9346a0e392cdaeaeb01c0774cfd7c5c489394d1c",
    ): ("TILE_N",),
}

_native_commands = []
_admitted_kernels = {}
_launched_kernels = {}
_audit_installed = False
_launcher_module = None
_device_artifacts = None
_inductor_autotuner = ContextVar("inductor_autotuner", default=None)


def _same_torch_cohort(manifest, torch):
    built_version = manifest["torch"]
    if built_version == torch.__version__:
        return True
    # An optional CUDA extension can use the same release of PyTorch as a
    # CPU-built NVIDIA wheel when both the source revision and C++ ABI match.
    built_release, _, built_variant = built_version.partition("+")
    revision = manifest.get("torch_git_version")
    return (
        manifest.get("backend", {}).get("vendor") == "nvidia"
        and built_variant == "cpu"
        and torch.version.cuda is not None
        and built_release == torch.__version__.partition("+")[0]
        and bool(revision)
        and revision == torch.version.git_version
        and manifest.get("torch_cxx11_abi") is torch.compiled_with_cxx11_abi()
    )


@cache
def _verified_singleton_tuner():
    for module, expected in _SINGLETON_TUNER_SOURCES.items():
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin or not Path(spec.origin).is_file():
            return False
        if hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != expected:
            return False
    return True


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
        tuner = _inductor_autotuner.get()
        if tuner is None or tuner.custom_kernel or source.fn is not tuner.fn:
            raise RuntimeError(f"kernel has no admitted FlagGems source: {source.fn}")
    else:
        if (
            hashlib.sha256(source.fn.src.encode()).hexdigest()
            != provenance["jit_sha256"]
        ):
            raise RuntimeError("FlagGems JIT source changed after admission")
        _launched_kernels[metadata.name] = provenance
    return _launcher_module.Launcher(source, metadata)


def _with_inductor_autotuner(method):
    @wraps(method)
    def with_autotuner(tuner, *args, **kwargs):
        token = _inductor_autotuner.set(tuner)
        try:
            return method(tuner, *args, **kwargs)
        finally:
            _inductor_autotuner.reset(token)

    with_autotuner._flagos_native_host = True
    return with_autotuner


def bind_inductor_launcher():
    """Use the wheel's launcher while Inductor binds a generated Triton kernel."""
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    # Both initial binding (including worker results) and coordinate descent
    # create launchers in the parent process, where the wheel's driver is active.
    for name in ("_make_launchers", "_coordinate_descent_tuning"):
        method = getattr(CachingAutotuner, name)
        if not getattr(method, "_flagos_native_host", False):
            setattr(CachingAutotuner, name, _with_inductor_autotuner(method))


def module_name(name):
    return "m_" + hashlib.sha256(name.encode()).hexdigest()[:16]


def function_identity(function):
    code = function.__code__
    return f"{function.__module__}:{function.__qualname__}:{Path(code.co_filename).name}:{code.co_firstlineno}"


def _validate_launcher(record, package, path, launcher):
    import triton
    from torch_fl import _build_config
    from torch_fl.native_host.backend import WHEEL_PACKAGE, get_backend, validate_record

    backend = get_backend(_build_config.ACCELERATOR)
    # Standalone build probes carry their verification inputs with the launcher.
    sources = path / "csrc"
    native_dir = sources if package != WHEEL_PACKAGE and sources.is_dir() else None
    validate_record(
        record,
        backend,
        Path(triton.__file__).parent,
        launcher.backend_contract(),
        native_dir=native_dir,
    )
    return backend


def install_launcher(package="fg_aot", *, enforce_native_compilation=True):
    global _audit_installed, _launcher_module, _device_artifacts
    from torch_fl import _env
    from torch_fl.flagos._streams import get_raw_stream
    from torch_fl.native_host.backends.driver import install_driver
    from torch_fl.native_host import device_artifacts

    required = enforce_native_compilation and _env.flag(
        device_artifacts.REQUIRE_ARTIFACTS
    )
    if required:
        # FlagGems builds an empty SQL registry during import even when there is
        # no tuning. Keep that initialization process-local in strict eager.
        os.environ["FLAGGEMS_DB_URL"] = "sqlite:///:memory:"
    package_module = importlib.import_module(package)
    path = Path(package_module.__file__).parent
    manifest = json.loads((path / "manifest.json").read_text())
    _launcher_module = importlib.import_module(package + "._launcher")
    backend = _validate_launcher(
        manifest.get("backend"), package, path, _launcher_module
    )
    install_driver(backend, package, get_raw_stream, kernel_launcher)
    if enforce_native_compilation:
        device_artifacts.stabilize_scan_key()
        _device_artifacts = device_artifacts.install(
            path / device_artifacts.ARTIFACT_DIRECTORY,
            required=required,
        )
        install_data = getattr(backend, "install_device_artifact_data", None)
        if required and install_data is not None:
            install_data(_device_artifacts.backend_data)
    if enforce_native_compilation and not _audit_installed:
        sys.addaudithook(audit_native_compilers)
        _audit_installed = True


def _pointwise_helpers(modules):
    from flag_gems.utils.device_info import get_device_capability

    shape = modules["flag_gems.utils.shape_utils"]
    pointwise = modules["flag_gems.utils.pointwise_dynamic"]
    return (
        shape.heuristics_for_tile_size,
        shape.heuristics_for_num_warps,
        shape.stride_order,
        pointwise._balanced_grid_partition,
        pointwise._tensor_inputs_all_complex,
        get_device_capability()[0],
    )


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
        self.package = self.manifest.get("package", "fg_aot")
        self.source_variant_functions = {}
        if "source_variants" in self.manifest:
            from torch_fl.native_host.source_variants import (
                load as load_source_variants,
            )

            bytecode = self.manifest.get("source_variant_bytecode")
            if _device_artifacts is not None and _device_artifacts.required:
                bytecode = bytecode or {}
            self.source_variant_functions = load_source_variants(
                self.manifest["source_variants"],
                self.manifest["modules"],
                package=self.path,
                bytecode=bytecode,
            )
        self.original_ops = {
            name: getattr(flag_gems, name)
            for name in self.manifest["ops"]
            if hasattr(flag_gems, name)
        }
        if (
            not _same_torch_cohort(self.manifest, torch)
            or self.manifest["triton"] != triton.__version__
        ):
            raise RuntimeError(
                "native hosts require their build-time Torch/FlagTree cohort"
            )
        _launcher = importlib.import_module(self.package + "._launcher")
        self.backend = _validate_launcher(
            self.manifest.get("backend"), self.package, self.path, _launcher
        )
        runtime = importlib.import_module(self.package + ".runtime")
        self.modules = {}
        self.originals = {}
        self.objects = {}
        self.pointwise_build_outputs = None
        self.closures = {
            identity: importlib.import_module(self.package + "." + record["module"])
            for identity, record in self.manifest.get("closures", {}).items()
        }
        self.provenance = []
        self.padding_ranks = self.manifest.get("padding_ranks", {})
        self.index_copy_ranks = self.manifest.get("index_copy_ranks", {})
        self.generated_kernel_sources = {
            (self.path / f"_flaggems_{family}_rank_{int(rank)}.py").resolve(): (
                name,
                generator,
            )
            for family, ranks, generator in (
                ("pad", self.padding_ranks, "flag_gems.ops.pad.generate_code"),
                (
                    "index_copy",
                    self.index_copy_ranks,
                    "flag_gems.ops.index_copy_.generate_code",
                ),
            )
            for rank, name in ranks.items()
        }
        self.LibEntry = LibEntry
        self.Pointwise = PointwiseDynamicFunction
        self.runtime = runtime
        _launcher.configure_runtime(
            str(Path(torch_fl.__file__).parent / "lib/libflagos.so")
        )
        for name, record in self.manifest["modules"].items():
            if (
                name in self.padding_ranks.values()
                or name in self.index_copy_ranks.values()
            ):
                source = (self.path / (name.rsplit(".", 1)[1] + ".py")).resolve()
                if (
                    self.generated_kernel_sources.get(source, (None, None))[0] != name
                    or name != f"{self.package}.{source.stem}"
                    or hashlib.sha256(source.read_bytes()).hexdigest()
                    != record["sha256"]
                ):
                    raise RuntimeError(f"generated rank source changed: {name}")
            original = importlib.import_module(name)
            source = Path(inspect.getsourcefile(original)).resolve()
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
        runtime.configure(
            self.resolve,
            self.pointwise_kernel,
            _pointwise_helpers(self.modules),
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
            generated = self.generated_kernel_sources.get(source)
            if "flag_gems" not in source.parts and generated is None:
                raise RuntimeError(f"device kernel is not from FlagGems: {source}")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            vendor = self.manifest["backend"]["vendor"]
            self.provenance.append(
                {
                    "kernel": jit.__name__,
                    "source": str(source),
                    "sha256": source_hash,
                    "jit_sha256": hashlib.sha256(jit.src.encode()).hexdigest(),
                }
            )
            if generated is not None:
                self.provenance[-1]["generator"] = generated[1]
            _admitted_kernels[id(jit)] = self.provenance[-1]
            eager_configs = getattr(self.backend, "eager_kernel_configs", {})
            key = jit.__name__, source_hash
            found = self.runtime.KernelEntry(
                value,
                entry,
                compile_before_enqueue=vendor in _COMPILE_BEFORE_ENQUEUE,
                permit_single_config=_verified_singleton_tuner(),
                eager_config=eager_configs.get(key, _PUBLISHED_EAGER_CONFIGS.get(key)),
                power_of_two_heuristics=_POWER_OF_TWO_HEURISTICS.get(key, ()),
            )
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
                if type(value).__name__ == "PointwiseDynamicFunction":
                    found.overloads = {}
                    found._kernel_info_cache = {}
                    found._capture_output_pool = {}
                    if not isinstance(value, self.Pointwise):
                        found._native_pointwise_module = kind.__module__
                        found._native_pointwise_hygon = kind.__module__ in {
                            "_hygon.utils.pointwise_dynamic",
                            "flag_gems.runtime.backend._hygon.utils.pointwise_dynamic",
                        }
                elif (
                    self.padding_ranks
                    and kind is self.originals["flag_gems.ops.pad"].PadFunction
                ):
                    found.overloads = {
                        rank: self.modules[name]._pad_wrapper
                        for rank, name in self.padding_ranks.items()
                    }
                elif (
                    self.index_copy_ranks
                    and kind
                    is self.originals["flag_gems.ops.index_copy_"].IndexCopyFunction
                ):
                    found.overloads = {
                        int(rank): self.modules[name]._index_copy_wrapper
                        for rank, name in self.index_copy_ranks.items()
                    }
            else:
                return value
        self.objects[id(value)] = found
        return found

    def pointwise_kernel(self, owner, ndim):
        # The upstream generator emits only its original device kernel here.
        # The rank-independent host adapter was compiled during the build.
        provider = getattr(
            owner, "_native_pointwise_module", "flag_gems.utils.pointwise_dynamic"
        )
        pointwise = self.originals[provider]
        cambricon = provider in {
            "_cambricon.utils.pointwise_dynamic",
            "flag_gems.runtime.backend._cambricon.utils.pointwise_dynamic",
        } and hashlib.sha256(Path(pointwise.__file__).read_bytes()).hexdigest() == (
            "d7ba2e471aa41b638c70b0ffa635e59aa4ce43ab7e4da8a00f95bf557ff5cd8f"
        )
        tsingmicro = (
            provider.removeprefix("flag_gems.runtime.backend.")
            == "_tsingmicro.utils.pointwise_dynamic"
            and (
                owner._scalar_fn.__module__.removeprefix("flag_gems.runtime.backend."),
                owner._scalar_fn.__name__,
            )
            in {
                ("_tsingmicro.ops.add", "add_func"),
                ("_tsingmicro.ops.add", "add_func_tensor_scalar"),
                ("_tsingmicro.ops.copy", "_copy_kernel"),
                ("_tsingmicro.ops.mul", "mul_func"),
                ("_tsingmicro.ops.mul", "mul_func_scalar"),
                ("_tsingmicro.ops.pow", "pow_func_tensor_scalar"),
                ("_tsingmicro.ops.rsqrt", "rsqrt_func"),
            }
            and hashlib.sha256(Path(pointwise.__file__).read_bytes()).hexdigest()
            == "c75ec27aab2176c1a85fa023137513073a11b950741741feaf679c1ec01c3276"
        )
        if cambricon or tsingmicro:
            # Keep the published scalar kernel and generate its host-facing
            # device entry with the shared pointwise ABI.
            scalar_provider = provider
            provider = "flag_gems.utils.pointwise_dynamic"
            pointwise = self.originals[provider]
        kunlunxin = provider in {
            "_kunlunxin.utils.pointwise_dynamic",
            "flag_gems.runtime.backend._kunlunxin.utils.pointwise_dynamic",
        } and hashlib.sha256(Path(pointwise.__file__).read_bytes()).hexdigest() == (
            "4a6edc2c57e3a8f1772430ad3cccbac55560d502c8520d0e4336826cfedebf56"
        )
        owner._native_pointwise_kunlunxin = kunlunxin
        single_kernel = (
            provider == "flag_gems.utils.pointwise_dynamic"
            or getattr(owner, "_native_pointwise_hygon", False)
            or kunlunxin
        )

        name = f"{owner._scalar_fn.__name__}_kernel_rank_{ndim}"
        from torch_fl.native_host.device_artifacts import (
            PointwiseArtifactLoader,
            pointwise_key,
        )

        key = None
        if _device_artifacts is not None or self.pointwise_build_outputs is not None:
            original_provider = getattr(
                owner, "_native_pointwise_module", "flag_gems.utils.pointwise_dynamic"
            )
            key = pointwise_key(owner, original_provider, ndim)
        artifact = (
            None
            if _device_artifacts is None
            else _device_artifacts.pointwise_module(key)
        )
        if artifact is None:
            config = owner.config
            if tsingmicro and not hasattr(config, "balance_grid"):
                config = types.SimpleNamespace(**vars(config), balance_grid=False)
            generator = pointwise.ModuleGenerator(
                owner.fx, owner._scalar_fn, ndim, name, "unused_host", config
            )
            code = pointwise.IndentedBuffer()
            generator.generate_imports(code)
            if not single_kernel:
                helpers = [
                    node.module
                    for node in ast.walk(ast.parse(code.getvalue()))
                    if isinstance(node, ast.ImportFrom)
                    and any(
                        alias.name == "heuristics_for_tile_size_with_mmu_constraint"
                        for alias in node.names
                    )
                ]
                if len(helpers) != 1:
                    raise RuntimeError(
                        f"unsupported published pointwise host ABI: {provider}"
                    )
                owner._native_pointwise_helpers = helpers[0]
            if owner.config.prefer_1d_tile:
                generator.kernel_gen.codegen_1d_tile(code)
            else:
                generator.kernel_gen.codegen_nd_tile(code)
            text = code.getvalue()
            if cambricon or tsingmicro:
                text = text.replace(
                    "from utils.pointwise_dynamic import ",
                    f"from {scalar_provider} import ",
                )
            for source in self.manifest.get("source_variants", {}).get("entries", ()):
                if (
                    source.get("shared_pointwise")
                    and source["module"] == owner._scalar_fn.__module__
                ):
                    # The published generator drops the relative-import level when
                    # collecting this provider's dependencies. Use the same shared
                    # pointwise module that loaded the verified scalar functions.
                    text = text.replace(
                        "from utils.pointwise_dynamic import pointwise_dynamic\n",
                        "from flag_gems.utils.pointwise_dynamic import pointwise_dynamic\n",
                    )
            digest = hashlib.sha256(text.encode()).hexdigest()
            path = pointwise.code_cache_dir() / f"native_host_kernel_{digest}.py"
            cache_writer = (
                self.originals["flag_gems.utils.pointwise_dynamic"]
                if kunlunxin
                else pointwise
            )
            cache_writer.write_atomic(str(path), text)
            spec = importlib.util.spec_from_file_location(f"_fg_device_{digest}", path)
            if self.pointwise_build_outputs is not None:
                self.pointwise_build_outputs[key] = {
                    "path": str(path),
                    "sha256": digest,
                    "helpers": getattr(owner, "_native_pointwise_helpers", None),
                }
        else:
            path, helpers = artifact
            if helpers is not None:
                owner._native_pointwise_helpers = helpers
            digest = path.stem.rsplit("_", 1)[1]
            module_name = f"_fg_device_{digest}"
            spec = importlib.util.spec_from_file_location(
                module_name,
                path,
                loader=PointwiseArtifactLoader(module_name, str(path)),
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__dict__.update(owner._scalar_fn.__globals__)
        module.__dict__[owner._scalar_fn.__name__] = owner._scalar_fn
        reuse_runtime_floats = self.manifest["backend"]["vendor"] == "hygon" and (
            owner._scalar_fn.__name__
            in {"true_div_func_tensor_scalar", "addcdiv_kernel"}
        )
        entries = []
        for kernel_name in (name, name + "_stride_constexpr"):
            if not hasattr(module, kernel_name):
                continue
            entry = getattr(module, kernel_name)
            self.provenance.append(
                {
                    "kernel": kernel_name,
                    "source": str(path),
                    "sha256": digest,
                    "generator": provider + ".KernelGenerator",
                    "scalar_kernel": owner._scalar_fn.__name__,
                    "jit_sha256": hashlib.sha256(
                        entry.jit_function.src.encode()
                    ).hexdigest(),
                }
            )
            _admitted_kernels[id(entry.jit_function)] = self.provenance[-1]
            entries.append(
                self.runtime.KernelEntry(
                    entry,
                    entry,
                    reuse_runtime_floats,
                    compile_before_enqueue=self.manifest["backend"]["vendor"]
                    in _COMPILE_BEFORE_ENQUEUE,
                    permit_single_config=_verified_singleton_tuner(),
                )
            )
        # Generated device code is admitted only through the wheel's generator.
        return entries[0] if single_kernel else tuple(entries)

    def op(self, name):
        return self.resolve(self.original_ops[name])

    def _artifacts(self):
        return {
            str(path.relative_to(self.path)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in self.path.rglob("*.so")
            # Numerical vendor binaries have their own immutable manifest and
            # vendor audit; they are not compiled source-host artifacts.
            if path.parent != self.path / "metax_vendor"
        }

    def audit(self):
        if self._artifacts() != self.artifacts:
            raise RuntimeError("host artifacts changed during execution")
        hits = {}
        if self.manifest.get("source_variants"):
            module = importlib.import_module(self.package + ".typed._source_hosts")
            hits = dict(module.source_variant_hits())
        return {
            "host_artifacts": self.artifacts,
            "backend": self.manifest["backend"],
            "source_variants": self.manifest.get("source_variants"),
            "matmul_adaptation": self.manifest.get("matmul_adaptation"),
            "copy_adaptation": self.manifest.get("copy_adaptation"),
            "source_variant_hits": hits,
            "subprocess_commands": _native_commands,
            "pointwise_sources": self.pointwise_build_outputs or {},
            "device_artifacts": None
            if _device_artifacts is None
            else _device_artifacts.audit(),
            "launched_kernels": _launched_kernels,
            "enqueued_kernels": _launcher_module.enqueue_stats()
            if hasattr(_launcher_module, "enqueue_stats")
            else {},
        }

    def activate(self):
        import flag_gems
        import flag_gems.ops
        import torch
        from torch_fl import _build_config

        matched_cpu_build = self.manifest["torch"].endswith(
            "+cpu"
        ) and _same_torch_cohort(self.manifest, torch)
        if (
            torch.version.cuda is not None
            and not matched_cpu_build
            or "boxing" in _build_config.KERNELS
        ):
            raise RuntimeError("native hosts require the recorded CPU Torch cohort")
        if getattr(_build_config, "NATIVE_HOSTS", False):
            torch.ops.flagos.strict_host_routes(True)
        if _device_artifacts is not None and _device_artifacts.required:
            _device_artifacts.block_tuning_writes()
        for name in self.manifest["ops"]:
            function = self.op(name)
            setattr(flag_gems, name, function)
            if hasattr(flag_gems.ops, name):
                setattr(flag_gems.ops, name, function)
        if self.manifest.get("typed_hosts"):
            from .typed import activate

            self.typed = activate(self)
