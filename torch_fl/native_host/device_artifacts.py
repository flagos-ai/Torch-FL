# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Package and load device code prepared with the published compiler."""

import _imp
import argparse
from collections import Counter
from functools import wraps
import hashlib
import importlib
from importlib.machinery import SourceFileLoader, SourcelessFileLoader
import importlib.metadata
import json
from pathlib import Path
import py_compile
import re
import shutil
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace


INDEX_NAME = "index.json"
ARTIFACT_DIRECTORY = "device_kernels"
REQUIRE_ARTIFACTS = "FLAGOS_NATIVE_HOST_REQUIRE_DEVICE_ARTIFACTS"
BUILD_BUNDLE = "FLAGOS_NATIVE_HOST_DEVICE_BUNDLE"
_SHA256 = re.compile("[0-9a-f]{64}")
_SCAN_SOURCE_SHA256 = "ef7f010fdc53b3eacef469326bffd030c961887e5fa0ec1c71f861a62793a502"


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def pointwise_key(owner, provider, rank):
    fn = owner._scalar_fn
    recipe = (
        provider,
        fn.__module__,
        getattr(fn, "__qualname__", fn.__name__),
        _digest(fn.src.encode()),
        owner.fx.signature(outputs_in_arg=True),
        vars(owner.config),
        rank,
    )
    return _digest(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())


def _pointwise_source(digest):
    return Path("pointwise") / f"native_host_kernel_{digest}.py"


def _target_key(target):
    return json.dumps(
        (target.backend, target.arch, target.warp_size), separators=(",", ":")
    )


def _record_backend_fingerprints(targets):
    compiler = importlib.import_module("triton.compiler.compiler")
    fingerprints = {}
    for descriptor in targets.values():
        target = compiler.GPUTarget(**descriptor)
        fingerprints[_target_key(target)] = compiler.make_backend(target).hash()
    return fingerprints


def _specialization_key_function(compiler):
    if hasattr(compiler, "get_cache_key"):
        return compiler.get_cache_key
    if importlib.metadata.version("triton") != "3.4.0+mlu2.1.1":
        raise RuntimeError("This compiler does not expose a device specialization key")

    def cambricon_key(source, backend, options, env):
        # The pinned Cambricon compiler predates get_cache_key().
        return (
            f"{compiler.triton_key()}-{source.hash()}-{backend.hash()}-"
            f"{options.hash()}-{sorted(env.items())}"
        )

    return cambricon_key


class PointwiseArtifactLoader(SourceFileLoader):
    """Load the wheel's verified bytecode while preserving Triton's source path."""

    def get_code(self, fullname):
        source = Path(self.path)
        bytecode = SourcelessFileLoader(fullname, str(source.with_suffix(".pyc")))
        code = bytecode.get_code(fullname)
        _imp._fix_co_filename(code, str(source))
        return code


def stabilize_scan_key():
    """Give the published FlagGems scan's Python constexpr a portable identity."""
    from triton.language.core import constexpr

    original = constexpr.__repr__
    if getattr(original, "_flagos_stable_scan", False):
        return

    @wraps(original)
    def repr_constexpr(value):
        fn = value.value
        if (
            type(fn) is FunctionType
            and fn.__module__ == "flag_gems.ops.cumsum"
            and fn.__qualname__ == "get_scan_accum_type"
            and fn.__closure__ is None
            and fn.__defaults__ is None
            and fn.__kwdefaults__ is None
            and _digest(Path(fn.__code__.co_filename).read_bytes())
            == _SCAN_SOURCE_SHA256
        ):
            # Triton hashes the repr of captured globals. Python's default
            # function repr embeds an address that changes on every import.
            return f"constexpr[{fn.__module__}.{fn.__qualname__}:{_SCAN_SOURCE_SHA256}]"
        return original(value)

    repr_constexpr._flagos_stable_scan = True
    constexpr.__repr__ = repr_constexpr


def _read_index(directory):
    payload = (directory / INDEX_NAME).read_bytes()
    index = json.loads(payload)
    if index["version"] != 1 or not index["artifacts"]:
        raise ValueError("Invalid Native Host device artifact index")
    for key, files in index["artifacts"].items():
        if not _SHA256.fullmatch(key) or len(files) != 2:
            raise ValueError(f"Invalid Native Host device artifact {key}")
        for name, digest in files.items():
            if Path(name).name != name or not _SHA256.fullmatch(digest):
                raise ValueError(f"Invalid Native Host device artifact file {name}")
    for key, module in index.get("pointwise", {}).items():
        hashes = (key, module["sha256"], module.get("bytecode_sha256", key))
        if not all(_SHA256.fullmatch(value) for value in hashes):
            raise ValueError(f"Invalid Native Host pointwise artifact {key}")
    return index, _digest(payload)


def bundle(
    sources,
    destination,
    binary_extension,
    *,
    pointwise_reports=(),
    record_backend=False,
):
    """Stage executables and launch metadata from an isolated compiler build."""
    if not binary_extension.isascii() or not binary_extension.isalnum():
        raise ValueError(f"Invalid device binary extension {binary_extension}")
    records = {}
    targets = {}
    for source in sources:
        for metadata in sorted(source.glob("*/*.json")):
            if metadata.name.startswith("__grp__"):
                continue
            payload = json.loads(metadata.read_bytes())
            key = payload["hash"]
            if record_backend:
                target = payload["target"]
                targets[json.dumps(target, sort_keys=True)] = target
            if not _SHA256.fullmatch(key):
                raise ValueError(f"Invalid compiler key in {metadata}")
            binary = metadata.with_suffix(f".{binary_extension}")
            files = {
                path.name: _digest(path.read_bytes()) for path in (metadata, binary)
            }
            if key in records:
                previous_files, previous_metadata, _ = records[key]
                if files == previous_files:
                    continue
                if (
                    binary_extension == "so"
                    and payload.get("target", {}).get("backend") == "txda"
                    and files.keys() == previous_files.keys()
                    and files[binary.name] == previous_files[binary.name]
                ):
                    previous = json.loads(previous_metadata.read_bytes())
                    old_path = previous.pop("kernel_path", None)
                    new_path = payload.pop("kernel_path", None)
                    # The native TX launcher uses the loaded function handle.
                    # The published Python launcher records the offline build path.
                    if (
                        isinstance(old_path, str)
                        and isinstance(new_path, str)
                        and Path(old_path).is_absolute()
                        and Path(new_path).is_absolute()
                        and Path(old_path).name == Path(new_path).name
                        and payload == previous
                    ):
                        continue
                raise ValueError(f"Conflicting device artifacts for compiler key {key}")
            records[key] = (files, metadata, binary)
    if not records:
        raise ValueError("No device artifacts in the supplied compiler build outputs")

    destination.mkdir()
    for key, (_, metadata, binary) in records.items():
        entry = destination / key
        entry.mkdir()
        shutil.copyfile(metadata, entry / metadata.name)
        shutil.copyfile(binary, entry / binary.name)
    index = {"version": 1, "artifacts": {key: item[0] for key, item in records.items()}}
    if pointwise_reports:
        index["pointwise"] = _bundle_pointwise(pointwise_reports, destination)
    if record_backend:
        index["backend_fingerprints"] = _record_backend_fingerprints(targets)
    (destination / INDEX_NAME).write_text(json.dumps(index, sort_keys=True) + "\n")


def _bundle_pointwise(reports, destination):
    records = {}
    for report in reports:
        native = json.loads(report.read_bytes())["native_host"]
        for key, entry in native["pointwise_sources"].items():
            source = Path(entry["path"])
            payload = source.read_bytes()
            digest = _digest(payload)
            if digest != entry["sha256"] or not _SHA256.fullmatch(key):
                raise ValueError(f"Invalid Native Host pointwise source {source}")
            metadata = {"sha256": digest, "helpers": entry["helpers"]}
            if key in records and records[key] != metadata:
                raise ValueError(f"Conflicting pointwise source for {key}")
            records[key] = metadata
            output = destination / _pointwise_source(digest)
            output.parent.mkdir(exist_ok=True)
            output.write_bytes(payload)
    return records


class DeviceArtifacts:
    def __init__(self, directory, compiler, *, required):
        self.directory = directory
        self.compiler = compiler
        self.specialization_key = (
            _specialization_key_function(compiler) if compiler is not None else None
        )
        index, self.index_sha256 = _read_index(directory)
        self.records = index["artifacts"]
        self.pointwise = index.get("pointwise", {})
        self.backend_fingerprints = index.get("backend_fingerprints", {})
        self.backend_data = index.get("backend_data")
        self.loaded = Counter()
        self.required = required
        self.tuning_blocked = False

    def audit(self):
        return {
            "required": self.required,
            "index_sha256": self.index_sha256,
            "available_specializations": len(self.records),
            "loaded_specializations": len(self.loaded),
            "runtime_tuning_blocked": self.tuning_blocked,
        }

    def block_tuning_writes(self):
        from flag_gems.utils.models.sql import SQLPersistantModel

        def reject(*_args, **_kwargs):
            raise RuntimeError(
                "Runtime tuning is forbidden for precompiled Native Host"
            )

        SQLPersistantModel.put_config = reject
        SQLPersistantModel.put_benchmark = reject
        self.tuning_blocked = True

    def pointwise_module(self, key):
        entry = self.pointwise.get(key)
        if entry is None:
            if self.required:
                raise RuntimeError(
                    f"Missing packaged Native Host pointwise module {key}"
                )
            return None
        path = self.directory / _pointwise_source(entry["sha256"])
        for source, digest in (
            (path, entry["sha256"]),
            (path.with_suffix(".pyc"), entry["bytecode_sha256"]),
        ):
            if _digest(source.read_bytes()) != digest:
                raise RuntimeError(
                    f"Native Host pointwise artifact checksum mismatch for {source}"
                )
        return path, entry["helpers"]

    def lookup(self, src, target, options, env_vars):
        compiler = self.compiler
        if not isinstance(src, compiler.ASTSource):
            return None, None
        if target is None:
            driver = compiler.driver.active
            current_target = getattr(driver, "get_current_target_inside", None)
            target = (current_target or driver.get_current_target)()
        backend = compiler.make_backend(target)
        raw_options = dict(options or {}, **src.parse_options())
        flagprism = getattr(compiler, "_flagprism", None)
        if flagprism is not None:
            flagprism.apply_compile_options(raw_options)
        options = backend.parse_options(raw_options)
        if env_vars is None:
            env_vars = compiler.get_cache_invalidating_env_vars()
        key_backend = backend
        if self.backend_fingerprints:
            fingerprint = self.backend_fingerprints.get(_target_key(target))
            if fingerprint is None:
                raise RuntimeError(
                    f"Native Host device bundle has no compiler fingerprint for {target}"
                )
            # The pinned compiler hashes the backend's version string. Use the
            # recorded build value so loading a binary needs no SDK compiler.
            key_backend = SimpleNamespace(hash=lambda: fingerprint)
        key = _digest(
            self.specialization_key(src, key_backend, options, env_vars).encode()
        )
        records = self.records.get(key)
        if records is None:
            return None, key
        metadata = f"{src.name[:150]}.json"
        binary = Path(metadata).with_suffix(f".{backend.binary_ext}").name
        if records.keys() != {metadata, binary}:
            raise RuntimeError(f"Native Host device artifact file mismatch for {key}")
        files = {}
        for name, expected in records.items():
            path = self.directory / key / name
            if _digest(path.read_bytes()) != expected:
                raise RuntimeError(
                    f"Native Host device artifact checksum mismatch for {path}"
                )
            files[name] = str(path)
        kernel = compiler.CompiledKernel(src, files, key)
        self.loaded[key] += 1
        return kernel, key


def install(directory, *, required):
    """Load packaged binaries through the installed compiler's specialization key."""
    if not (directory / INDEX_NAME).is_file():
        if required:
            raise RuntimeError(
                "This Native Host wheel has no precompiled device kernels"
            )
        return None
    import triton

    compiler = importlib.import_module("triton.compiler.compiler")
    runtime_jit = importlib.import_module("triton.runtime.jit")
    artifacts = DeviceArtifacts(directory, compiler, required=required)
    original = compiler.compile

    @wraps(original)
    def compile_or_load(src, target=None, options=None, _env_vars=None):
        kernel, key = artifacts.lookup(src, target, options, _env_vars)
        if kernel is not None:
            return kernel
        if required:
            name = getattr(src, "name", str(src))
            raise RuntimeError(
                f"Missing precompiled Native Host device kernel {name} ({key})"
            )
        kwargs = {} if _env_vars is None else {"_env_vars": _env_vars}
        return original(src, target=target, options=options, **kwargs)

    for module in (triton, triton.compiler, compiler, runtime_jit):
        if hasattr(module, "compile"):
            module.compile = compile_or_load
    if required:

        def reject_compiler_cache(*_args, **_kwargs):
            raise RuntimeError("Runtime device compilation is forbidden in Native Host")

        # An already-bound JIT function can retain the original compile routine.
        compiler.get_cache_manager = reject_compiler_cache
    return artifacts


def stage(bundle_directory, package, *, backend_data=None):
    """Copy the verified closure and compile its host bytecode in the build."""
    output = package / ARTIFACT_DIRECTORY
    if bundle_directory is None:
        if output.is_dir():
            shutil.rmtree(output)
        return
    source = Path(bundle_directory).resolve()
    if output.resolve() in source.parents:
        raise ValueError(
            "The build bundle cannot be inside the staged artifact directory"
        )
    index, _ = _read_index(source)
    if backend_data is not None:
        index["backend_data"] = backend_data
    files = {
        Path(key) / name: digest
        for key, records in index["artifacts"].items()
        for name, digest in records.items()
    }
    modules = index.get("pointwise", {})
    files.update(
        {
            _pointwise_source(entry["sha256"]): entry["sha256"]
            for entry in modules.values()
        }
    )

    with TemporaryDirectory(dir=package) as temporary:
        prepared = Path(temporary) / ARTIFACT_DIRECTORY
        prepared.mkdir()
        for name, digest in files.items():
            payload = (source / name).read_bytes()
            if _digest(payload) != digest:
                raise ValueError(
                    f"Native Host device artifact checksum mismatch for {source / name}"
                )
            target = prepared / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(payload)
        for entry in modules.values():
            target = prepared / _pointwise_source(entry["sha256"])
            bytecode = target.with_suffix(".pyc")
            if not bytecode.exists():
                py_compile.compile(
                    str(target),
                    cfile=str(bytecode),
                    dfile=target.name,
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
                )
            entry["bytecode_sha256"] = _digest(bytecode.read_bytes())
        (prepared / INDEX_NAME).write_text(json.dumps(index, sort_keys=True) + "\n")
        if output.is_dir():
            shutil.rmtree(output)
        prepared.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-extension", required=True)
    parser.add_argument("--pointwise-report", type=Path, action="append", default=[])
    parser.add_argument(
        "--record-backend-fingerprints",
        action="store_true",
        help="record compiler tool versions in the pinned build environment",
    )
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    bundle(
        args.sources,
        args.output,
        args.binary_extension,
        pointwise_reports=args.pointwise_report,
        record_backend=args.record_backend_fingerprints,
    )


if __name__ == "__main__":
    main()
