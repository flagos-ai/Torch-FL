# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2] / "torch_fl/native_host/device_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("native_device_artifacts_test", SOURCE)
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


def compiler_output(path, key, binary=b"device binary"):
    group = path / "compiler-directory"
    group.mkdir(parents=True)
    (group / "kernel.json").write_text(json.dumps({"hash": key}))
    (group / "kernel.fatbin").write_bytes(binary)
    (group / "__grp__kernel.json").write_text("{}")
    (group / "kernel.llir").write_text("compiler intermediate")
    return path


def test_build_stages_only_verified_device_artifacts(tmp_path):
    key = "1" * 64
    source = compiler_output(tmp_path / "build", key)
    bundle = tmp_path / "bundle"
    artifacts.bundle([source], bundle, "fatbin")

    package = tmp_path / "wheel"
    package.mkdir()
    artifacts.stage(bundle, package)
    installed = package / artifacts.ARTIFACT_DIRECTORY
    assert {p.name for p in installed.rglob("*") if p.is_file()} == {
        "index.json",
        "kernel.json",
        "kernel.fatbin",
    }
    (bundle / key / "kernel.fatbin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifacts.stage(bundle, package)
    assert (installed / key / "kernel.fatbin").read_bytes() == b"device binary"

    different = compiler_output(tmp_path / "different-build", key, b"different")
    with pytest.raises(ValueError, match="Conflicting device artifacts"):
        artifacts.bundle([source, different], tmp_path / "combined", "fatbin")


def test_tsingmicro_bundle_merges_identical_kernels_from_separate_builds(tmp_path):
    key = "2" * 64
    builds = [tmp_path / name for name in ("first", "second")]
    for build in builds:
        group = build / "compile-key"
        group.mkdir(parents=True)
        binary = group / "kernel.so"
        binary.write_bytes(b"device binary")
        (group / "kernel.json").write_text(
            json.dumps(
                {
                    "hash": key,
                    "target": {"backend": "txda", "arch": 1, "warp_size": 16},
                    "kernel_path": str(binary),
                    "shared": 0,
                }
            )
        )
    output = tmp_path / "bundle"
    artifacts.bundle(builds, output, "so")
    records, _ = artifacts._read_index(output)
    assert list(records["artifacts"]) == [key]

    metadata = builds[1] / "compile-key/kernel.json"
    payload = json.loads(metadata.read_bytes())
    payload["shared"] = 64
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Conflicting device artifacts"):
        artifacts.bundle(builds, tmp_path / "conflict", "so")


def test_required_device_artifacts_load_without_compiler_fallback(
    tmp_path, monkeypatch
):
    full_key = "compiler source, target, options and environment"
    digest = hashlib.sha256(full_key.encode()).hexdigest()
    output = compiler_output(tmp_path / "build", digest)
    bundle = tmp_path / "bundle"
    artifacts.bundle([output], bundle, "fatbin")

    class ASTSource:
        name = "kernel"

        def parse_options(self):
            return {"num_warps": 4}

    def select_key(src, backend, options, env):
        assert backend.binary_ext == "fatbin"
        assert options == {"num_warps": 4, "num_stages": 2}
        assert env == {"TOOLCHAIN": "published"}
        return full_key if src.name == "kernel" else src.name

    def compile_original(*args, **kwargs):
        raise AssertionError("The original compiler must not run")

    compiler = ModuleType("triton.compiler.compiler")
    compiler.ASTSource = ASTSource
    compiler.get_cache_key = select_key
    compiler.get_cache_invalidating_env_vars = lambda: {"TOOLCHAIN": "published"}
    compiler.make_backend = lambda target: SimpleNamespace(
        binary_ext="fatbin", parse_options=lambda options: options
    )
    compiler.CompiledKernel = lambda _src, files, _key: Path(
        files["kernel.fatbin"]
    ).read_bytes()
    compiler.compile = compile_original
    compiler.get_cache_manager = compile_original
    triton = ModuleType("triton")
    triton.compiler = ModuleType("triton.compiler")
    runtime_jit = ModuleType("triton.runtime.jit")
    for module in (triton, triton.compiler, compiler, runtime_jit):
        module.compile = compile_original
        monkeypatch.setitem(sys.modules, module.__name__, module)

    installed = artifacts.install(bundle, required=True)
    source = ASTSource()
    options = {"num_stages": 2}
    assert (
        runtime_jit.compile(source, target="enflame", options=options)
        == b"device binary"
    )
    assert installed.audit()["loaded_specializations"] == 1
    source.name = "unexpected"
    with pytest.raises(
        RuntimeError, match="Missing precompiled Native Host device kernel"
    ):
        triton.compile(source, target="enflame", options=options)
    with pytest.raises(RuntimeError, match="Runtime device compilation is forbidden"):
        compiler.get_cache_manager(digest)
    with pytest.raises(RuntimeError, match="wheel has no precompiled"):
        artifacts.install(tmp_path / "unbundled", required=True)


def test_wheel_loads_packaged_pointwise_bytecode_at_the_installed_source_path(tmp_path):
    digest = "7" * 64
    selector = "9" * 64
    source = tmp_path / "generated.py"
    source.write_text("def kernel_source():\n    return 42\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    report = tmp_path / "build.json"
    report.write_text(
        json.dumps(
            {
                "native_host": {
                    "pointwise_sources": {
                        selector: {
                            "path": str(source),
                            "sha256": source_sha,
                            "helpers": None,
                        },
                    },
                },
            },
        ),
    )
    output = compiler_output(tmp_path / "build", digest)
    bundle = tmp_path / "device-artifacts"
    artifacts.bundle([output], bundle, "fatbin", pointwise_reports=[report])
    package = tmp_path / "wheel"
    package.mkdir()
    artifacts.stage(bundle, package)

    installed = artifacts.DeviceArtifacts(
        package / artifacts.ARTIFACT_DIRECTORY, None, required=True
    )
    module_path, helpers = installed.pointwise_module(selector)
    assert helpers is None
    name = "native_artifact_test_kernel_source"
    loader = artifacts.PointwiseArtifactLoader(name, str(module_path))
    spec = importlib.util.spec_from_file_location(name, module_path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    assert module.kernel_source() == 42
    assert module.kernel_source.__code__.co_filename == str(module_path)
    assert not (module_path.parent / "__pycache__").exists()

    module_path.with_suffix(".pyc").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="pointwise artifact checksum mismatch"):
        installed.pointwise_module(selector)
    with pytest.raises(
        RuntimeError, match="Missing packaged Native Host pointwise module"
    ):
        installed.pointwise_module("uncovered")


def test_pinned_legacy_cambricon_compiler_uses_its_published_key(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "3.4.0+mlu2.1.1")
    compiler = SimpleNamespace(triton_key=lambda: "compiler")
    source = SimpleNamespace(hash=lambda: "source")
    backend = SimpleNamespace(hash=lambda: "backend")
    options = SimpleNamespace(hash=lambda: "options")
    key = artifacts._specialization_key_function(compiler)
    env = {"Z": "2", "A": "1"}
    assert key(source, backend, options, env) == (
        "compiler-source-backend-options-[('A', '1'), ('Z', '2')]"
    )
