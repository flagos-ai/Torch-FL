# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.metadata
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import setuptools

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, the DCU/MUSA interpreter
    tomllib = None
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parents[2]

KERNEL_SWITCHES = (
    "FLAGOS_BUILD_VENDOR",
    "FLAGOS_BUILD_FLAGGEMS",
    "FLAGOS_BUILD_BOXING",
    "FLAGOS_BUILD_FLAGGEMS_CPP",
    "FLAGOS_BUILD_TILEOPS",
)


def _load_setup(monkeypatch):
    kwargs = {}
    monkeypatch.setattr(setuptools, "setup", lambda **values: kwargs.update(values))
    namespace = runpy.run_path(str(ROOT / "setup.py"))
    return namespace, kwargs


@pytest.mark.parametrize("enabled", [False, True])
def test_cython_is_only_a_native_host_build_requirement(
    monkeypatch, enabled, clean_kernel_env
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", str(int(enabled)))
    monkeypatch.setattr(sys, "argv", ["setup.py", "egg_info"])
    _, kwargs = _load_setup(monkeypatch)
    assert kwargs["setup_requires"] == (["Cython==3.2.4"] if enabled else [])
    source = (ROOT / "pyproject.toml").read_text()
    build_system = source.split("[build-system]", 1)[1].split("[project]", 1)[0]
    assert "Cython" not in build_system


def test_native_host_package_data_only_includes_generated_manifests(
    monkeypatch, tmp_path
):
    _, kwargs = _load_setup(monkeypatch)
    package = tmp_path / "torch_fl"
    generated = package / "native_host/_generated"
    generated.mkdir(parents=True)
    for name in (
        "source_variants.json",
        "vendor_catalog.json",
        "vendor_precision.json",
    ):
        (generated.parent / name).write_text("{}")
    manifest = generated / "vendor_manifest.json"
    manifest.write_text("{}")
    command = build_py(
        Distribution(
            {
                "package_data": kwargs["package_data"],
                "include_package_data": kwargs["include_package_data"],
            }
        )
    )
    command.ensure_finalized()
    command.analyze_manifest()
    assert command.find_data_files("torch_fl", str(package)) == [str(manifest)]


def test_preflight_is_packaged_as_a_standalone_script():
    source = (ROOT / "pyproject.toml").read_text()
    assert 'script-files = ["scripts/tools/torch-fl-preflight"]' in source


def test_iluvatar_native_build_uses_active_cpu_torch(monkeypatch, tmp_path):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "iluvatar")
    # A legacy image variable must not redirect the native build to vendor torch.
    monkeypatch.setenv("FLAGOS_VENDOR_TORCH_LIB", str(tmp_path / "vendor/lib"))
    cpu_root = tmp_path / "venv/torch"
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __file__=str(cpu_root / "__init__.py"),
            _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
        ),
    )
    namespace, _ = _load_setup(monkeypatch)
    assert namespace["get_pytorch_dir"]() == str(cpu_root)
    assert namespace["_torch_cxx11_abi"]() == "1"


def test_iluvatar_manifest_records_official_cpu_runtime(
    monkeypatch, tmp_path, clean_kernel_env
):
    """Describe Iluvatar defaults without inheriting another runner's switches."""
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "iluvatar")
    namespace, _ = _load_setup(monkeypatch)
    root = tmp_path / "src"
    (root / "torch_fl").mkdir(parents=True)
    monkeypatch.setitem(
        namespace["_write_compatibility_manifest"].__globals__, "BASE_DIR", str(root)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "2.10.0+cpu" if name == "torch" else None,
    )
    namespace["_write_compatibility_manifest"](
        "2.10.0+corex4.5.0", _build_config("iluvatar", "flaggems", "vendor")
    )
    manifest = json.loads((root / "torch_fl/compatibility.json").read_text())
    assert manifest["build"]["distributions"]["torch"] == "2.10.0+cpu"
    assert manifest["build"]["torch_cxx11_abi"] is True
    assert manifest["build"]["bundle_libdir"] == "lib"
    assert manifest["build"]["vendor_torch_libraries"] is False


@pytest.fixture
def clean_kernel_env(monkeypatch):
    """No build switch or vendor cohort inherited from the runner."""
    monkeypatch.delenv("FLAGOS_ACCELERATOR", raising=False)
    for name in (
        *KERNEL_SWITCHES,
        "FLAGOS_BUILD_NATIVE_HOST",
        "FLAGOS_VENDOR_TORCH_LIB",
        "FLAGOS_VENDOR_TORCH_VERSION",
        "FLAGOS_SKIP_CUDA_ASSETS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "mudnn_header", [False, True], ids=["missing-mudnn", "full-sdk"]
)
def test_build_ext_stages_effective_musa_build_config(
    monkeypatch, tmp_path, clean_kernel_env, mudnn_header
):
    if shutil.which("cmake") is None:
        pytest.skip("CMake is unavailable")
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "mthreads")
    monkeypatch.setenv("FLAGOS_BUILD_VENDOR", "1")
    _, setup_kwargs = _load_setup(monkeypatch)
    assert "scripts" not in setup_kwargs
    source_root = tmp_path / "source"
    package_root = source_root / "torch_fl"
    package_root.mkdir(parents=True)
    sdk = tmp_path / "musa"
    (sdk / "lib").mkdir(parents=True)
    (sdk / "lib/libmudnn.so").touch()
    (sdk / "lib/libmurand.so").touch()
    (sdk / "include/mudnncxx").mkdir(parents=True)
    (sdk / "include/murand.h").touch()
    if mudnn_header:
        (sdk / "include/mudnncxx/mudnn.h").touch()
    # Configure with the production SDK probe, registration selector, and
    # metadata writer. A language-free project exercises setuptools staging
    # on hosts without the MUSA compiler or device libraries.
    (source_root / "csrc").mkdir()
    (source_root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\n"
        "project(musa_fixture LANGUAGES NONE)\n"
        f'include("{ROOT}/cmake/FlagosPlatforms.cmake")\n'
        f'set(MUSA_HOME "{sdk}")\n'
        'set(CMAKE_FIND_LIBRARY_PREFIXES "lib")\n'
        'set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")\n'
        f'include("{ROOT}/cmake/MusaKernels.cmake")\n'
        "add_subdirectory(csrc)\n"
    )
    (source_root / "csrc/CMakeLists.txt").write_text(
        f'include("{ROOT}/cmake/FlagosRegistration.cmake")\n'
        f'flagos_write_registration("{ROOT}/csrc/aten" '
        '"${PROJECT_BINARY_DIR}/register_ops.inc")\n'
        'flagos_write_build_config("${PROJECT_BINARY_DIR}/build-config.json" kernels)\n'
        'install(CODE [[message(STATUS "Fixture build configured")]])\n'
    )

    command_type = setup_kwargs["cmdclass"]["build_ext"]
    setup_globals = command_type.run.__globals__
    monkeypatch.setitem(setup_globals, "BASE_DIR", str(source_root))
    monkeypatch.setitem(setup_globals, "get_pytorch_dir", lambda: str(tmp_path))
    monkeypatch.setitem(setup_globals, "_verify_built_native_libs", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "2.10.0+cpu" if name == "torch" else None,
    )
    monkeypatch.setattr(build_ext, "run", lambda self: None)

    command = command_type(Distribution(setup_kwargs))
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "wheel")
    package = tmp_path / "wheel" / "torch_fl"
    package.mkdir(parents=True)
    (package / "_platform_alias.py").write_text('ACCELERATOR = "cuda"\n')
    command.run()

    staged = package / "_build_config.py"
    record = runpy.run_path(str(staged))
    expected = ("flaggems", "vendor") if mudnn_header else ("flaggems",)
    assert record["PLATFORM"] == "musa"
    assert record["ACCELERATOR"] == "mthreads"
    assert record["KERNELS"] == expected
    assert record["NATIVE_HOSTS"] is False
    assert record["NATIVE_VENDOR"] is False
    registrations = _platform_table()["accelerators"]["musa"]["registration"]
    configured = (source_root / "build/register_ops.inc").read_text()
    assert registrations["flaggems"] in configured
    assert (registrations["vendor"] in configured) == mudnn_header
    platform_table = ROOT / "cmake" / "flagos_platforms.json"
    assert (
        staged.with_name("_platforms.json").read_bytes() == platform_table.read_bytes()
    )
    assert not staged.with_name("_platform_alias.py").exists()
    compatibility = json.loads(
        (tmp_path / "wheel" / "torch_fl" / "compatibility.json").read_text()
    )
    assert compatibility["platform"] == "musa"
    # The local segment is the platform's `wheel_local` from
    # cmake/flagos_platforms.json -- the SDK the wheel was built against.
    assert compatibility["wheel_version"] == "2.10.0+musa5.2.0"
    assert compatibility["build"]["kernels"] == list(expected)
    assert compatibility["build"]["vendor_torch_libraries"] is False
    assert compatibility["build"]["distributions"]["torch"] == "2.10.0+cpu"
    assert compatibility["build"]["sdk_version"] is None
    assert compatibility["build"]["vendor_torch_version"] is None


def test_cuda_manifest_records_no_vendor_torch_in_an_ordinary_build(
    monkeypatch, tmp_path, clean_kernel_env
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    namespace, _ = _load_setup(monkeypatch)
    root = tmp_path / "src"
    (root / "torch_fl").mkdir(parents=True)
    globals_ = namespace["_write_compatibility_manifest"].__globals__
    monkeypatch.setitem(globals_, "BASE_DIR", str(root))
    monkeypatch.setitem(globals_, "FLAGOS_ACCELERATOR", "nvidia")
    # The manifest writer imports torch and reads its C++ ABI flag; a stub keeps
    # the test off the real interpreter, whose device-backend autoload is not
    # guaranteed to import cleanly in a bare checkout.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "2.10.0+cuda13.3" if name == "torch" else None,
    )

    namespace["_write_compatibility_manifest"](
        "2.10.0+cuda13.3", _build_config("cuda", "flaggems", "vendor")
    )

    manifest = json.loads((root / "torch_fl" / "compatibility.json").read_text())
    assert manifest["build"]["vendor_torch_libraries"] is False


@pytest.mark.parametrize("bundled", [False, True])
def test_hygon_manifest_records_device_libraries_without_boxing(
    monkeypatch, tmp_path, clean_kernel_env, bundled
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "hygon")
    monkeypatch.setenv("FLAGOS_VENDOR_TORCH_VERSION", "2.10.0+das.opt1.dtk2604")
    namespace, _ = _load_setup(monkeypatch)
    package = tmp_path / "torch_fl"
    (package / "lib_hygon").mkdir(parents=True)
    if bundled:
        (package / "lib_hygon/libtorch_hip.so").touch()
    monkeypatch.setitem(
        namespace["_write_compatibility_manifest"].__globals__,
        "BASE_DIR",
        str(tmp_path),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "2.10.0+cpu" if name == "torch" else None,
    )
    namespace["_write_compatibility_manifest"](
        "2.10.0+dtk2604", _build_config("dcu", "flaggems", "vendor")
    )
    manifest = json.loads((package / "compatibility.json").read_text())
    assert "boxing" not in manifest["build"]["kernels"]
    assert manifest["build"]["vendor_torch_libraries"] is bundled
    assert manifest["build"]["vendor_torch_version"] == "2.10.0+das.opt1.dtk2604"


def test_build_config_is_executable_python(monkeypatch, tmp_path, clean_kernel_env):
    """The generated file is imported by torch_fl._env.build_kernels()."""
    namespace, _ = _load_setup(monkeypatch)
    root = tmp_path / "src"
    (root / "torch_fl").mkdir(parents=True)
    # run_path returns a copy, so the redirect has to go through the function's
    # own globals -- assigning into `namespace` would leave BASE_DIR alone.
    globals_ = namespace["_write_build_config"].__globals__
    monkeypatch.setitem(globals_, "BASE_DIR", str(root))
    monkeypatch.setitem(globals_, "FLAGOS_ACCELERATOR", "nvidia")
    namespace["_write_build_config"](
        _build_config("cuda", "vendor", "flaggems_cpp", "flaggems", "tileops")
    )

    generated = {}
    exec((root / "torch_fl" / "_build_config.py").read_text("utf-8"), generated)
    assert generated["ACCELERATOR"] == "nvidia"
    assert generated["KERNELS"] == (
        "flaggems",
        "flaggems_cpp",
        "tileops",
        "vendor",
    )
    assert generated["NATIVE_HOSTS"] is False


@pytest.mark.parametrize(
    "vendor",
    [
        "nvidia",
        "hygon",
        "dcu",
        "metax",
        "ascend",
        "enflame",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "mthreads",
        "thead",
    ],
)
def test_native_host_build_uses_staged_package(
    monkeypatch, tmp_path, clean_kernel_env, vendor
):
    # Keep the upstream non-CUDA fixtures independent of local vendor SDKs.
    # Hygon and Enflame also check their public names and device aliases.
    monkeypatch.setenv(
        "FLAGOS_ACCELERATOR",
        vendor if vendor in {"hygon", "dcu", "enflame", "gcu"} else "nvidia",
    )
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "1")
    monkeypatch.setenv("FLAGOS_BUILD_FLAGGEMS", "1")
    for name in (
        "FLAGOS_BUILD_BOXING",
        "FLAGOS_BUILD_FLAGGEMS_CPP",
        "FLAGOS_BUILD_TILEOPS",
    ):
        monkeypatch.setenv(name, "0")
    namespace, kwargs = _load_setup(monkeypatch)
    globals_ = namespace["BuildExtWithCmake"]._build_native_hosts.__globals__
    monkeypatch.setitem(globals_, "FLAGOS_ACCELERATOR", vendor)
    native_precision = vendor in {"nvidia", "hygon", "dcu", "metax"}
    monkeypatch.setitem(
        globals_,
        "_native_vendor_cohort",
        lambda: (
            ({}, {"library_dir": "/published-vendor/torch/lib"})
            if native_precision
            else None
        ),
    )
    if vendor in {"metax", "ascend", "iluvatar", "kunlunxin", "mthreads", "thead"}:
        monkeypatch.setitem(
            globals_,
            "_setup_cuda_build_env",
            lambda env: pytest.fail(
                f"{vendor} native hosts must not discover a CUDA toolkit"
            ),
        )
    command = kwargs["cmdclass"]["build_ext"](Distribution(kwargs))
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "staging")
    command.build_temp = str(tmp_path / "objects")
    calls = []
    monkeypatch.setattr(
        subprocess, "check_call", lambda args, **kw: calls.append((args, kw))
    )
    command._build_native_hosts()
    args, options = calls[0]
    assert args[args.index("--package") + 1] == "torch_fl.native_host._generated"
    assert args[args.index("--output") + 1] == command.build_lib
    assert args[args.index("--source-root") + 1] == str(ROOT)
    assert args[args.index("--build-temp") + 1] == str(
        tmp_path / "objects/native_hosts"
    )
    assert options["env"]["_TORCH_FL_BUILDING_NATIVE_HOST"] == "1"
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert options["env"]["PYTHONPATH"].startswith(command.build_lib)
    assert len(calls) == 1 + native_precision
    if native_precision:
        vendor_args, _ = calls[1]
        assert vendor_args[1].endswith("native_host/vendor_build.py")
        assert vendor_args[vendor_args.index("--vendor") + 1] == (
            "hygon" if vendor == "dcu" else vendor
        )


def test_native_metax_rejects_staged_vendor_libtorch(
    monkeypatch, tmp_path, clean_kernel_env
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "1")
    for name in (
        "FLAGOS_BUILD_BOXING",
        "FLAGOS_BUILD_FLAGGEMS_CPP",
        "FLAGOS_BUILD_TILEOPS",
    ):
        monkeypatch.setenv(name, "0")
    namespace, kwargs = _load_setup(monkeypatch)
    globals_ = namespace["BuildExtWithCmake"]._build_native_hosts.__globals__
    monkeypatch.setitem(globals_, "FLAGOS_ACCELERATOR", "metax")
    command = kwargs["cmdclass"]["build_ext"](Distribution(kwargs))
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "staging")
    bundle = Path(command.build_lib) / "torch_fl/lib_metax"
    bundle.mkdir(parents=True)
    (bundle / "libtorch_cuda.so").write_bytes(b"previous vendor build")
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda *args, **kwargs: pytest.fail(
            "a dirty native build must not import Torch"
        ),
    )
    with pytest.raises(RuntimeError, match="clean staging tree"):
        command._build_native_hosts()


@pytest.mark.parametrize("vendor", ["metax", "ascend", "dcu", "iluvatar"])
@pytest.mark.parametrize("different_owner", [False, True])
def test_native_host_ci_build_excludes_the_default_build(
    tmp_path, clean_kernel_env, vendor, different_owner
):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("the CI build script requires Git and Bash")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    helper = workspace / ".github/scripts/lib/set_env_common.sh"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / ".github/scripts/lib/set_env_common.sh").read_bytes())
    recorder = Path(".github/scripts/record_native_host_wheel.py")
    (workspace / recorder).write_bytes((ROOT / recorder).read_bytes())
    install = tmp_path / "installed"
    (workspace / "setup.py").write_text("# tracked source\n")
    for arguments in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(workspace), *arguments],
            check=True,
            capture_output=True,
            timeout=30,
        )
    for relative in (
        "torch_fl/lib_metax/libtorch_cuda.so",
        "build/stale.so",
        "dist/default.whl",
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("previous default build\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    python = tools / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, os, sys, zipfile\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == 'setup.py':\n"
        "    root = Path.cwd()\n"
        "    assert (root / 'setup.py').read_text() == '# tracked source\\n'\n"
        "    assert not (root / 'build/stale.so').exists(), 'inherited build output'\n"
        "    if os.environ['FLAGOS_ACCELERATOR'] == 'iluvatar':\n"
        "        cache = (root / 'build/CMakeCache.txt').read_text().strip()\n"
        "        assert cache == 'CUDA_OCCUPANCY_INCLUDE_DIR:PATH='\n"
        "    else:\n"
        "        assert not (root / 'build').exists(), 'inherited build cache'\n"
        "    assert not (root / 'torch_fl/lib_metax').exists(), 'inherited vendor bundle'\n"
        "    assert os.environ['FLAGOS_BUILD_NATIVE_HOST'] == '1'\n"
        "    if os.environ['FLAGOS_ACCELERATOR'] == 'metax':\n"
        "        assert 'FLAGOS_VENDOR_TORCH_LIB' not in os.environ, 'inherited optional vendor cohort'\n"
        "        assert os.environ['FLAGOS_BUILD_VENDOR'] == '1'\n"
        "    if os.environ['FLAGOS_ACCELERATOR'] == 'dcu':\n"
        "        assert os.environ['FLAGOS_BUILD_VENDOR'] == '1'\n"
        "        assert Path(os.environ['FLAGOS_VENDOR_TORCH_LIB']).is_dir()\n"
        "    Path(os.environ['BUILD_RECEIPT']).write_text(str(root))\n"
        "    output = Path(sys.argv[sys.argv.index('--dist-dir') + 1])\n"
        "    with zipfile.ZipFile(output / 'native.whl', 'w') as wheel:\n"
        "        wheel.writestr('torch_fl/_build_config.py', 'NATIVE_HOSTS = True\\n')\n"
        "        wheel.writestr('torch_fl/compatibility.json', '{}')\n"
        "elif sys.argv[1] == '-m' and sys.argv[-1].endswith('.whl'):\n"
        "    installed = Path(os.environ['PYTHONPATH']) / 'torch_fl-2.10.0.dist-info'\n"
        "    installed.mkdir(parents=True)\n"
        "    (installed / 'METADATA').write_text('Name: torch_fl\\nVersion: 2.10.0\\n')\n"
        "    digest = hashlib.sha256(Path(sys.argv[-1]).read_bytes()).hexdigest()\n"
        "    (installed / 'direct_url.json').write_text(json.dumps(\n"
        "        {'archive_info': {'hashes': {'sha256': digest}}}))\n"
        "elif sys.argv[1] == '-m' or sys.argv[1].endswith('torch-fl-preflight'):\n"
        "    pass\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    receipt = tmp_path / "receipt"
    github_env = tmp_path / "github-env"
    git_config = tmp_path / "gitconfig"
    git_config.write_text("")
    git_environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(git_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1" if different_owner else "0",
    }
    ownership_check = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        env=git_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if different_owner:
        if ownership_check.returncode == 0:
            pytest.skip("Git does not support synthetic ownership checks")
        assert ownership_check.returncode != 0
        assert "dubious ownership" in ownership_check.stderr
    else:
        assert ownership_check.returncode == 0
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/build_native_hosts_ci.sh")],
        env={
            **git_environment,
            "GITHUB_WORKSPACE": str(workspace),
            "GITHUB_ENV": str(github_env),
            "RUNNER_TEMP": str(runner_temp),
            "FLAGOS_ACCELERATOR": vendor,
            "FLAGOS_VENDOR_TORCH_LIB": str(workspace / "torch_fl/lib_metax"),
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "BUILD_RECEIPT": str(receipt),
            "PYTHONPATH": str(install),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert git_config.read_text() == ""
    build_source = Path(receipt.read_text())
    assert build_source.parent == runner_temp
    assert not build_source.exists()
    assert (workspace / "dist/default.whl").read_text() == "previous default build\n"
    assert (workspace / "torch_fl/lib_metax/libtorch_cuda.so").is_file()
    evidence = runner_temp / "torch-fl-native-host-evidence"
    assert github_env.read_text() == f"TORCH_FL_NATIVE_HOST_EVIDENCE={evidence}\n"
    assert (evidence / "native.whl").is_file()
    wheel_record = json.loads((evidence / "wheel.json").read_text())
    assert wheel_record["wheel"] == "native.whl"
    assert wheel_record["package"] == str((install / "torch_fl").resolve())
    assert (
        evidence / "torch_fl/_build_config.py"
    ).read_text() == "NATIVE_HOSTS = True\n"


def test_native_host_build_rejects_boxing(monkeypatch, clean_kernel_env):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "1")
    monkeypatch.setenv("FLAGOS_BUILD_BOXING", "1")
    _, kwargs = _load_setup(monkeypatch)
    command = kwargs["cmdclass"]["build_ext"](Distribution(kwargs))
    command.ensure_finalized()
    with pytest.raises(ValueError, match="FLAGOS_BUILD_BOXING has been removed"):
        command._build_native_hosts()


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "vendor,route",
    [
        ("ascend", "mean.dim = ascend\n"),
        ("cambricon", "add.Tensor = flaggems\n"),
        ("cambricon", "add.Scalar = flaggems; Float,Half,BFloat16,Int:cambricon\n"),
        ("metax", "bmm = flaggems; BFloat16:metax\n"),
        ("metax", "_log_softmax = flaggems\n"),
        ("thead", "bmm = flaggems; BFloat16:thead\n"),
        ("thead", "mean.dim = flaggems\n"),
        ("thead", "sum.dim_IntList = flaggems\n"),
        ("thead", "silu_backward = flaggems\n"),
    ],
)
def test_native_host_stages_one_config_and_restores_the_default(
    monkeypatch, tmp_path, clean_kernel_env, inplace, vendor, route
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "1")
    monkeypatch.setenv("FLAGOS_BUILD_VENDOR", "1")
    namespace, kwargs = _load_setup(monkeypatch)
    write_record = namespace["_write_build_config"]
    platform = namespace["_normalize_accelerator"](vendor)
    monkeypatch.setitem(write_record.__globals__, "FLAGOS_ACCELERATOR", platform)
    monkeypatch.setitem(write_record.__globals__, "BASE_DIR", str(tmp_path))
    monkeypatch.setitem(write_record.__globals__, "_native_vendor_cohort", lambda: None)
    command = kwargs["cmdclass"]["build_ext"](Distribution(kwargs))
    command.ensure_finalized()
    command.inplace = inplace

    package = tmp_path / "torch_fl"
    default = package / f"configs/backends_{vendor}.conf"
    source = ROOT / "torch_fl" / f"configs/backends_{vendor}.conf"
    ordinary = source.read_bytes()
    default.parent.mkdir(parents=True)
    default.write_bytes(ordinary)
    legacy = default.with_name("backends_metax_native_host.conf")
    if not inplace and vendor == "metax":
        legacy.touch()

    write_record(
        _build_config(platform, "flaggems", "vendor", native_hosts=True),
        inplace=inplace,
    )
    command._stage_native_host_config(tmp_path, vendor, native=True)
    record = runpy.run_path(str(package / "_build_config.py"))
    directory = "lib" if inplace else "configs"
    expected = f"{directory}/backends_{vendor}.conf"
    assert record["BACKEND_CONFIG"] == expected
    staged = package / expected
    assert route in staged.read_text()
    assert source.read_bytes() == ordinary
    assert not legacy.exists()

    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "0")
    write_record(_build_config(platform, "flaggems"), inplace=inplace)
    command._stage_native_host_config(tmp_path, vendor, native=False)
    assert "BACKEND_CONFIG" not in runpy.run_path(str(package / "_build_config.py"))
    assert default.read_bytes() == ordinary
    if inplace:
        assert not staged.exists()


def test_ordinary_wheel_clears_stale_native_host_stage(
    monkeypatch, tmp_path, clean_kernel_env
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "apple")
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "0")
    _, kwargs = _load_setup(monkeypatch)
    command = kwargs["cmdclass"]["build_ext"](Distribution(kwargs))
    command.ensure_finalized()
    command.build_lib = str(tmp_path)
    command.inplace = False
    native = tmp_path / "torch_fl/native_host"
    (native / "csrc").mkdir(parents=True)
    (native / "csrc/launcher.cc").touch()
    (native / "_generated").mkdir()
    (native / "_generated/manifest.json").touch()
    (native / "load.py").touch()

    command._build_native_hosts()

    assert sorted(path.name for path in native.iterdir()) == ["load.py"]


@pytest.mark.parametrize(
    "vendor, requirement, supported",
    [
        ("cambricon", "triton==3.4.0+mlu2.1.1", True),
        ("tsingmicro", "flagtree==0.7.0+tsingmicro3.6", True),
        ("drobotics", None, False),
    ],
)
def test_native_host_build_selects_the_platform_launch_adapter(
    monkeypatch, tmp_path, clean_kernel_env, vendor, requirement, supported
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "1")
    monkeypatch.setenv("FLAGOS_BUILD_FLAGGEMS", "1")
    for name in (
        "FLAGOS_BUILD_BOXING",
        "FLAGOS_BUILD_FLAGGEMS_CPP",
        "FLAGOS_BUILD_TILEOPS",
    ):
        monkeypatch.setenv(name, "0")
    namespace, kwargs = _load_setup(monkeypatch)
    monkeypatch.setitem(
        namespace["BuildExtWithCmake"]._build_native_hosts.__globals__,
        "FLAGOS_ACCELERATOR",
        vendor,
    )
    calls = []
    monkeypatch.setattr(
        subprocess, "check_call", lambda command, **_: calls.append(command)
    )
    command = kwargs["cmdclass"]["build_ext"](Distribution(kwargs))
    command.ensure_finalized()
    command.build_lib = str(tmp_path)
    if supported:
        assert requirement in namespace["_flagos_sibling_requires"]()
        command._build_native_hosts()
        assert len(calls) == 1
        assert calls[0][1].endswith("torch_fl/native_host/build.py")
    else:
        with pytest.raises(
            RuntimeError, match="no validated native-host launch adapter"
        ):
            command._build_native_hosts()
        assert not calls


def test_native_vendor_records_the_verified_cohort(
    monkeypatch, tmp_path, clean_kernel_env
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    namespace, _ = _load_setup(monkeypatch)
    globals_ = namespace["_write_compatibility_manifest"].__globals__
    record = {
        "version": "2.10.0+cu130",
        "flag_gems": "5.4.0",
        "requirements": ["nvidia-cublas==13.1.0.3"],
    }
    cohort = SimpleNamespace(bundle_directory=lambda vendor: Path("lib"))
    monkeypatch.setitem(globals_, "_native_vendor_cohort", lambda: (cohort, record))
    monkeypatch.setitem(globals_, "BASE_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "2.10.0+cpu" if name == "torch" else None,
    )
    (tmp_path / "torch_fl").mkdir()
    build_config = _build_config("cuda", "flaggems", "vendor", native_hosts=True)
    namespace["_write_build_config"](build_config)
    namespace["_write_compatibility_manifest"]("2.10.0+cuda13.3", build_config)
    manifest = json.loads((tmp_path / "torch_fl/compatibility.json").read_text())
    assert (
        "NATIVE_VENDOR = True" in (tmp_path / "torch_fl/_build_config.py").read_text()
    )
    assert manifest["build"]["vendor_torch_libraries"] is True
    assert manifest["build"]["vendor_torch_version"] == "2.10.0+cu130"
    assert manifest["build"]["bundle_libdir"] == "lib"
    requirements = namespace["_install_requires"]()
    assert "torch==2.10.0+cpu" in requirements
    assert "flag_gems==5.4.0" in requirements
    assert requirements.count("flag_gems==5.4.0") == 1
    assert "nvidia-cublas==13.1.0.3" in requirements
    assert not any("cu12" in value for value in requirements)
    record["flag_gems"] = "5.4.1"
    with pytest.raises(RuntimeError, match="revalidate the native vendor policy"):
        namespace["_install_requires"]()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("FLAGOS_BUILD_VENDOR", "0"),
        ("FLAGOS_SKIP_CUDA_ASSETS", "1"),
    ],
)
def test_native_vendor_refuses_incompatible_build(
    monkeypatch, tmp_path, clean_kernel_env, flag, value
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    namespace, _ = _load_setup(monkeypatch)
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", "1")
    monkeypatch.setenv("FLAGOS_VENDOR_TORCH_LIB", str(tmp_path / "torch/lib"))
    monkeypatch.setenv(flag, value)
    with pytest.raises(RuntimeError, match="bundled vendor libraries"):
        namespace["_native_vendor_cohort"]()


def _switches(namespace, accelerator):
    """Resolve the kernel set for a vendor name (the table keys the platform id)."""
    return namespace["_kernel_switches"](
        namespace["_VENDOR_TO_PLATFORM"].get(accelerator, accelerator)
    )


@pytest.mark.parametrize(
    "accelerator, expected",
    [
        # No platform compiles the CUDA boxing kernels by default now: boxing is
        # the explicit opt-in that these defaults leave off, so a default wheel
        # depends on no vendor torch package. CUDA still compiles everything
        # else; the platforms below drop TileOps and/or the FlagGems C++ wrapper
        # as before.
        ("nvidia", {"vendor", "flaggems", "flaggems_cpp", "tileops"}),
        ("metax", {"flaggems"}),
        ("ascend", {"vendor", "flaggems"}),
        ("hygon", {"vendor", "flaggems"}),
        ("thead", {"vendor", "flaggems"}),
        # tsingmicro has no FlagGems C++ library to link: flag_gems ships
        # py3-none-any, and CMakeLists.txt treats a missing FlagGems_DIR as a
        # FATAL_ERROR, so leaving FLAGOS_BUILD_FLAGGEMS_CPP on would make a
        # default `pip install` fail at configure. Nothing routes to it here.
        ("tsingmicro", {"vendor", "flaggems"}),
        ("enflame", {"vendor", "flaggems"}),
        ("mthreads", {"vendor", "flaggems"}),
        # Iluvatar is FlagGems-primary but no longer vendor-less: the native,
        # non-boxing kernels under aten/backends/iluvatar/ wrap corex's
        # CUDA-named C ABI (ixdnn/ixblas) for the ops FlagGems does not cover.
        ("iluvatar", {"vendor", "flaggems"}),
        # BPU keeps the FLAGOS_BUILD_VENDOR default, but csrc/aten/backends/bpu/ does
        # not exist, so "vendor" here records a switch that is on rather than a
        # kernel set that is present.
        ("bpu", {"vendor"}),
        # MPS is native (no CUDA runtime to box onto) and has no FlagGems/Triton
        # line, so only FLAGOS_BUILD_VENDOR is on; as for BPU its operator tree
        # does not exist yet, so "vendor" is a switch rather than a present set.
        ("apple", {"vendor"}),
    ],
)
def test_default_kernel_set_per_accelerator(
    monkeypatch, accelerator, expected, clean_kernel_env
):
    namespace, _ = _load_setup(monkeypatch)
    resolved = _switches(namespace, accelerator)
    names = _platform_table()["kernel_set_names"]
    assert {names[switch] for switch, on in resolved.items() if on} == expected


def test_build_record_carries_both_platform_names(
    monkeypatch, tmp_path, clean_kernel_env
):
    namespace, _ = _load_setup(monkeypatch)
    emit = namespace["_write_build_config"]
    normalize = namespace["_normalize_accelerator"]
    package = tmp_path / "torch_fl"
    package.mkdir()
    monkeypatch.setitem(emit.__globals__, "BASE_DIR", str(tmp_path))

    table = json.loads(
        (ROOT / "cmake" / "flagos_platforms.json").read_text(encoding="utf-8")
    )
    for expected, row in table["accelerators"].items():
        vendor = row["vendor"]
        for name in (expected, vendor):
            platform = normalize(name)
            assert platform == expected
            monkeypatch.setitem(emit.__globals__, "FLAGOS_ACCELERATOR", platform)
            emit(_build_config(platform))
            record = runpy.run_path(str(package / "_build_config.py"))
            assert record["PLATFORM"] == expected
            assert record["ACCELERATOR"] == vendor
    assert normalize("D-Robotics") == "bpu"


def test_explicit_kernel_switch_overrides_the_vendor_default(
    monkeypatch, clean_kernel_env
):
    """MetaX enables FlagGems by default and honors an explicit opt-out."""
    namespace, _ = _load_setup(monkeypatch)
    monkeypatch.setenv("FLAGOS_BUILD_FLAGGEMS", "0")
    assert namespace["_kernel_switches"]("metax")["FLAGOS_BUILD_FLAGGEMS"] is False
    # And on CUDA the same variable turns it off too.
    assert _switches(namespace, "nvidia")["FLAGOS_BUILD_FLAGGEMS"] is False


def test_kernel_switch_uses_the_shared_truth_table(monkeypatch, clean_kernel_env):
    """The build-side parser must not accept what the runtime side rejects."""
    namespace, _ = _load_setup(monkeypatch)
    for raw, expected in (
        ("1", True),
        ("on", True),
        ("YES", True),
        ("  true  ", True),
        ("0", False),
        ("off", False),
        ("No", False),
        ("2", True),  # not a boolean: falls back to the vendor default, not "on"
    ):
        monkeypatch.setenv("FLAGOS_BUILD_VENDOR", raw)
        assert _switches(namespace, "nvidia")["FLAGOS_BUILD_VENDOR"] is expected, raw
    # On a set whose default is off the same input stays off, which is what
    # distinguishes "falls back to the default" from the hand-rolled parser's
    # "anything that is not 0/false/off/no is on". MetaX now admits an explicit
    # TileOPs experiment, while ppu remains outside it.
    monkeypatch.setenv("FLAGOS_BUILD_TILEOPS", "2")
    assert namespace["_kernel_switches"]("ppu")["FLAGOS_BUILD_TILEOPS"] is False


def test_metax_can_opt_into_the_tileops_experiment(monkeypatch, clean_kernel_env):
    """MetaX remains off by default but accepts an explicit experiment build."""
    namespace, _ = _load_setup(monkeypatch)
    monkeypatch.setenv("FLAGOS_BUILD_TILEOPS", "1")
    resolved = namespace["_kernel_switches"]("metax")
    assert resolved["FLAGOS_BUILD_TILEOPS"] is True
    assert "FLAGOS_BUILD_BOXING" not in resolved


def test_impossible_kernel_switch_raises_instead_of_being_ignored(
    monkeypatch, clean_kernel_env
):
    """MUSA requests vendor kernels before checking SDK availability."""
    namespace, _ = _load_setup(monkeypatch)
    monkeypatch.setenv("FLAGOS_BUILD_VENDOR", "0")
    with pytest.raises(ValueError, match="FLAGOS_BUILD_VENDOR"):
        _switches(namespace, "mthreads")
    # The pin agrees with the default, so this is accepted.
    monkeypatch.setenv("FLAGOS_BUILD_VENDOR", "1")
    assert _switches(namespace, "mthreads")["FLAGOS_BUILD_VENDOR"] is True
    # And a platform with no pin has no such objection.
    monkeypatch.setenv("FLAGOS_BUILD_FLAGGEMS_CPP", "1")
    assert _switches(namespace, "thead")["FLAGOS_BUILD_FLAGGEMS_CPP"] is True


def _platform_table() -> dict:
    return json.loads((ROOT / "cmake" / "flagos_platforms.json").read_text())


def _build_config(platform, *kernels, native_hosts=False):
    row = _platform_table()["accelerators"][platform]
    return {
        "platform": platform,
        "vendor": row["vendor"],
        "bundle_libdir": row["bundle_libdir"],
        "kernels": list(kernels),
        "native_hosts": native_hosts,
    }


def test_platform_table_is_internally_consistent():
    """Every accelerator row is complete, and each pin agrees with its default.

    cmake/flagos_platforms.json is the single source setup.py and CMake read; a
    missing default, or a pin that disagrees with the default, would resolve
    differently on the two sides -- the drift the table exists to prevent.
    """
    table = _platform_table()
    switches = table["kernel_switches"]
    assert len(switches) == 4 and len(set(switches)) == 4
    assert set(table["kernel_set_names"]) == set(switches)

    required = (
        "project_languages",
        "runtime_dir",
        "bundle_libdir",
        "vendor_torch_libraries",
        "wheel_local",
        "use_macro",
        "writes_platform_marker",
        "c10_cuda_no_cmake_configure",
        "kernel_defaults",
        "pins",
    )
    for accelerator, row in table["accelerators"].items():
        for field in required:
            assert field in row, (accelerator, field)
        assert isinstance(row["vendor_torch_libraries"], bool), accelerator
        # vendor_torch_gate, when present, names the one kernel switch the build
        # must compile for the wheel to source a vendor libtorch. It only makes
        # sense where the platform declares one at all.
        gate = row.get("vendor_torch_gate")
        if gate is not None:
            assert row["vendor_torch_libraries"], accelerator
            assert gate in switches, (accelerator, gate)
        assert set(row["kernel_defaults"]) == set(switches), accelerator
        assert all(
            isinstance(value, bool) for value in row["kernel_defaults"].values()
        ), accelerator
        for name, pin in row["pins"].items():
            assert name in switches, (accelerator, name)
            assert pin["value"] is row["kernel_defaults"][name], (accelerator, name)
            assert pin["reason"], (accelerator, name)
        # CUDA-toolchain platforms enable CUDA; every other platform builds host
        # C++ only. Objective-C++ joins the list for platforms whose runtime is a
        # native Objective-C API (Metal on Apple Silicon).
        languages = row["project_languages"]
        assert "CUDA" in languages or languages in (
            ["CXX", "C"],
            ["OBJCXX", "CXX", "C"],
        ), accelerator


def test_cmake_declares_every_kernel_switch():
    """The root declares the switches and csrc uses the resolved values."""
    switches = set(_platform_table()["kernel_switches"])
    pattern = r"^option\((FLAGOS_BUILD_\w+)"
    root_options = re.findall(
        pattern, (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"), re.M
    )
    assert set(root_options) == switches
    assert not re.search(
        pattern, (ROOT / "csrc/CMakeLists.txt").read_text(encoding="utf-8"), re.M
    )


def test_setup_resolves_the_table_defaults(monkeypatch, clean_kernel_env):
    """setup.py's resolved defaults are the table's, for every accelerator.

    Ties the Python consumer to the shared source directly, rather than through
    the hand-written expected sets above.
    """
    namespace, _ = _load_setup(monkeypatch)
    for accelerator, row in _platform_table()["accelerators"].items():
        assert namespace["_kernel_switches"](accelerator) == row["kernel_defaults"], (
            accelerator
        )


def test_a_build_without_a_marker_drops_an_inherited_one(monkeypatch, tmp_path):
    """torch_fl/lib/ is never cleaned, so another build's marker must not survive.

    A MetaX build whose tree still carried an "mps" marker (left by the marker
    tests' copies, or by an earlier apple build) read the empty
    backends_apple.conf and raised "routed to 'none'" on every op it does not
    list. setup.py removes it for the platforms whose row writes no marker, and
    keeps it where the row does.
    """
    # Not metax: importing setup.py with it compiles and loads the cu-bridge
    # shim, which only a MetaX box can do. The removal is keyed on the row's
    # writes_platform_marker flag, which metax and ppu both set to false.
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "ppu")
    namespace, _ = _load_setup(monkeypatch)
    root = tmp_path / "src"
    marker = root / "torch_fl" / "lib" / "flagos_platform"
    marker.parent.mkdir(parents=True)
    marker.write_text("mps\n")
    monkeypatch.setitem(
        namespace["_drop_inherited_platform_marker"].__globals__,
        "BASE_DIR",
        str(root),
    )
    namespace["_drop_inherited_platform_marker"]()
    assert not marker.exists()


def test_a_marker_writing_build_keeps_its_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "ascend")
    namespace, _ = _load_setup(monkeypatch)
    root = tmp_path / "src"
    marker = root / "torch_fl" / "lib" / "flagos_platform"
    marker.parent.mkdir(parents=True)
    marker.write_text("ascend\n")
    monkeypatch.setitem(
        namespace["_drop_inherited_platform_marker"].__globals__,
        "BASE_DIR",
        str(root),
    )
    namespace["_drop_inherited_platform_marker"]()
    assert marker.read_text() == "ascend\n"


def _make_minimal_dtk(root):
    """Just enough of a DTK tree for the DCU _flagos_nccl link set to resolve."""
    for relative in (
        "cuda/cuda-12/include/cuda.h",
        "include/rccl/nccl.h",
        "lib/librccl.so",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _make_minimal_vendor_torch_lib(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("libc10_hip.so", "libtorch_hip.so"):
        (root / name).touch()


def _dcu_bridge_env(monkeypatch, tmp_path):
    """Point a DCU build at throwaway DTK and vendor torch/lib trees."""
    dtk = tmp_path / "dtk"
    vendor_lib = tmp_path / "vendor-torch" / "lib"
    _make_minimal_dtk(dtk)
    _make_minimal_vendor_torch_lib(vendor_lib)
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "hygon")
    monkeypatch.setenv("ROCM_PATH", str(dtk))
    monkeypatch.setenv("FLAGOS_VENDOR_TORCH_LIB", str(vendor_lib))


def test_build_ext_appends_the_dcu_comm_bridge_target(
    monkeypatch, tmp_path, clean_kernel_env
):
    """A DCU build must describe the bridge, so setuptools builds and stages it.

    It has to be in ext_modules from _get_setup_kwargs() rather than appended by
    the build_ext command: finalize_options fills ext_map/_needs_stub and
    get_outputs from that list before run(), and the inplace copy back into
    torch_fl/comm/_nccl_ext/ is driven by it (issue #366).
    """
    _dcu_bridge_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["setup.py", "build_ext"])

    _, setup_kwargs = _load_setup(monkeypatch)

    assert [extension.name for extension in setup_kwargs["ext_modules"]] == [
        "torch_fl._C",
        "torch_fl.comm._nccl_ext._flagos_nccl",
    ]


def test_metadata_only_pass_does_not_resolve_the_dcu_link_set(
    monkeypatch, tmp_path, clean_kernel_env
):
    """egg_info/sdist must not need DTK to be installed.

    FLAGOS_ACCELERATOR=dcu with no DTK anywhere is exactly the metadata case: it
    resolves, but only because the gate keeps the bridge out of the target list.
    """
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "hygon")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "missing-dtk"))
    monkeypatch.setattr(sys, "argv", ["setup.py", "egg_info"])

    _, setup_kwargs = _load_setup(monkeypatch)

    assert [extension.name for extension in setup_kwargs["ext_modules"]] == [
        "torch_fl._C"
    ]


def test_build_ext_adds_torch_paths_to_the_comm_bridge_only(
    monkeypatch, clean_kernel_env
):
    """Only the bridge target gains torch's include/library paths and link set.

    torch_fl._C is a C stub over libtorch_bindings.so and must not inherit the
    c10/torch/torch_cpu link set the pybind factory needs.
    """
    # Stubbed rather than imported: torch.utils.cpp_extension reports the paths
    # of the *installed* torch, and importing the real one from this checkout
    # pulls in torch_fl's device-backend preload.
    cpp_extension = SimpleNamespace(
        include_paths=lambda: ["/torch/include"],
        library_paths=lambda: ["/torch/lib"],
    )
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)

    namespace, _ = _load_setup(monkeypatch)
    bridge_name = namespace["_load_nccl_ext_builder"]().WHEEL_EXTENSION_NAME
    bridge = Extension(name=bridge_name, sources=["nccl_backend.cpp"])
    other = Extension(name="torch_fl._C", sources=["csrc/bindings/stub.c"])
    command = SimpleNamespace(extensions=[other, bridge])

    namespace["BuildExtWithCmake"]._prepare_nccl_extension(command)

    assert bridge.include_dirs == ["/torch/include"]
    assert bridge.library_dirs == ["/torch/lib"]
    assert bridge.libraries == ["c10", "torch", "torch_cpu"]
    assert other.include_dirs == [] and other.libraries == []


def test_a_pre_release_suffix_lands_between_the_version_and_the_sdk(
    monkeypatch, clean_kernel_env
):
    """`v2.10.0rc1` has to be buildable, and PEP 440 puts the suffix first.

    The release workflow sets this from the tag; the version then reads
    `2.10.0rc1+cuda13.3`, which sorts below `2.10.0` so a plain
    `pip install torch_fl==2.10.0` does not pick the candidate up.
    """
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "hygon")
    monkeypatch.setenv("FLAGOS_WHEEL_PRERELEASE", "rc1")

    _, setup_kwargs = _load_setup(monkeypatch)

    assert setup_kwargs["version"] == "2.10.0rc1+dtk2604"


def test_no_pre_release_suffix_leaves_the_release_version_alone(
    monkeypatch, clean_kernel_env
):
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "hygon")
    monkeypatch.delenv("FLAGOS_WHEEL_PRERELEASE", raising=False)

    _, setup_kwargs = _load_setup(monkeypatch)

    assert setup_kwargs["version"] == "2.10.0+dtk2604"


@pytest.mark.parametrize("value", ["final", "rc", "1", "rc1.post1", ""])
def test_an_unusable_pre_release_value_is_refused(monkeypatch, clean_kernel_env, value):
    """Guessing here would publish under a version nobody asked for."""
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "nvidia")
    monkeypatch.setenv("FLAGOS_WHEEL_PRERELEASE", value)

    if value == "":
        _, setup_kwargs = _load_setup(monkeypatch)
        assert setup_kwargs["version"] == "2.10.0+cuda13.3"
        return
    with pytest.raises(ValueError, match="PEP 440 pre-release"):
        _load_setup(monkeypatch)
