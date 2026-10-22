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

"""Assert the wheel landed in the environment its platform config describes.

This replaced the hand-written per-platform heredocs, each re-stating the
same handful of facts about the environment and drifting from the others in
which of them it stated. The facts are the platform's -- a stock +cpu torch,
the vendor bundle in place of the CUDA one, so many devices -- so they live in
that platform's `environment_check` block and this script reads them from the
`FLAGOS_ENVIRONMENT_CHECK` variable the manifest step is handed.

One step for every platform, and every assertion is stated once. What the
platform config supplies is a value; there is no per-platform code path here
to hide a check that only one image happens to run.

The assertions are deliberately about the *shape* of the environment and not
about what the tests then find: this is the check that names a broken image
once, instead of ninety errors into the FlagGems group. Keep the messages
specific enough to say which expectation failed.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace


ENVIRONMENT_VARIABLE = "FLAGOS_ENVIRONMENT_CHECK"
FLAGGEMS_KERNEL_SET = "flaggems"

QWEN3_REPOSITORY = "Qwen/Qwen3-0.6B"
QWEN3_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN3_FILE_SIZES = {
    "config.json": 726,
    "generation_config.json": 239,
    "merges.txt": 1671853,
    "model.safetensors": 1503300328,
    "tokenizer.json": 11422654,
    "tokenizer_config.json": 9732,
    "vocab.json": 2776833,
}
QWEN3_SHA256 = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}

_BOUNDS = ("at_least", "more_than", "exactly")
_TLE_IMPORT_FAILURES = set()

# Mach-O cputype -> the name distutils tags a wheel with. Only the two this
# project can produce are named; anything else is reported by number, because a
# wrong name would read as a matched architecture.
_CPU_TYPES = {7: "x86_64", 16777223: "x86_64", 12: "arm64", 16777228: "arm64"}

# A fat file states its own entry count, which is read from the file and then
# used to size the read that follows. Bounded so a corrupt or hostile header
# cannot ask for an unbounded one.
FAT_ARCH_LIMIT = 64

# Every key a platform may state. The list is closed on purpose: a misspelled
# key reads as an expectation that was never checked, and the platform would
# pass this step while asserting nothing about the fact it meant to assert.
# tests/unit/test_integration_manifest.py holds all configured platforms to this.
SUPPORTED_KEYS = frozenset(
    {
        "torch_version_prefix",
        "torch_version_base",
        "cuda_version",
        "torch_path_prefix",
        "torch_path_forbidden",
        "require_package_files",
        "forbid_package_files",
        "require_build_kernels",
        "forbid_build_kernels",
        "flagos_devices",
        "cuda_available",
        "cuda_devices",
        "triton_backend",
        "flaggems_vendor",
        "require_flaggems_tle_modules",
        "require_imports",
        "native_extensions_arch",
        "torch_musa_shim_only",
        "flagtree",
    }
)


def load_spec():
    """The platform's expectations, as the manifest step passed them.

    A missing or unreadable block is fatal rather than an empty check: a
    default that asserts nothing would turn every platform whose data never
    arrived into a green run.
    """
    raw = os.environ.get(ENVIRONMENT_VARIABLE)
    if not raw:
        raise SystemExit(
            f"{ENVIRONMENT_VARIABLE} is not set; this step must be run from the "
            "integration manifest, which passes the platform's environment_check "
            "block through it"
        )
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"{ENVIRONMENT_VARIABLE} is not valid JSON: {error}"
        ) from error
    if not isinstance(spec, dict):
        raise SystemExit(f"{ENVIRONMENT_VARIABLE} must be a JSON object")
    unknown = sorted(set(spec) - SUPPORTED_KEYS)
    if unknown:
        raise SystemExit(
            f"{ENVIRONMENT_VARIABLE} states {', '.join(unknown)}, which this "
            f"script does not read; supported keys are "
            f"{', '.join(sorted(SUPPORTED_KEYS))}"
        )
    return spec


def import_dependencies(spec):
    """Import the stack this platform's expectations are stated against.

    `torch_fl` precedes `flag_gems` for every platform, and that order is the
    one thing this step cannot get wrong: FlagGems 5.x resolves its vendor from
    the torch_fl device surface, so importing it against stock CPU PyTorch
    raises "No device were detected on your machine !" and this check would die
    with a message about the environment it was written to certify.

    Triton and FlagGems are imported only when the spec has something to say
    about them. A platform whose image carries neither still gets its wheel and
    device assertions, rather than an ImportError standing in for them.
    """
    import torch_fl
    import torch

    # Anything else the platform needs present before its tests can run. The
    # import is the whole check: a platform that lists `flag_gems` here has
    # said the package has to resolve, and an ImportError naming it is a better
    # failure than the first test that reaches for it.
    for module_name in spec.get("require_imports") or []:
        importlib.import_module(module_name)

    dependencies = SimpleNamespace(
        torch=torch,
        torch_fl=torch_fl,
        triton=None,
        flag_gems=None,
        flagtree_is_active=None,
    )
    if (
        spec.get("triton_backend")
        or spec.get("flagtree")
        or FLAGGEMS_KERNEL_SET in (spec.get("require_build_kernels") or [])
    ):
        import triton

        dependencies.triton = triton
    if (
        spec.get("flaggems_vendor")
        or spec.get("flagtree")
        or FLAGGEMS_KERNEL_SET in (spec.get("require_build_kernels") or [])
    ):
        import flag_gems

        dependencies.flag_gems = flag_gems
    if spec.get("flagtree"):
        from torch_fl.compile.flagtree_shim import is_flagtree_active

        dependencies.flagtree_is_active = is_flagtree_active
    return dependencies


def assert_bound(bound, value, what):
    """Check one device count against the bound the config states."""
    if bound is None:
        return
    if not isinstance(bound, dict):
        raise SystemExit(f"{what} expectations must be a mapping, found {bound!r}")
    given = [key for key in _BOUNDS if key in bound]
    if len(given) != 1:
        raise SystemExit(
            f"{what} expectations must state exactly one of "
            f"{', '.join(_BOUNDS)}, found {given or 'none'}"
        )
    key = given[0]
    expected = bound[key]
    if key == "at_least":
        ok = value >= expected
    elif key == "more_than":
        ok = value > expected
    else:
        ok = value == expected
    assert ok, f"expected {what} {key} {expected}, found {value}"


def assert_torch_version(dependencies, spec):
    prefix = spec.get("torch_version_prefix")
    if prefix:
        assert dependencies.torch.__version__.startswith(prefix), (
            dependencies.torch.__version__
        )
    base = spec.get("torch_version_base")
    if base:
        assert dependencies.torch.__version__.split("+", 1)[0] == base, (
            dependencies.torch.__version__
        )


def assert_torch_path(spec, torch_path):
    expected_prefix = spec.get("torch_path_prefix")
    if expected_prefix:
        assert str(torch_path).startswith(expected_prefix), torch_path
    for forbidden in spec.get("torch_path_forbidden") or []:
        assert forbidden not in str(torch_path), f"{forbidden} in {torch_path}"


def assert_build_kernels(spec, torch_fl, dependencies):
    """The kernel sets the wheel was built with, from the build record.

    `require_package_files` says what the wheel ships, which is one step removed
    from what it dispatches to: a platform whose accelerator kernels are opt-in
    builds the same wheel name either way. This reads `torch_fl/_build_config.py`
    through `torch_fl._env.build_kernels()`, so a lane that means to test the
    default wheel can say so as an expectation and fail naming the kernel set it
    found rather than as an op that reached no kernel.

    A checkout that was never built records no kernel set at all, so a platform
    requiring one fails here instead of passing on an empty tuple.
    """
    required = spec.get("require_build_kernels") or []
    forbidden = spec.get("forbid_build_kernels") or []
    if not required and not forbidden:
        return None
    kernels = torch_fl._env.build_kernels()
    for name in required:
        assert name in kernels, (
            f"expected the {name!r} kernel set, found {sorted(kernels)}"
        )
    if FLAGGEMS_KERNEL_SET in required:
        if dependencies.triton is None:
            import triton

            dependencies.triton = triton
        if dependencies.flag_gems is None:
            import flag_gems

            dependencies.flag_gems = flag_gems
        available = sorted(dependencies.triton.backends.backends)
        assert available, "the installed Triton has no backend for FlagGems"
        print(
            f"FlagGems runtime: {dependencies.flag_gems.__file__}; "
            f"Triton backends: {available}"
        )
    for name in forbidden:
        assert name not in kernels, (
            f"the {name!r} kernel set is built into this wheel: {sorted(kernels)}"
        )
    return kernels


def assert_package_files(spec, package_root):
    """The vendor libraries the wheel is expected to ship, and not to ship."""
    for relative in spec.get("require_package_files") or []:
        assert (package_root / relative).is_file(), package_root / relative
    for relative in spec.get("forbid_package_files") or []:
        assert not (package_root / relative).is_file(), package_root / relative


def assert_cuda_version(dependencies, spec):
    if "cuda_version" not in spec:
        return
    expected = spec["cuda_version"]
    if expected is None:
        assert dependencies.torch.version.cuda is None, dependencies.torch.version.cuda
    else:
        assert dependencies.torch.version.cuda == expected, (
            dependencies.torch.version.cuda
        )


def assert_torch_musa_shim():
    """MUSA: the vendor package must not have leaked into the isolated venv.

    torch_fl installs a shim named ``torch_musa`` with ``origin ==
    "torch_fl_shim"``; the real vendor package in the same interpreter means
    an ABI mismatch, which is worth failing on here rather than as a device
    error later.
    """
    print("=== sys.path ===")
    for entry in sys.path:
        print(f"  {entry}")
    spec = importlib.util.find_spec("torch_musa")
    if spec is None:
        return
    print("=== torch_musa found ===")
    print(f"  Location: {spec.origin}")
    print(f"  Submodule search locations: {spec.submodule_search_locations}")
    if spec.origin != "torch_fl_shim":
        raise AssertionError(
            f"vendor torch_musa leaked into the isolated venv (ABI mismatch): {spec.origin}"
        )
    print("  OK: this is torch_fl's shim, not the vendor package")


def assert_triton_backend(dependencies, backend):
    """The Triton in the image has to be the one carrying this backend."""
    available = sorted(dependencies.triton.backends.backends)
    assert backend in available, available


def assert_flaggems_vendor(dependencies, vendor):
    assert dependencies.flag_gems.vendor_name == vendor, (
        dependencies.flag_gems.vendor_name
    )


def assert_flagtree(dependencies, flagtree):
    """The FlagTree build is the one the wheel's Triton routes need.

    The FlagTree wheel is source-free: it ships its own Triton, and the
    expected version comes from the setup script so the two stay in step.
    """
    expected = os.environ.get(flagtree["version_env"], flagtree["version_default"])
    distribution = importlib.metadata.distribution("flagtree")
    assert distribution.version == expected, distribution.version
    assert dependencies.flagtree_is_active(), dependencies.triton.__file__

    triton_root = Path(dependencies.triton.__file__).resolve().parent
    assert Path(distribution.locate_file("triton")).resolve() == triton_root, (
        triton_root
    )
    assert dependencies.triton.__version__.startswith(flagtree["triton_prefix"]), (
        dependencies.triton.__version__
    )
    return expected


def _flagtree_version_from_pin_table(platform):
    """Read the selected FlagTree build from the shared setup environment."""

    pins = Path(__file__).resolve().parents[1] / "version-pins.env"
    for line in pins.read_text(encoding="utf-8").splitlines():
        if line.startswith("FLAGTREE_VERSION_"):
            name, separator, value = line.partition("=")
            if separator and name.removeprefix("FLAGTREE_VERSION_") == platform:
                return value
    return None


def assert_flaggems_tle(module_name, dependencies):
    """Require the TLE package FlagGems imports rather than only its namespace.

    A FlagTree release can carry ``triton.experimental`` without a ``tle``
    submodule. A release can also ship a partial TLE package, so the configured
    modules are imported rather than inferred from the parent namespace.

    TLE's Ascend communicator needs AscendSHMEM, absent from FlagTree images.
    The scripts that prepare a build run before ``torch_fl._C`` exists. Loading
    FlagGems' concrete operator there starts FlagGems' package initializer,
    which probes the accelerator before torch_fl can register it.
    """
    if not isinstance(module_name, str) or not module_name:
        raise SystemExit(
            f"require_flaggems_tle_modules must name modules: {module_name!r}"
        )
    if dependencies.flag_gems is None:
        raise SystemExit("a FlagGems TLE check requires importing flag_gems")
    if module_name in _TLE_IMPORT_FAILURES:
        raise SystemExit(
            f"{module_name} already failed to import in this process; run the "
            "prepared environment in a fresh Python process"
        )
    try:
        if importlib.import_module(module_name) is None:
            raise SystemExit(f"{module_name} did not import")
    except (ImportError, ModuleNotFoundError) as error:
        _TLE_IMPORT_FAILURES.add(module_name)
        if error.name != "shmem":
            raise
        path = Path(__file__).resolve().parents[2] / "torch_fl" / "compile"
        spec = importlib.util.spec_from_file_location(
            "flagtree_ascend_tle", path / "flagtree_ascend_tle.py"
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load the TLE compatibility helper: {path}")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)

        # A failed import may leave a parent package in sys.modules. Retrying
        # here could then skip the old stopgap tester and appear successful.
        helper.ensure_tle_importable()
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as retry_error:
            raise SystemExit(
                f"{module_name} failed to import after the Ascend shmem "
                f"placeholder was registered: {retry_error}"
            ) from retry_error
    print(f"FlagGems TLE imports succeeded: {module_name}")


def assert_model_dir(model_path=None, verify_weights=False, verify_files=False):
    """Check that the mount is a loadable Transformers checkpoint.

    A directory can exist while its host mount is empty or carries an unrelated
    image. Check the configuration before a model test starts, so the failure
    names the checkpoint instead of appearing as an unrecognized model later.
    Weight validation reads only the safetensors headers. It catches a truncated
    transfer without loading the model into host memory.
    """
    model_path = model_path or os.environ.get("MODEL_PATH", "")
    if not model_path:
        raise AssertionError("MODEL_PATH is empty")
    directory = Path(model_path)
    if not directory.is_dir():
        raise AssertionError(f"MODEL_PATH is not a directory: {model_path}")
    config_path = directory / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AssertionError(
            f"MODEL_PATH has no Transformers config.json: {model_path}"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssertionError(
            f"MODEL_PATH config.json cannot be read as JSON: {config_path}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise AssertionError(
            f"MODEL_PATH config.json must be a JSON object: {config_path}"
        )
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise AssertionError(
            f"MODEL_PATH config.json has no non-empty model_type: {config_path}"
        )

    if verify_weights:
        assert_safetensors_header(directory)
    if verify_files:
        for name, expected_size in QWEN3_FILE_SIZES.items():
            path = directory / name
            try:
                size = path.stat().st_size
            except OSError as error:
                raise AssertionError(
                    f"MODEL_PATH lacks downloaded file {path}"
                ) from error
            if size != expected_size:
                raise AssertionError(
                    f"MODEL_PATH file has {size} bytes; {name} needs "
                    f"{expected_size} from Qwen/{QWEN3_REVISION}"
                )

        # The safetensors header proves the checkpoint can be mapped. Pinned
        # whole-file hashes cover the tokenizer and weights content too.
        for name, expected_digest in QWEN3_SHA256.items():
            digest = hashlib.sha256()
            with (directory / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise AssertionError(
                    f"MODEL_PATH file does not match Qwen/{QWEN3_REVISION}: {name}"
                )


def assert_safetensors_header(directory):
    """Read one safetensors header without mapping its tensor data."""
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise AssertionError(f"MODEL_PATH has no .safetensors weights: {directory}")
    for shard in shards:
        try:
            size = shard.stat().st_size
            with shard.open("rb") as stream:
                length_bytes = stream.read(8)
                if len(length_bytes) != 8:
                    raise AssertionError(
                        f"MODEL_PATH safetensors file is truncated: {shard}"
                    )
                header_length = int.from_bytes(length_bytes, "little")
                if not header_length or 8 + header_length > size:
                    raise AssertionError(
                        f"MODEL_PATH safetensors header exceeds file size: "
                        f"{shard} ({header_length} bytes, {size} total)"
                    )
                header = json.loads(stream.read(header_length).decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AssertionError(
                f"MODEL_PATH safetensors header cannot be read: {shard}: {error}"
            ) from error
        if not isinstance(header, dict):
            raise AssertionError(
                f"MODEL_PATH safetensors header must be a JSON object: {shard}"
            )
        tensors = [key for key in header if key != "__metadata__"]
        if not tensors:
            raise AssertionError(
                f"MODEL_PATH safetensors header has no tensors: {shard}"
            )
        print(f"model weights: {shard.name} {size} bytes, {len(tensors)} tensors")


def macho_architectures(path):
    """The architectures one Mach-O file carries, or None if it is not one.

    Read from the file itself rather than asked of `lipo` or `file`: this step
    runs in the venv and the format is two fixed headers wide, so the only
    dependency a subprocess would add is a tool that might be spelled
    differently on the next runner image. `torch_fl._C` is already imported by
    the time this is called, so its own `__file__` is the artifact to inspect --
    a wheel built for the wrong architecture on a host that can still dlopen it
    is the case the caller exists to catch, and its tag cannot be trusted for
    that.
    """
    with open(path, "rb") as handle:
        header = handle.read(8)
    if len(header) < 8:
        return None
    magic = struct.unpack(">I", header[:4])[0]

    # One architecture is a thin file: the load-bearing fact is the cputype
    # that follows the magic, in the header's own byte order.
    mentally_thin = {
        0xFEEDFACE: ">",
        0xCEFAEDFE: "<",
        0xFEEDFACF: ">",
        0xCFFAEDFE: "<",
    }
    if magic in mentally_thin:
        cputype = struct.unpack(mentally_thin[magic] + "i", header[4:8])[0]
        return [_CPU_TYPES.get(cputype, str(cputype))]

    # Both architectures is a fat file, and this is the case the check exists
    # for: "universal2" is two entries, and the quiet failure is a wheel that
    # lists two and contains one. 0xCAFEBABE is the big-endian table, its
    # byte-swapped self the little-endian one; the "fat64" pair carries 32-byte
    # entries and a 64-bit offset the entries beyond the first are useless
    # without. Two architectures fit in 20 bytes either way, so the count and
    # the cputypes are all that is read.
    fat = {0xCAFEBABE: (">", 20), 0xBEBAFECA: ("<", 20)}
    fat |= {0xCAFEBABF: (">", 32), 0xBFBAFECA: ("<", 32)}
    if magic not in fat:
        return None
    endian, entry_size = fat[magic]
    with open(path, "rb") as handle:
        blob = handle.read(8 + entry_size * FAT_ARCH_LIMIT)
    count = struct.unpack(endian + "I", blob[4:8])[0]
    if not 0 < count <= FAT_ARCH_LIMIT:
        return None
    arches = []
    for index in range(count):
        entry = blob[8 + entry_size * index : 8 + entry_size * (index + 1)]
        if len(entry) < 4:
            break
        cputype = struct.unpack(endian + "i", entry[:4])[0]
        arches.append(_CPU_TYPES.get(cputype, str(cputype)))
    return arches or None


def assert_native_extensions_arch(dependencies, expected):
    """The compiled extension is built for the architecture it claims.

    The macOS lane found this the expensive way: a universal2 interpreter makes
    distutils tag the wheel `macosx-10.13-universal2` and compile with `-arch
    arm64 -arch x86_64` while CMake builds the host's architecture alone, so
    `ld` drops the dylib it cannot match and `_C.so` is left without the link
    its platform needs -- a wheel that installs on both architectures and works
    on one. Asserting the architecture of the loaded extension is what makes
    the wheel's tag describe the artifact instead of the interpreter that
    produced it.
    """
    extension = getattr(dependencies.torch_fl, "_C", None)
    assert extension is not None, "torch_fl._C was not imported"
    path = Path(extension.__file__).resolve()
    arches = macho_architectures(path)
    assert arches is not None, f"{path} is not a Mach-O file"
    assert expected in arches, f"{path} is built for {arches}, expected {expected}"
    print(f"native extension: {path} built for {arches}")


class _ArgumentParser(argparse.ArgumentParser):
    """Let pytest's ``-v`` pass through when the loader imports this CLI."""

    def parse_args(self, args=None, namespace=None):
        try:
            return super().parse_args(args, namespace)
        except SystemExit as error:
            # Nested pytest configuration makes the plugin load every module in
            # tests/unit. Patching sys.argv for that reason would be global
            # state; this import-time call is the only invocation inside it.
            if error.code == 0:
                return argparse.Namespace(
                    command="check", path=None, verify_weights=False, verify_files=False
                )
            raise


def main(argv=None):
    parser = _ArgumentParser()
    parser.add_argument(
        "command", nargs="?", choices=("check", "check-model"), default="check"
    )
    parser.add_argument("--path")
    parser.add_argument("--verify-weights", action="store_true")
    parser.add_argument("--verify-files", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "check-model":
        if not arguments.path:
            raise SystemExit("check-model requires --path")
        assert_model_dir(
            arguments.path,
            verify_weights=arguments.verify_weights,
            verify_files=arguments.verify_files,
        )
        print(f"Qwen3 model: {arguments.path}")
        return 0
    return real_main()


def real_main():
    spec = load_spec()
    dependencies = import_dependencies(spec)
    torch = dependencies.torch
    torch_fl = dependencies.torch_fl

    assert_torch_version(dependencies, spec)
    assert_cuda_version(dependencies, spec)

    torch_path = Path(torch.__file__).resolve()
    torch_fl_path = Path(torch_fl.__file__).resolve()
    package_root = torch_fl_path.parent
    assert_torch_path(spec, torch_path)
    assert_package_files(spec, package_root)
    build_kernels = assert_build_kernels(spec, torch_fl, dependencies)

    if spec.get("torch_musa_shim_only"):
        assert_torch_musa_shim()

    assert torch_fl.flagos.is_available(), "flagos device is unavailable"
    flagos_devices = torch_fl.flagos.device_count()
    assert_bound(spec.get("flagos_devices"), flagos_devices, "flagos device count")

    cuda_available = spec.get("cuda_available")
    if cuda_available:
        assert torch.cuda.is_available(), "torch.cuda reports no device"
    if spec.get("cuda_devices") is not None:
        assert_bound(
            spec.get("cuda_devices"),
            torch.cuda.device_count(),
            "torch.cuda device count",
        )

    if spec.get("triton_backend"):
        assert_triton_backend(dependencies, spec["triton_backend"])
    if spec.get("flaggems_vendor"):
        assert_flaggems_vendor(dependencies, spec["flaggems_vendor"])
    tle_modules = spec.get("require_flaggems_tle_modules") or []
    if not isinstance(tle_modules, list) or any(
        not isinstance(name, str) or not name for name in tle_modules
    ):
        raise SystemExit(
            "require_flaggems_tle_modules must be a list of module names: "
            f"{tle_modules!r}"
        )
    for module_name in tle_modules:
        assert_flaggems_tle(module_name, dependencies)
    flagtree_version = None
    if spec.get("flagtree"):
        flagtree_version = assert_flagtree(dependencies, spec["flagtree"])
    if spec.get("native_extensions_arch"):
        assert_native_extensions_arch(dependencies, spec["native_extensions_arch"])

    print(f"environment expectations: {json.dumps(spec, sort_keys=True)}")
    print(f"CPU PyTorch: {torch.__version__}")
    print(f"torch path: {torch_path}")
    print(f"torch_fl path: {torch_fl_path}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"flagos devices: {flagos_devices}")
    if build_kernels is not None:
        print(f"Kernel sets: {sorted(build_kernels)}")
    if cuda_available or spec.get("cuda_devices") is not None:
        print(f"torch.cuda devices: {torch.cuda.device_count()}")
    if dependencies.triton is not None:
        print(
            f"Triton: {dependencies.triton.__version__} "
            f"(backends: {sorted(dependencies.triton.backends.backends)})"
        )
    if dependencies.flag_gems is not None:
        print(
            f"FlagGems: {getattr(dependencies.flag_gems, '__version__', 'unknown')} "
            f"({dependencies.flag_gems.__file__})"
        )
    if flagtree_version:
        print(f"FlagTree: {flagtree_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
