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

This was eight hand-written heredocs, one per platform, each re-stating the
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

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ENVIRONMENT_VARIABLE = "FLAGOS_ENVIRONMENT_CHECK"

_BOUNDS = ("at_least", "more_than", "exactly")

# Every key a platform may state. The list is closed on purpose: a misspelled
# key reads as an expectation that was never checked, and the platform would
# pass this step while asserting nothing about the fact it meant to assert.
# tests/unit/test_integration_manifest.py holds the eight configs to this.
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
        "require_imports",
        "torch_musa_shim_only",
        "flagtree",
        "model_dir",
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
    if spec.get("triton_backend") or spec.get("flagtree"):
        import triton

        dependencies.triton = triton
    if spec.get("flaggems_vendor") or spec.get("flagtree"):
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


def assert_build_kernels(spec, torch_fl):
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
    """CUDA: the FlagTree build is the one the wheel's Triton routes need.

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


def assert_model_dir():
    model_path = os.environ.get("MODEL_PATH", "")
    assert model_path and Path(model_path).is_dir(), model_path


def main():
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
    build_kernels = assert_build_kernels(spec, torch_fl)

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
    flagtree_version = None
    if spec.get("flagtree"):
        flagtree_version = assert_flagtree(dependencies, spec["flagtree"])
    if spec.get("model_dir"):
        assert_model_dir()

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
