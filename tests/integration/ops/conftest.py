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

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.integration.skip_policy import add_platform_skip

_INTEGRATION_DIR = str(Path(__file__).resolve().parents[1])
if _INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, _INTEGRATION_DIR)


def _platform_path() -> Path:
    """Locate torch_fl/_platform.py without importing the package.

    From wherever torch_fl resolves: the installed wheel in CI's wheel-only test
    workspace (which has no source tree), or the repo copy in a source checkout.
    A unit-test stub torch_fl carries no _platform.py, so fall back to this
    file's repo copy.
    """
    spec = importlib.util.find_spec("torch_fl")
    if spec is not None and spec.origin:
        candidate = Path(spec.origin).resolve().parent / "_platform.py"
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[3] / "torch_fl" / "_platform.py"


def _load_platform():
    """Load torch_fl/_platform.py by path.

    The one platform detector, shared with the package and
    tests/integration/platform_support.py. Loaded by path because importing
    torch_fl executes torch_fl/__init__.py, whose device-claiming side effects
    must not run while the integration conftest is being loaded.
    """
    loader = importlib.machinery.SourceFileLoader(
        "torch_fl._platform_under_test", str(_platform_path())
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_platform = _load_platform()


# Markers to skip per platform (tests for other backends are not compiled/available).
#
# Every non-MPS row carries `mps`. The migrated Apple files (tests/integration/
# ops/test_mps_*.py) keep a module-level `pytest.mark.mps` and mark the cases
# that are portable contract checks `anyplatform`, so a lane that selects the
# shared cohort collects them and relies on this gate to skip them off Metal.
_PLATFORM_SKIP_MARKERS: dict[str, tuple[str, ...]] = {
    "metax": ("ascend", "musa", "dcu", "mps"),
    "ascend": ("metax", "musa", "dcu", "mps"),
    "musa": ("metax", "ascend", "dcu", "mps", "flaggems_python"),
    "gcu": ("metax", "ascend", "musa", "dcu", "mps"),
    "ppu": ("metax", "ascend", "musa", "dcu", "mps"),
    "cuda": ("metax", "ascend", "musa", "dcu", "mps"),
    "dcu": ("metax", "ascend", "musa", "gcu", "mps"),
    "kunlunxin": ("metax", "ascend", "musa", "dcu", "gcu", "mps"),
    "mps": ("metax", "ascend", "musa", "dcu", "gcu", "ppu"),
    # `flaggems`/`flaggems_python` are deliberately NOT skipped: this platform
    # does build the FlagGems Python path, so those tests are real coverage. They
    # still fail on a TsingMicro host for an unrelated reason -- the TX runtime
    # allows one process per device, and the tests that spawn a subprocess leave
    # it on the device the parent already holds, so the child blocks until the
    # test's 60 s subprocess timeout. Skipping them would hide that.
    "tsingmicro": ("metax", "ascend", "musa", "dcu", "gcu", "mps", "ppu"),
    "iluvatar": ("metax", "ascend", "musa", "dcu", "gcu", "mps", "ppu"),
}


def _flaggems_cpp_enabled() -> bool:
    """True when this wheel has the FlagGems C++ runtime compiled in.

    Read from the build record (setup.py writes ``KERNELS`` into
    ``torch_fl/_build_config.py``), not from an environment variable. It used to
    be ``FLAGOS_USE_FLAGGEMS_CPP``, which had to be exported by hand and kept in
    step with the ``FLAGOS_BUILD_FLAGGEMS_CPP`` build switch; the record cannot disagree with
    the wheel it is inside, so tests marked ``flaggems_cpp`` are now collected
    exactly when the feature exists.
    """
    from torch_fl import _env

    return "flaggems_cpp" in _env.build_kernels()


def _host_compiler_enabled() -> bool:
    """The experimental compiler targets NVIDIA's CPU-Torch FlagGems build."""
    import torch
    from torch_fl import _env, _build_config

    kernels = _env.build_kernels()
    return (
        torch.__version__.split("+")[0] == "2.10.0"
        and torch.version.cuda is None
        and "flaggems" in kernels
        and not getattr(_build_config, "NATIVE_HOSTS", False)
    )


def _native_source_host_enabled() -> bool:
    from torch_fl import _build_config

    return bool(getattr(_build_config, "NATIVE_HOSTS", False))


def _native_vendor_precision_enabled() -> bool:
    from torch_fl import _build_config

    return bool(getattr(_build_config, "NATIVE_VENDOR", False))


def _kunlunxin_xblaslt_enabled() -> bool:
    import torch

    return torch._C._dispatch_has_kernel_for_dispatch_key("aten::linear", "PrivateUse1")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    platform = _platform.detect_platform()
    markers_to_skip = _PLATFORM_SKIP_MARKERS.get(platform, ())
    if platform != "gcu" and "gcu" not in markers_to_skip:
        markers_to_skip += ("gcu",)
    flaggems_cpp_on = _flaggems_cpp_enabled()
    for item in items:
        required_platform = next(
            (
                name
                for name in ("kunlunxin", "tsingmicro")
                if item.get_closest_marker(name)
            ),
            None,
        )
        if required_platform and platform != required_platform:
            add_platform_skip(
                item,
                gate="platform_support",
                detail=f"case requires the {required_platform} runtime",
                reason=f"{required_platform} runtime contract",
            )
            continue
        # Legacy @metax cases assert the unbuilt handwritten mxcc operator
        # backend. Native-source-host cases instead exercise published FlagGems
        # kernels on the MetaX runtime and must reach their build-record gate.
        if (
            platform == "metax"
            and item.get_closest_marker("metax")
            and not item.get_closest_marker("native_source_host")
        ):
            add_platform_skip(
                item,
                gate="backend_marker",
                detail="handwritten MetaX operator backend is not built",
                reason="requires handwritten MetaX operator backend",
            )
            continue
        if item.get_closest_marker("enflame_native_host"):
            if platform != "gcu" or not _native_source_host_enabled():
                add_platform_skip(
                    item,
                    gate="platform_support" if platform != "gcu" else "build_record",
                    detail="requires an Enflame wheel with precompiled native hosts",
                    reason="Enflame native-host contract",
                )
                continue
        nvidia_only = (
            item.get_closest_marker("nvidia")
            or item.get_closest_marker("host_compiler")
            or (
                item.get_closest_marker("native_vendor_precision")
                and not item.get_closest_marker("ppu")
            )
        )
        if nvidia_only and platform != "cuda":
            add_platform_skip(
                item,
                gate="platform_support",
                detail="case requires the NVIDIA runtime",
                reason="NVIDIA runtime contract",
            )
            continue
        if item.get_closest_marker("native_source_host") and platform not in {
            "cuda",
            "dcu",
            "metax",
            "ascend",
            "musa",
            "ppu",
            "tsingmicro",
        }:
            add_platform_skip(
                item,
                gate="platform_support",
                detail="case requires an admitted native-host backend",
                reason="native-host backend contract",
            )
            continue
        if (
            item.get_closest_marker("native_source_host")
            and item.get_closest_marker("ppu")
            and platform != "ppu"
        ):
            add_platform_skip(
                item,
                gate="platform_support",
                detail="case requires the T-Head native-host backend",
                reason="T-Head native-host backend contract",
            )
            continue
        if (
            item.get_closest_marker("native_source_host")
            and not _native_source_host_enabled()
        ):
            add_platform_skip(
                item,
                gate="build_record",
                detail="source hosts require FLAGOS_BUILD_NATIVE_HOST",
                reason="precompiled source host build contract",
            )
            continue
        if (
            item.get_closest_marker("native_vendor_precision")
            and not _native_vendor_precision_enabled()
        ):
            add_platform_skip(
                item,
                gate="build_record",
                detail="wheel has no published native vendor precision module",
                reason="native vendor precision build contract",
            )
            continue
        if (
            item.get_closest_marker("kunlunxin_xblaslt")
            and not _kunlunxin_xblaslt_enabled()
        ):
            add_platform_skip(
                item,
                gate="build_record",
                detail="wheel has no XBLAS linear SDK entry",
                reason="Kunlunxin XBLAS linear contract",
            )
            continue
        if item.get_closest_marker("host_compiler") and not _host_compiler_enabled():
            add_platform_skip(
                item,
                gate="build_record",
                detail="host compiler requires CPU Torch 2.10.0 and FlagGems",
                reason="native host compiler build contract",
            )
            continue
        if minimum := item.get_closest_marker("min_devices"):
            import torch_fl

            required = minimum.args[0]
            if torch_fl.flagos.device_count() < required:
                add_platform_skip(
                    item,
                    gate="device_count",
                    detail=f"case requires {required} visible devices",
                    reason=f"requires {required} visible devices",
                )
                continue
        if item.get_closest_marker("cpython_dict_watchers") and not (
            sys.implementation.name == "cpython" and sys.version_info >= (3, 12)
        ):
            add_platform_skip(
                item,
                gate="platform_support",
                detail="dictionary watchers require CPython 3.12 or newer",
                reason="CPython dictionary watcher runtime contract",
            )
            continue
        if item.get_closest_marker("soft_lowp") and platform not in ("dcu", "metax"):
            add_platform_skip(
                item,
                gate="platform_support",
                detail="software lowp matrix path requires DCU or MetaX",
                reason="software low-precision matrix path requires DCU or MetaX",
            )
            continue
        # The FlagGems C++ path requires a wheel built with the flaggems_cpp
        # kernel set linked in (liboperators.so).
        if item.get_closest_marker("flaggems_cpp") and not flaggems_cpp_on:
            add_platform_skip(
                item,
                gate="build_record",
                detail="flaggems_cpp kernel set is absent from the build record",
                reason=(
                    "FlagGems C++ kernels are not compiled into this wheel "
                    "(rebuild with FLAGOS_BUILD_FLAGGEMS_CPP=ON)"
                ),
            )
            continue
        # NOTE: there is deliberately no gate on the plain `flaggems` mark.
        # Routing is stated per op in the single backends_<platform>.conf
        # (FlagGems first, vendor fallback, CPU fallback), so a flaggems-marked
        # test asserts the conf's decision wherever it is collected -- the
        # retired FLAGOS_USE_FLAGGEMS switch no longer exists to gate it.
        # Manifests keep the vendor/FlagGems group split with `-m "... and not
        # flaggems"` on the vendor side.
        for marker_name in markers_to_skip:
            if item.get_closest_marker(marker_name):
                add_platform_skip(
                    item,
                    gate="backend_marker",
                    detail=f"runtime is {platform}; case requires @{marker_name}",
                    reason=(
                        f"Skipped on {platform} runtime: "
                        f"requires @{marker_name} backend"
                    ),
                )
                break


def pytest_configure(config):
    config.addinivalue_line("markers", "anyplatform: runs on any platform")
    config.addinivalue_line("markers", "nvidia: requires the NVIDIA runtime")
    config.addinivalue_line(
        "markers", "enflame_native_host: requires Enflame native hosts"
    )
    config.addinivalue_line(
        "markers", "host_compiler: requires the experimental native host compiler build"
    )
    config.addinivalue_line(
        "markers", "native_source_host: requires precompiled FlagGems source hosts"
    )
    config.addinivalue_line(
        "markers",
        "native_vendor_precision: requires the published vendor precision cohort",
    )
    config.addinivalue_line("markers", "min_devices(count): minimum visible devices")
    config.addinivalue_line(
        "markers", "cpython_dict_watchers: requires CPython 3.12 dictionary watchers"
    )
    config.addinivalue_line("markers", "metax: requires MetaX platform")
    config.addinivalue_line("markers", "ascend: requires Ascend platform")
    config.addinivalue_line("markers", "musa: requires Moore Threads MUSA platform")
    config.addinivalue_line("markers", "dcu: requires Hygon DCU platform")
    config.addinivalue_line("markers", "mps: requires Apple Silicon (Metal) platform")
    config.addinivalue_line("markers", "kunlunxin: requires the Kunlunxin runtime")
    config.addinivalue_line("markers", "kunlunxin_xblaslt: requires XBLAS linear")
    config.addinivalue_line(
        "markers", "soft_lowp: requires the DCU or MetaX software lowp path"
    )
    config.addinivalue_line(
        "markers",
        "flaggems: asserts the FlagGems route from backends_<platform>.conf "
        "(FlagGems-first routing is the default; no switch needed)",
    )
    config.addinivalue_line(
        "markers",
        "flaggems_cpp: requires torch_fl built with FLAGOS_BUILD_FLAGGEMS_CPP=ON",
    )
    config.addinivalue_line(
        "markers", "flaggems_python: requires FlagGems Python wrapper backend"
    )
    config.addinivalue_line(
        "markers",
        "main_ops: representative operator in the CI smoke subset "
        "(select with -m main_ops); orthogonal to the backend markers",
    )
