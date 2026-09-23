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
from pathlib import Path

import pytest


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
_PLATFORM_SKIP_MARKERS: dict[str, tuple[str, ...]] = {
    "metax": ("cuda", "ascend", "musa", "dcu"),
    "ascend": ("cuda", "metax", "musa", "dcu"),
    # MUSA builds compile no CUDA boxing kernels (no cudart on the platform), so
    # `-> cuda` routing assertions cannot hold; nor is FlagGems built.
    "musa": ("cuda", "metax", "ascend", "dcu", "flaggems_python"),
    # GCU has no CUDA boxing runtime, but it does compile the FlagGems Python
    # dispatcher alongside topsaten and selects between them per overload.
    "gcu": ("cuda", "metax", "ascend", "musa", "dcu"),
    # PPU is a CUDA-ABI boxing backend: the CUDA dispatch key carries vendor
    # kernels, so cuda-marked tests run here -- same skip set as cuda, which is
    # where PPU implicitly landed before it was named. Naming it makes the
    # platform first-class in this gate instead of correct by accident of the
    # fallback bucket.
    "ppu": ("metax", "ascend", "musa", "dcu"),
    # cuda -- and an unidentified wheel, which detect_platform() defaults to --
    # runs the CUDA-marked tests and skips every other backend's.
    "cuda": ("metax", "ascend", "musa", "dcu"),
    "dcu": ("metax", "ascend", "musa", "gcu"),
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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    platform = _platform.detect_platform()
    markers_to_skip = list(_PLATFORM_SKIP_MARKERS.get(platform, ()))
    # MetaX builds are boxing-only: the hand-written mxcc backend is NOT
    # compiled, so ops run through the CUDA boxing kernels (and optionally the
    # FlagGems Python path). Tests asserting a `-> metax` dispatch (mark.metax)
    # cannot pass, so skip them. No mode variable to consult -- the platform
    # name already says it.
    if platform == "metax":
        markers_to_skip.append("metax")
    flaggems_cpp_on = _flaggems_cpp_enabled()
    for item in items:
        if item.get_closest_marker("soft_lowp") and platform not in ("dcu", "metax"):
            item.add_marker(
                pytest.mark.skip(
                    reason="software low-precision matrix path requires DCU or MetaX"
                )
            )
            continue
        # The FlagGems C++ path requires a wheel built with the flaggems_cpp
        # kernel set linked in (liboperators.so).
        if item.get_closest_marker("flaggems_cpp") and not flaggems_cpp_on:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "FlagGems C++ kernels are not compiled into this wheel "
                        "(rebuild with FLAGOS_BUILD_FLAGGEMS_CPP=ON)"
                    )
                )
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
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"Skipped on {platform} runtime: "
                            f"requires @{marker_name} backend"
                        )
                    )
                )
                break


def pytest_configure(config):
    config.addinivalue_line("markers", "anyplatform: runs on any platform")
    config.addinivalue_line("markers", "cuda: requires CUDA platform")
    config.addinivalue_line("markers", "metax: requires MetaX platform")
    config.addinivalue_line("markers", "ascend: requires Ascend platform")
    config.addinivalue_line("markers", "musa: requires Moore Threads MUSA platform")
    config.addinivalue_line("markers", "dcu: requires Hygon DCU platform")
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
