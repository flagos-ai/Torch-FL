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
    "metax": ("cuda", "ascend", "musa", "dcu", "gcu", "iluvatar", "kunlunxin", "ppu"),
    "ascend": ("cuda", "metax", "musa", "dcu", "gcu", "iluvatar", "kunlunxin", "ppu"),
    "musa": (
        "cuda",
        "metax",
        "ascend",
        "dcu",
        "flaggems_python",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "ppu",
    ),
    "gcu": ("cuda", "metax", "ascend", "musa", "dcu", "iluvatar", "kunlunxin", "ppu"),
    "ppu": (
        "metax",
        "ascend",
        "musa",
        "dcu",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "ppu",
    ),
    "cuda": (
        "metax",
        "ascend",
        "musa",
        "dcu",
        "gcu",
        "iluvatar",
        "kunlunxin",
        "ppu",
    ),
    "dcu": ("metax", "ascend", "musa", "gcu", "iluvatar", "kunlunxin", "ppu"),
    "kunlunxin": (
        "cuda",
        "metax",
        "ascend",
        "musa",
        "dcu",
        "gcu",
        "mps",
        "iluvatar",
        "ppu",
    ),
    "mps": (
        "cuda",
        "metax",
        "ascend",
        "musa",
        "dcu",
        "gcu",
        "ppu",
        "flaggems_python",
        "iluvatar",
        "kunlunxin",
    ),
    "tsingmicro": (
        "cuda",
        "metax",
        "ascend",
        "musa",
        "dcu",
        "gcu",
        "mps",
        "ppu",
        "iluvatar",
        "kunlunxin",
    ),
    "iluvatar": (
        "cuda",
        "metax",
        "ascend",
        "musa",
        "dcu",
        "gcu",
        "mps",
        "ppu",
        "kunlunxin",
    ),
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


def _boxing_enabled() -> bool:
    """True when this wheel has the generated CUDA boxing kernels compiled in.

    Read from the build record (``KERNELS`` in ``torch_fl/_build_config.py``),
    for the same reason ``_flaggems_cpp_enabled()`` is: the record cannot
    disagree with the wheel it lives in. Boxing became an explicit opt-in on
    every CUDA-ABI platform in #520, so the default wheel answers False here and
    a ``cuda``-marked case -- which asserts that an op dispatches to the CUDA
    dispatch key -- has no kernel behind that key to reach.
    """
    from torch_fl import _env

    return "boxing" in _env.build_kernels()


def _tileops_enabled() -> bool:
    """True when this wheel recorded the generated TileOPs kernel set."""
    from torch_fl import _env

    return "tileops" in _env.build_kernels()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply build-record gates to every ops case before collection finishes."""
    platform = _platform.detect_platform()
    markers_to_skip = list(_PLATFORM_SKIP_MARKERS.get(platform, ()))
    # The table gates `metax` to the MetaX lane. The CI wheel does not compile
    # the MXCC route, so dispatch cases asserting `-> metax` opt out of this
    # lane through their runtime backend check; the platform mark must stay
    # usable for MetaX regression cases that test the shipped FlagGems routes.
    flaggems_cpp_on = _flaggems_cpp_enabled()
    # The `cuda` mark asserts a `-> cuda` dispatch, and on a CUDA-ABI platform
    # that route is served by the generated boxing kernels. A FlagGems-primary
    # wheel (the default since #520, and the only wheel the CUDA-ABI CI lanes
    # build now) compiles none, so the ops the conf routes to `cuda` are left
    # unregistered and reach cpu_fallback: the assertion has nothing to hold on
    # and the case skips rather than failing. Platforms whose conf has no `cuda`
    # route at all (ascend, gcu, musa, kunlunxin, bpu, mps) already skip the mark
    # from the table above; appending it again is a no-op for them.
    if not _boxing_enabled():
        markers_to_skip.append("cuda")
    for item in items:
        if item.get_closest_marker("tileops") and not _tileops_enabled():
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "TileOPs kernels are not compiled into this wheel "
                        "(rebuild with the tileops kernel set)"
                    )
                )
            )
            continue
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
    config.addinivalue_line(
        "markers", "tileops: requires the generated TileOPs kernels"
    )
    config.addinivalue_line(
        "markers", "metax: runs only on MetaX (not a native MXCC route assertion)"
    )
    config.addinivalue_line("markers", "ascend: requires Ascend platform")
    config.addinivalue_line("markers", "musa: requires Moore Threads MUSA platform")
    config.addinivalue_line("markers", "dcu: requires Hygon DCU platform")
    config.addinivalue_line("markers", "mps: requires Apple Silicon (Metal) platform")
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
