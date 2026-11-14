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
# Without the entry the shared selection ran test_mps_argmin_arpextreme.py on
# MUSA hardware and its 13 NaN-placement cases failed: mudnn's reductions do not
# name a NaN the way the host reference does, and no Metal kernel is involved.
_PLATFORM_SKIP_MARKERS: dict[str, tuple[str, ...]] = {
    "metax": ("cuda", "ascend", "musa", "dcu", "mps"),
    "ascend": ("cuda", "metax", "musa", "dcu", "mps"),
    # MUSA builds compile no CUDA boxing kernels (no cudart on the platform), so
    # `-> boxing` routing assertions cannot hold; nor is FlagGems built.
    "musa": ("cuda", "metax", "ascend", "dcu", "mps", "flaggems_python"),
    # GCU has no CUDA boxing runtime, but it does compile the FlagGems Python
    # dispatcher alongside topsaten and selects between them per overload.
    "gcu": ("cuda", "metax", "ascend", "musa", "dcu", "mps"),
    # PPU is a CUDA-ABI platform: when its wheel carries the boxing kernels the
    # CUDA dispatch key serves the vendor kernels and cuda-marked tests run
    # here -- same skip set as cuda, which is where PPU implicitly landed before
    # it was named. Naming it makes the platform first-class in this gate
    # instead of correct by accident of the fallback bucket. On a default
    # (FlagGems-primary) wheel the build-record gate below adds `cuda` to this
    # set, because that key has no kernel behind it.
    "ppu": ("metax", "ascend", "musa", "dcu", "mps"),
    # cuda -- and an unidentified wheel, which detect_platform() defaults to --
    # runs the CUDA-marked tests when its wheel carries the boxing kernels, and
    # skips every other backend's. The build-record gate below covers the
    # default FlagGems-primary wheel, where the `cuda` route has no kernels.
    "cuda": ("metax", "ascend", "musa", "dcu", "mps"),
    "dcu": ("metax", "ascend", "musa", "gcu", "mps"),
    # Kunlunxin is a native-kernel vendor like musa/gcu/ascend: backends_kunlunxin.conf
    # routes ops to the native `kunlunxin` key (the xdnn op library) or leaves them at
    # `none` (CPU fallback), with no `cuda` route, so a `-> boxing` dispatch assertion
    # cannot hold. Every other backend's marker is skipped for the same reason.
    "kunlunxin": ("cuda", "metax", "ascend", "musa", "dcu", "gcu", "mps"),
    # Apple Silicon (MPS) is a native Metal backend: it compiles neither the CUDA
    # boxing kernels nor any other vendor's kernels, and FlagGems/Triton has no
    # Metal line, so every other backend's marked tests are skipped here. The
    # FlagGems Python mark is a wheel capability question, not a platform
    # exclusion: a migrated case checks inside its body whether this build can
    # serve the override, then records the gap when it cannot.
    "mps": ("cuda", "metax", "ascend", "musa", "dcu", "gcu", "ppu"),
    # TsingMicro is FlagGems-primary and compiles no CUDA boxing kernels and no
    # vendor op library, so `-> boxing` and every other backend's dispatch
    # assertions cannot hold. Without this row the platform fell through to the
    # empty skip set and the cuda/ascend-marked tests ran here and failed.
    #
    # `flaggems`/`flaggems_python` are deliberately NOT skipped: this platform
    # does build the FlagGems Python path, so those tests are real coverage. They
    # still fail on a TsingMicro host for an unrelated reason -- the TX runtime
    # allows one process per device, and the tests that spawn a subprocess leave
    # it on the device the parent already holds, so the child blocks until the
    # test's 60 s subprocess timeout. Skipping them would hide that.
    "tsingmicro": ("cuda", "metax", "ascend", "musa", "dcu", "gcu", "mps", "ppu"),
    # CoreX exposes a CUDA-shaped runtime, but Iluvatar uses its native vendor
    # slot rather than CUDA boxing. Keep FlagGems coverage enabled.
    "iluvatar": ("cuda", "metax", "ascend", "musa", "dcu", "gcu", "mps", "ppu"),
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


def _host_compiler_enabled() -> bool:
    """The experimental compiler targets NVIDIA's CPU-Torch FlagGems build."""
    import torch
    from torch_fl import _env

    kernels = _env.build_kernels()
    return (
        torch.__version__.split("+")[0] == "2.10.0"
        and torch.version.cuda is None
        and "flaggems" in kernels
        and "boxing" not in kernels
    )


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
    # The `cuda` mark asserts a `-> boxing` dispatch, and on a CUDA-ABI platform
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
        if (
            item.get_closest_marker("nvidia")
            or item.get_closest_marker("host_compiler")
        ) and platform != "cuda":
            add_platform_skip(
                item,
                gate="platform_support",
                detail="case requires the NVIDIA runtime",
                reason="NVIDIA runtime contract",
            )
            continue
        if item.get_closest_marker("host_compiler") and not _host_compiler_enabled():
            add_platform_skip(
                item,
                gate="build_record",
                detail="host compiler requires CPU Torch 2.10.0 and FlagGems without boxing",
                reason="native host compiler build contract",
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
    config.addinivalue_line("markers", "cuda: requires CUDA platform")
    config.addinivalue_line("markers", "nvidia: requires the NVIDIA runtime")
    config.addinivalue_line(
        "markers", "host_compiler: requires the experimental native host compiler build"
    )
    config.addinivalue_line("markers", "metax: requires MetaX platform")
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
