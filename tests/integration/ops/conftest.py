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
import os
from pathlib import Path

import pytest


def _build_accelerator() -> str:
    """The accelerator the installed wheel was built for ("" if unknown).

    From the build record setup.py writes (torch_fl/_build_config.py), the same
    source torch_fl itself reads: FLAGOS_ACCELERATOR is a build input and no
    longer overrides the record at run time, so reading it here would let a
    stale export from another build choose this gate's skip set.

    find_spec() locates the package without executing it, so this stays free of
    the torch import that must not happen before the assets are preloaded; the
    record itself imports nothing.
    """
    try:
        spec = importlib.util.find_spec("torch_fl")
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.origin:
        return ""
    path = Path(spec.origin).resolve().parent / "_build_config.py"
    try:
        loader = importlib.machinery.SourceFileLoader("_flagos_build_record", str(path))
        module = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(loader.name, loader)
        )
        loader.exec_module(module)
    except (OSError, ImportError, SyntaxError):
        # A source checkout with no build yet; the marker below is what
        # identifies such an install.
        return ""
    return str(getattr(module, "ACCELERATOR", "")).strip().lower()


def _resolved_conf() -> str:
    """Path of the conf the routing table is read from, "" if none resolved.

    Asked of torch_fl rather than read from the environment, which torch_fl no
    longer writes: FLAGOS_BACKEND_CONFIG now holds only what a user set, and the
    wheel's own choice lives in torch_fl.backend_config_path(). A stub package
    with no accessor (the unit tests import one) falls back to the variable,
    which is what that accessor itself falls back to.
    """
    try:
        import torch_fl
    except ImportError:
        return os.environ.get("FLAGOS_BACKEND_CONFIG", "")
    resolve = getattr(torch_fl, "backend_config_path", None)
    return resolve() if resolve else os.environ.get("FLAGOS_BACKEND_CONFIG", "")


def _detect_platform() -> str:
    """Infer the active hardware/backend platform.

    The accelerator is read from the wheel's build record, which is what
    torch_fl consults. The lib/flagos_platform marker that native-kernel builds
    write is authoritative for those platforms, and the name of the conf torch_fl
    resolved is the last resort.

    Every chip has its own record value, PPU included (it is a CUDA-ABI boxing
    vendor, not a cuda build). Wheels built before PPU had a value of its own
    report cuda and are still recognised through the PPU_SDK environment or the
    lib_ppu/ bundle directory.
    """
    accelerator = _build_accelerator()
    if accelerator == "ascend":
        return "ascend"
    if accelerator in ("metax", "maca"):
        return "metax"
    if accelerator == "musa":
        return "musa"
    if accelerator == "dcu":
        return "dcu"
    if accelerator == "ppu":
        return "ppu"
    if os.environ.get("PPU_SDK"):
        return "ppu"

    try:
        import torch_fl

        lib_root = os.path.dirname(torch_fl.__file__)
        try:
            with open(os.path.join(lib_root, "lib", "flagos_platform")) as f:
                marker = f.read().strip().lower()
            if marker:
                return marker
        except OSError:
            pass
        # PPU wheels ship no flagos_platform marker (CMake writes it only for
        # gcu/musa/bpu/ascend); their signal is the lib_ppu/ bundle dir. This
        # check must not sit behind the marker read: a missing marker raises
        # before it, which is exactly the PPU layout.
        if os.path.isdir(os.path.join(lib_root, "lib_ppu")):
            return "ppu"
    except ImportError:
        pass

    backend_cfg = _resolved_conf().lower()
    if "ascend" in backend_cfg:
        return "ascend"
    if "metax" in backend_cfg:
        return "metax"
    if "musa" in backend_cfg:
        return "musa"
    if "ppu" in backend_cfg:
        return "ppu"
    return "default"


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
    # kernels, so cuda-marked tests run here -- same skip set as "default",
    # which is where PPU implicitly landed before it was named. Naming it
    # makes the platform first-class in this gate instead of correct by
    # accident of the fallback bucket.
    "ppu": ("metax", "ascend", "musa", "dcu"),
    "default": ("metax", "ascend", "musa", "dcu"),
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
    platform = _detect_platform()
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
