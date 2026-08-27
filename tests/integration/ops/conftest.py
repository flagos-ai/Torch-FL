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

import os

import pytest


def _detect_platform() -> str:
    """Infer the active hardware/backend platform.

    ACCELERATOR is a *build*-time variable, so it is usually absent when running
    the tests against an installed wheel. The lib/flagos_platform marker that
    native-kernel builds write is authoritative in that case, and the resolved
    FLAGOS_BACKEND_CONFIG name is the last resort.
    """
    accelerator = os.environ.get("ACCELERATOR", "").lower()
    if accelerator == "ascend":
        return "ascend"
    if accelerator in ("metax", "maca"):
        return "metax"
    if accelerator == "musa":
        return "musa"
    if accelerator == "dcu":
        return "dcu"

    try:
        import torch_fl

        marker = os.path.join(
            os.path.dirname(torch_fl.__file__), "lib", "flagos_platform"
        )
        with open(marker) as f:
            platform = f.read().strip().lower()
        if platform:
            return platform
    except (ImportError, OSError):
        pass

    backend_cfg = os.environ.get("FLAGOS_BACKEND_CONFIG", "").lower()
    if "ascend" in backend_cfg:
        return "ascend"
    if "metax" in backend_cfg:
        return "metax"
    if "musa" in backend_cfg:
        return "musa"
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
    "default": ("metax", "ascend", "musa", "dcu"),
    "dcu": ("metax", "ascend", "musa", "gcu"),
}


def _flaggems_cpp_enabled() -> bool:
    """True when the FlagGems C++ runtime path is switched on (FLAGOS_USE_FLAGGEMS_CPP=1).

    Tests marked ``flaggems_cpp`` require a wheel built with FLAGGEMS_KERNEL=ON
    (liboperators.so linked in) and FLAGOS_USE_FLAGGEMS_CPP=1 at runtime; they
    are skipped when the env var is off (default).
    """
    return os.environ.get("FLAGOS_USE_FLAGGEMS_CPP", "0").lower() not in (
        "0",
        "",
        "off",
        "false",
    )


def _flaggems_enabled() -> bool:
    """True when the FlagGems runtime path is switched on (FLAGOS_USE_FLAGGEMS=1).

    FlagGems and the vendor kernels are BOTH compiled into every wheel; which one
    an op runs on is chosen at runtime by this env var (see torch_fl.__init__
    ._select_backend_config -> backends_flaggems.conf). Tests marked ``flaggems``
    assert a ``-> flagos_python`` (or vendor-fallback) routing that only holds
    when the switch is on, so they are skipped otherwise.
    """
    return os.environ.get("FLAGOS_USE_FLAGGEMS", "0").lower() not in (
        "0",
        "",
        "off",
        "false",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    platform = _detect_platform()
    markers_to_skip = list(_PLATFORM_SKIP_MARKERS.get(platform, ()))
    # In MetaX boxing mode the hand-written mxcc backend is NOT compiled: ops run
    # through the CUDA boxing kernels (and optionally the FlagGems Python path).
    # Tests asserting a `-> metax` dispatch (mark.metax) cannot pass, so skip them.
    if platform == "metax" and os.environ.get("FLAGOS_METAX_BOXING", "0") == "1":
        markers_to_skip.append("metax")
    flaggems_on = _flaggems_enabled()
    flaggems_cpp_on = _flaggems_cpp_enabled()
    for item in items:
        # The FlagGems C++ path requires a FLAGGEMS_KERNEL=ON wheel and runtime env.
        if item.get_closest_marker("flaggems_cpp") and not flaggems_cpp_on:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "FlagGems C++ path is off "
                        "(set FLAGOS_USE_FLAGGEMS_CPP=1 with a FLAGGEMS_KERNEL=ON wheel)"
                    )
                )
            )
            continue
        # The flaggems runtime path is a runtime switch, not a build/platform gate:
        # skip its tests only when the switch is off, on any platform.
        if item.get_closest_marker("flaggems") and not flaggems_on:
            item.add_marker(
                pytest.mark.skip(
                    reason="FlagGems runtime path is off (set FLAGOS_USE_FLAGGEMS=1)"
                )
            )
            continue
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
        "markers",
        "flaggems: requires the FlagGems runtime path on (FLAGOS_USE_FLAGGEMS=1)",
    )
    config.addinivalue_line(
        "markers",
        "flaggems_cpp: requires torch_fl built with FLAGGEMS_KERNEL=ON and "
        "FLAGOS_USE_FLAGGEMS_CPP=1 at runtime",
    )
    config.addinivalue_line(
        "markers", "flaggems_python: requires FlagGems Python wrapper backend"
    )
    config.addinivalue_line(
        "markers",
        "main_ops: representative operator in the CI smoke subset "
        "(select with -m main_ops); orthogonal to the backend markers",
    )
