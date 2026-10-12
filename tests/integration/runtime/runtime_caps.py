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

"""Capability data for the hardware-agnostic runtime-conformance suite.

Nothing here decides anything from a platform name. The two kinds of fact the
suite is allowed to branch on both live in files the wheel carries with it:

    * the op-routing conf (``backends_<platform>.conf``), which says which
      backend each op reaches -- for a ``RUNTIME_OPS`` entry, ``none`` is a
      generation gap this suite rejects, not a permissible terminal state;
    * the build record (``torch_fl/_build_config.py``), which says which kernel
      sets were compiled into this wheel.

``detect_platform()`` and any vendor name are absent by construction: a test
that reads a platform name describes the machine it was written on, and this
package is the one contract every machine runs unchanged.

Import-visible surface is stdlib only. ``torch_fl`` is imported inside the
functions that need it, so importing this module -- which the integration
conftest does before ``torch_fl`` has preloaded its device assets -- never
pulls in torch or the package's import-time side effects.
"""

from __future__ import annotations

from pathlib import Path

#: The operators the runtime-conformance dispatch contract exercises. Names are
#: spelled the way the conf and the dispatch log spell them.
RUNTIME_OPS = [
    "add.Tensor",
    "mul.Tensor",
    "mm",
    "bmm",
    "cat",
    "neg",
    "sum.dim_IntList",
    "_softmax",
    "embedding",
    "where.self",
]

#: conf value -> the name ``[flagos dispatch]`` prints for the same backend.
#: The two vocabularies are kept apart on purpose (the log names are frozen for
#: test compatibility); this is the one place the mapping is owned.
_LOG_NAME = {
    "flaggems": "flagos_python",
    "flaggems_python": "flagos_python",
    "flagos_python": "flagos_python",
    "flaggems_cpp": "flagos",
}


def conf_path() -> Path:
    """Path of the op-routing conf this wheel resolved.

    Asked of ``torch_fl`` rather than read from ``FLAGOS_BACKEND_CONFIG``,
    because torch_fl no longer writes that variable: the wheel's own selection
    lives behind ``torch_fl.backend_config_path()``.
    """
    import torch_fl

    return Path(torch_fl.backend_config_path())


def load_routes() -> dict[str, str]:
    """Every ``op -> backend`` route in the conf, as written (``none`` included).

    Parsed the same way the C++ reader does: drop anything after ``#`` and
    split the remainder on the first ``=``. The returned value is the raw conf
    backend, before the log-name mapping ``route_reason`` applies.
    """
    routes: dict[str, str] = {}
    with open(conf_path()) as f:
        for line in f:
            name, sep, value = line.split("#")[0].partition("=")
            if sep:
                routes[name.strip()] = value.strip()
    return routes


def route_reason(op: str) -> tuple[str | None, str]:
    """The backend ``op`` dispatches to, plus why it resolved that way.

    The backend is named the way ``[flagos dispatch]`` names it, so a caller can
    compare it to a log line without restating the conf/log vocabulary split.
    The reason is ``conf_route_none`` when the conf hands ``op`` to
    cpu_fallback -- no dispatch is emitted for such an op -- and ``routed``
    otherwise. An op the conf does not list at all raises: that is a generation
    gap, not a routing decision.
    """
    raw = load_routes().get(op)
    if raw is None:
        raise AssertionError(f"{op} is not listed in {conf_path()}")
    if raw == "none":
        return None, "conf_route_none"
    return _LOG_NAME.get(raw, raw), "routed"


def build_kernels() -> frozenset[str]:
    """Kernel sets compiled into this wheel, from the build record.

    Names are setup.py's (``vendor``, ``flaggems``, ``boxing``, ``flaggems_cpp``,
    ``tileops``); empty for an unbuilt source checkout.
    """
    from torch_fl import _env

    return _env.build_kernels()


def flaggems_cpp_enabled() -> bool:
    """True when the FlagGems C++ runtime was compiled into this wheel."""
    return "flaggems_cpp" in build_kernels()


def require_routed(op: str) -> str:
    """Assert the conf routes ``op`` to an accelerator backend, and return it.

    A ``none`` route for a ``RUNTIME_OPS`` entry is a generation gap, not a
    decision to accept: the suite exists to prove every one of these ten ops
    reaches an accelerator backend on every platform. A platform that cannot
    route one of them must be fixed in its conf/codegen -- so this raises rather
    than skipping. The failure is data-driven (the conf says so), never a
    platform name.
    """
    backend, reason = route_reason(op)
    if backend is None and reason == "conf_route_none":
        raise AssertionError(
            f"{op} routes to none in {conf_path()}: a RUNTIME_OP with no "
            "accelerator backend is a generation gap, not an accepted route"
        )
    return backend
