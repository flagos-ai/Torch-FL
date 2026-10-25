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
      backend each op reaches. ``none`` is an explicit unimplemented route for
      this platform; the suite reports it as a coverage gap and does not attempt
      to read a dispatch line that a cpu_fallback call never writes. An op
      missing from the conf is still a generation gap and fails;
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
    # A conf row in the vendor spelling, against the id the log prints. The
    # NVIDIA rows print "boxing" (the slot they register into).
    "apple": "mps",
    "mthreads": "musa",
    "enflame": "gcu",
    "nvidia": "boxing",
    # The day-0 variant spells the same slot `boxing` (setup.py's day-0 name is
    # the log name), and the log prints `boxing` for it.
    "boxing": "boxing",
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


def effective_route(op: str) -> tuple[str | None, str]:
    """The route this wheel can verify, or ``unimplemented`` for a ``none``.

    ``none`` remains visible in the conf and in the summary's ``ops_none``
    count. The second value states whether the dispatch probe has work to do,
    so a test can distinguish this coverage gap from a missing conf entry.
    """
    backend, reason = route_reason(op)
    if backend is None and reason == "conf_route_none":
        return None, "unimplemented"
    return backend, reason


def unimplemented_runtime_ops() -> list[str]:
    """``RUNTIME_OPS`` this platform's conf explicitly routes to ``none``.

    This is a coverage report, not an allowance to omit an op from the conf.
    The dispatch contract asserts every op that has a route; the summary exposes
    the remaining unimplemented kernels so CI cannot hide them behind skips.
    """
    return [op for op in RUNTIME_OPS if effective_route(op) == (None, "unimplemented")]


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


def routed_backend(op: str) -> str:
    """Return the accelerator backend the conf assigns to ``op``."""
    backend, reason = route_reason(op)
    if backend is None and reason == "conf_route_none":
        raise AssertionError(
            f"{op} routes to none in {conf_path()}: there is no dispatch to verify"
        )
    return backend
