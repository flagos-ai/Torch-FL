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

"""Resolve an operator's configured backend into its dispatch-log name.

Every platform ships exactly one full-coverage conf (``backends_<platform>.conf``)
and that file -- not an environment switch -- is what decides routing, so the
backend named in ``[flagos dispatch] <op> -> <backend>`` has to be the one the
conf lists. A test that asserts a dispatch log should therefore ask this module
for the route instead of hard-coding ``cuda`` or ``flagos_python``: a hard-coded
name describes the platform the test was written on and silently becomes wrong on
every other one. ``add.Tensor``/``cat`` are the measured examples -- ``cuda`` on
CUDA/DCU/PPU, ``musa`` on MUSA.

The two sides do not share one vocabulary. The conf names the FlagGems Python
path ``flaggems`` while the log prints ``flagos_python``, and the C++ path is
``flaggems_cpp`` in the conf against ``flagos`` in the log -- both log names are
retained deliberately for test compatibility
(``csrc/aten/dispatcher.h``, ``LogDispatch``). This module owns that mapping so
the test files do not each restate it.
"""

from __future__ import annotations

import torch_fl  # noqa: F401  -- importing resolves the conf torch_fl routes by


# conf value -> the name LogDispatch prints for the same Backend enum value.
_LOG_NAME = {
    "flaggems": "flagos_python",
    "flaggems_python": "flagos_python",
    "flagos_python": "flagos_python",
    "flaggems_cpp": "flagos",
}


def _conf_path() -> str:
    """The conf the routing table is read from, as torch_fl resolved it.

    Asked of torch_fl rather than read from the environment, because torch_fl no
    longer writes FLAGOS_BACKEND_CONFIG: the variable now holds only what a user
    set, and the wheel's own selection lives in torch_fl.backend_config_path().
    """
    return torch_fl.backend_config_path()


def _conf_route(op: str) -> str:
    """Raw conf value for ``op`` -- ``none`` included, and ``none`` unmapped.

    An op missing from the conf is a generation gap rather than a routing
    decision, so it raises here for both public accessors instead of being
    folded into one of them.
    """
    conf = _conf_path()
    assert conf, "no backend conf resolved; import torch_fl before asking for a route"
    with open(conf) as f:
        for line in f:
            name, sep, value = line.split("#")[0].partition("=")
            if sep and name.strip() == op:
                return value.strip()
    raise AssertionError(f"{op} is not listed in {conf}")


def routed_backend(op: str) -> str:
    """Backend name this platform's conf routes ``op`` to, as the log prints it.

    ``op`` is spelled the way the conf and the dispatch log spell it
    (``add.Tensor``, ``_softmax``, ``cat``). The result is what
    ``LogDispatch`` writes after ``->`` on this platform.

    Raises rather than guessing: an op the conf routes to ``none`` reaches
    ATen's cpu_fallback and never logs a dispatch at all, and an op absent from
    the conf is a generation gap -- both are failures a caller wants to see, not
    values to assert against.
    """
    backend = _conf_route(op)
    conf = _conf_path()
    assert backend != "none", (
        f"{op} is routed to 'none' in {conf}: it reaches cpu_fallback "
        "and never logs a dispatch, so there is no route to assert"
    )
    return _LOG_NAME.get(backend, backend)


def routed_backend_or_none(op: str) -> str | None:
    """Like ``routed_backend``, but ``None`` for an op the conf routes to ``none``.

    The two accessors split on what the caller does with a cpu_fallback route.
    ``routed_backend`` is for a test asserting a log line: there is nothing to
    assert, so raising is the useful answer. This one is for a test that is
    *vacuous* rather than wrong on a platform that does not route the op -- a
    per-overload routing assertion on a platform whose conf hands the op to
    cpu_fallback, say -- and wants to skip instead of fail. ``None`` here means
    "the op is deliberately unaccelerated on this platform", not "no impl was
    found"; an op absent from the conf still raises.
    """
    backend = _conf_route(op)
    return None if backend == "none" else _LOG_NAME.get(backend, backend)


#: conf backend value -> the build-record kernel set whose kernels serve it.
#: ``_LOG_NAME`` above translates a conf value into what the dispatch log prints;
#: this translates the same value into what the build record (``KERNELS`` in
#: ``torch_fl/_build_config.py``) calls the code that backs it, which is the
#: other half of "can this wheel reach the route it was handed".
_BACKEND_KERNEL_SET = {
    "flaggems": "flaggems",
    "flaggems_python": "flaggems",
    "flagos_python": "flaggems",
    "flaggems_cpp": "flaggems_cpp",
    "cuda": "boxing",
    "tileops": "tileops",
}

#: The keys a native-kernel vendor routes to. Each platform's conf names its own
#: (``ascend`` on Ascend, ``musa`` on MUSA, ``gcu`` on GCU, ``kunlunxin`` on
#: Kunlunxin, ``iluvatar`` on Iluvatar, ``tsingmicro`` on TsingMicro), and
#: setup.py compiles every one of them from the single ``vendor`` kernel set
#: rather than a per-vendor switch -- so one membership test answers "did this
#: build compile the native kernels" for all of them.
_VENDOR_BACKENDS = frozenset(
    {"ascend", "musa", "gcu", "kunlunxin", "iluvatar", "tsingmicro"}
)


def route_is_served(op: str) -> bool:
    """True when this wheel compiled the backend the conf routes ``op`` to.

    A conf states per-op routing for a *platform*, not for one build of it: it
    names the CUDA boxing kernels whether or not ``FLAGOS_BUILD_BOXING`` pulled
    them in, and the FlagGems path whether or not ``FLAGOS_BUILD_FLAGGEMS`` did.
    When the named backend is not compiled in, the op is not claimed on
    PrivateUse1 at all, so the call reaches ATen's cpu_fallback and nothing logs
    a dispatch -- a test asserting ``[flagos dispatch] <op> -> <backend>`` then
    has nothing to hold on and has to skip rather than fail. That is exactly the
    default (FlagGems-primary) wheel's shape for every op
    ``backends_cuda.conf`` routes to ``cuda``: no boxing kernels, no dispatch.

    Both facts are asked of the wheel (``torch_fl._env.build_kernels()``), never
    of an environment variable, so the answer cannot disagree with the build it
    describes. ``none`` and an op the conf does not list both answer False:
    neither names a backend this wheel reaches, which is what a caller asking
    this question wants to hear.
    """
    backend = _conf_route(op)
    if backend == "none":
        return False
    kernels = torch_fl._env.build_kernels()
    if backend in _VENDOR_BACKENDS:
        return "vendor" in kernels
    kernel_set = _BACKEND_KERNEL_SET.get(backend)
    return kernel_set is not None and kernel_set in kernels
