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

import os

import torch_fl  # noqa: F401  -- importing resolves FLAGOS_BACKEND_CONFIG


# conf value -> the name LogDispatch prints for the same Backend enum value.
_LOG_NAME = {
    "flaggems": "flagos_python",
    "flaggems_python": "flagos_python",
    "flagos_python": "flagos_python",
    "flaggems_cpp": "flagos",
}


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
    conf = os.environ.get("FLAGOS_BACKEND_CONFIG")
    assert conf, (
        "FLAGOS_BACKEND_CONFIG is unset; import torch_fl before asking for a route"
    )
    with open(conf) as f:
        for line in f:
            name, sep, value = line.split("#")[0].partition("=")
            if sep and name.strip() == op:
                backend = value.strip()
                assert backend != "none", (
                    f"{op} is routed to 'none' in {conf}: it reaches cpu_fallback "
                    "and never logs a dispatch, so there is no route to assert"
                )
                return _LOG_NAME.get(backend, backend)
    raise AssertionError(f"{op} is not listed in {conf}")
