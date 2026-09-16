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

"""
Make FlagTree's ``triton.experimental.tle`` importable without AscendSHMEM.

FlagTree's Ascend wheel ships the TLE (Tensor Language Extension) package with
its Ascend implementation selected at import time, and that implementation
imports the AscendSHMEM Python package in its module body::

    triton/experimental/tle/language/dsa/ascend/communication.py
        import shmem as ash

AscendSHMEM is not declared as a dependency of the FlagTree wheel, is not on
PyPI or on the FlagOS package index, and is not present in FlagTree's own
910C container image, so on a stock Ascend install importing
``triton.experimental.tle`` raises ``ModuleNotFoundError: No module named
'shmem'``.

That would be harmless -- neither torch_fl nor FlagGems uses TLE -- except that
FlagGems' Ascend backend imports ``triton.experimental.tle.language`` at module
scope (``flag_gems/runtime/backend/_ascend/ops/cholesky_solve.py``). The
missing optional dependency therefore makes ``import flag_gems`` itself fail,
which takes down every FlagGems kernel on Ascend, not just the TLE ones.

Registering a placeholder ``shmem`` module restores importability. It is
inert: ``triton.experimental.tle`` only needs the names to exist so its module
bodies can execute, and nothing here is reachable from a FlagGems kernel. Every
entry point raises if it is called, rather than silently returning a plausible
wrong answer. The real fix is FlagTree importing TLE's Ascend communicator
lazily, or shipping AscendSHMEM with the wheel; until then this is what makes
the FlagGems side of the Ascend backend usable at all.
"""

import importlib.util
import sys
import types

# AscendSHMEM's import name. The distribution is not on any index we can reach,
# so there is nothing else to key the check on.
_SHARED_MEMORY_MODULE = "shmem"

_installed = False

_UNUSABLE = (
    "AscendSHMEM (the 'shmem' Python package) is not installed, and torch_fl "
    "registers this placeholder only so that triton.experimental.tle can be "
    "imported. TLE's distributed shared-memory operations are therefore "
    "unavailable; install AscendSHMEM to use them."
)


class _OptionAttr:  # noqa: D101 - mirrors AscendSHMEM's attribute bag
    data_op_engine_type = None


class InitAttr:  # noqa: D101 - mirrors AscendSHMEM's attribute bag
    def __init__(self):
        self.my_rank = 0
        self.n_ranks = 1
        self.local_mem_size = 0
        self.ip_port = ""
        self.option_attr = _OptionAttr()


class OpEngineType:  # noqa: D101 - mirrors AscendSHMEM's enum
    MTE = 0
    ROCE = 1
    RDMA = 2


def _unavailable(*_args, **_kwargs):
    raise NotImplementedError(_UNUSABLE)


def _build_placeholder() -> types.ModuleType:
    module = types.ModuleType(_SHARED_MEMORY_MODULE)
    module.__doc__ = _UNUSABLE
    module.InitAttr = InitAttr
    module.OpEngineType = OpEngineType
    for name in (
        "set_conf_store_tls",
        "aclshmem_init",
        "aclshmem_finalize",
        "aclshmem_create_tensor",
        "aclshmem_free_tensor",
    ):
        setattr(module, name, _unavailable)
    return module


def ensure_tle_importable() -> bool:
    """Register the ``shmem`` placeholder if AscendSHMEM is missing. Idempotent.

    Returns True when this call installed the placeholder, False when a real
    AscendSHMEM is importable and nothing was needed.

    Called before FlagGems is imported, because that import is what reaches
    ``triton.experimental.tle``. Deliberately not gated on the active Triton:
    the check is on whether AscendSHMEM can be imported at all, and on a Triton
    without TLE the placeholder is never reached.
    """
    global _installed
    if _installed:
        return False
    if _SHARED_MEMORY_MODULE in sys.modules:
        return False
    try:
        if importlib.util.find_spec(_SHARED_MEMORY_MODULE) is not None:
            return False
    except (ImportError, ValueError):
        # A broken parent package means it cannot be imported either; fall
        # through and let our placeholder stand in for it.
        pass

    sys.modules[_SHARED_MEMORY_MODULE] = _build_placeholder()
    _installed = True
    return True
