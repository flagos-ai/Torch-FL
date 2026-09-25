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

"""flagos NCCL backend extension.

Exposes c10d::ProcessGroupNCCL on CPU-only torch builds that ship without
USE_C10D_NCCL, by linking against an externally preloaded libtorch_cuda.so.
The compiled module (_flagos_nccl) is built out-of-tree via build.py; it may be
absent in slim installs, in which case ProcessGroupFlagOS falls back to FlagCX.

When it is absent the package attribute ``_flagos_nccl`` is None but still
present, so ``from torch_fl.comm._nccl_ext import _flagos_nccl`` succeeds and
binds the sentinel -- an import error is not what surfaces. Callers must go
through ``get_extension()``, which makes the absence explicit, and report
``load_error()`` rather than letting a later ``NoneType`` AttributeError stand in
for the loader failure that actually explains the situation.
"""

import importlib

try:
    # importlib rather than ``from . import _flagos_nccl``: the from-form goes
    # through _handle_fromlist, which swallows the ModuleNotFoundError for an
    # absent submodule and lets IMPORT_FROM fail instead with "cannot import name
    # '_flagos_nccl' from partially initialized module ... (most likely due to a
    # circular import)" -- a message that names neither the missing module nor
    # the real cause, and whose __context__ is None. A build without the .so is
    # the case this error has to describe, so it is worth getting right.
    _flagos_nccl = importlib.import_module(f"{__name__}._flagos_nccl")
except ImportError as exc:  # pragma: no cover - extension not built
    _flagos_nccl = None
    _load_error = exc
else:
    _load_error = None


def get_extension():
    """Return the compiled _flagos_nccl module, or None when it is not built.

    Callers that cannot proceed without the extension should test this for None
    and treat that as "unavailable"; the value being None is also why the
    original import failure has to be re-reported through ``load_error()``.
    """
    return _flagos_nccl


def load_error():
    """Return the ImportError that kept _flagos_nccl from loading, else None.

    None means the extension is loaded, so ``get_extension()`` is set; otherwise
    this is the diagnostic to surface when the extension is required but
    missing (e.g. a CPU-only torch wheel with no vendor toolkit present).
    """
    return _load_error
