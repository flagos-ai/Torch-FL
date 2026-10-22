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

"""Load individual DCU support modules without executing ``torch_fl/__init__.py``.

Importing the package runs ``setup_dcu_runtime()``, which dlopens DTK's device
libraries against whatever torch is installed.  On a DCU build host, where the
system interpreter often has DTK's *own* torch wheel, that means two copies of
the core runtime in one process and an abort in duplicate static init -- during
test collection, before any test runs.

The modules under test here depend only on the standard library at import time
(torch is imported lazily inside their functions), so they can be exec'd by path
under stub parent packages.  That keeps these unit tests runnable on any host.
"""

import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_STUB_PACKAGES = (
    ("torch_fl", "torch_fl"),
    ("torch_fl.accelerator", "torch_fl/accelerator"),
    ("torch_fl.accelerator.dcu", "torch_fl/accelerator/dcu"),
)


def _exec_from_file(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_module(relpath, alias, deps=()):
    """Exec ``relpath`` as ``alias``, first exec'ing any intra-package ``deps``.

    ``deps`` are ``(dotted_name, relpath)`` pairs that the target imports by
    absolute name and therefore must exist in ``sys.modules`` beforehand.  Real
    ``torch_fl`` modules already loaded in this process are left alone and
    restored afterwards, so this never shadows a genuine installed package.
    """
    names = ["torch_fl", *(name for name, _ in _STUB_PACKAGES[1:])]
    names += [name for name, _ in deps]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        if saved["torch_fl"] is None:
            for name, pkg_rel in _STUB_PACKAGES:
                pkg = types.ModuleType(name)
                pkg.__path__ = [str(REPO / pkg_rel)]
                sys.modules[name] = pkg
            for name, dep_rel in deps:
                _exec_from_file(name, dep_rel)
        return _exec_from_file(alias, relpath)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
