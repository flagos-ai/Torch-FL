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

"""Select one installed torch_fl across a source test session.

The checkout is on ``sys.path`` for its test helpers, and pytest's ``pythonpath``
setting can put it ahead of an installed wheel. A pre-import hook removes only
the source ``torch_fl`` entry, leaving the checkout available for scripts and
fixtures. An import after that entry was added clears the source package from
``sys.modules`` so a conftest and its subprocesses use the same native wheel.

Tests that install a stub package deliberately select their own path. During
that selection this hook stands down until their path entry is removed.
"""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


_PACKAGE = "torch_fl"
_INSTALLED_ROOT = ""
_SOURCE_ROOT = ""
_PATH_HOOK_INSTALLED = False
_SELECTION_HOOK_TAG = "_flagos_torch_fl_host"


def _same_path(first, second):
    try:
        return Path(first).resolve() == Path(second).resolve()
    except (OSError, ValueError):
        return False


def _is_source_torch_fl(entry):
    """Whether a path entry contains the checkout, rather than the wheel."""
    source = os.environ.get("TORCH_FL_SOURCE_ROOT", "")
    return bool(entry and source and _same_path(entry, source))


def _installed_root():
    if _INSTALLED_ROOT:
        return _INSTALLED_ROOT
    selected = os.environ.get("TORCH_FL_INSTALLED_PACKAGE_ROOT", "")
    if not selected:
        return ""
    root = str(Path(selected).resolve())
    package = Path(root) / _PACKAGE
    if not (package / "__init__.py").is_file():
        raise ImportError(f"installed torch_fl package does not exist: {package}")
    if os.environ.get("TORCH_FL_TEST_SITE_PACKAGE") != "1" and not any(
        (package / f"_C{suffix}").is_file()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise ImportError(f"installed torch_fl package has no native _C: {package}")
    return root


def _contains_installed_module(name, value):
    """Whether a loaded module came from the selected wheel."""
    top_level = name.split(".", 1)[0]
    if top_level != _PACKAGE:
        return True
    if value is None:
        return True
    root = _installed_root()
    if not root:
        return True
    locations = [getattr(value, "__file__", "")]
    locations.extend(getattr(value, "__path__", ()))
    package = Path(root)
    for location in locations:
        if not location:
            continue
        try:
            selected = Path(location).resolve()
            at_root = selected.name == _PACKAGE and _same_path(selected.parent, package)
            if not selected.is_relative_to(package) and not at_root:
                return False
        except (OSError, RuntimeError, ValueError):
            return False
    return True


def _source_modules(name):
    """Remove source package modules when replacing them with the wheel."""
    if name.split(".", 1)[0] != _PACKAGE:
        return
    return [
        key
        for key, value in list(sys.modules.items())
        if key.split(".", 1)[0] == _PACKAGE
        and not _contains_installed_module(key, value)
    ]


def install():
    """Put the selected wheel ahead of its source copy in this process."""
    global _INSTALLED_ROOT, _SOURCE_ROOT, _PATH_HOOK_INSTALLED
    root = _installed_root()
    if not root:
        return None
    if not _INSTALLED_ROOT:
        _INSTALLED_ROOT = root
        _SOURCE_ROOT = os.environ.get("TORCH_FL_SOURCE_ROOT", "")
        for name in _source_modules(_PACKAGE):
            sys.modules.pop(name, None)
        _install_import_hook()
    root_path = Path(root)
    sys.path[:] = [
        entry
        for entry in sys.path
        if not _is_source_torch_fl(entry) and not _same_path(entry, root_path)
    ]
    sys.path.insert(0, root)
    if _SOURCE_ROOT and not _same_path(_SOURCE_ROOT, Path.cwd()):
        sys.path.append(_SOURCE_ROOT)
    importlib.invalidate_caches()
    loaded = sys.modules.get(_PACKAGE)
    if loaded is not None and not _contains_installed_module(_PACKAGE, loaded):
        for name in [name for name in sys.modules if name.split(".", 1)[0] == _PACKAGE]:
            sys.modules.pop(name, None)
        loaded = None
    if loaded is None:
        # A reload must run the wheel's initializer so a private torch facade
        # is selected before any test module imports torch.
        spec = importlib.util.spec_from_file_location(
            _PACKAGE,
            root_path / _PACKAGE / "__init__.py",
            submodule_search_locations=[str(root_path / _PACKAGE)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load torch_fl from {root_path / _PACKAGE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_PACKAGE, None)
            raise
        loaded = sys.modules[_PACKAGE]
        if not _contains_installed_module(_PACKAGE, loaded):
            raise ImportError(
                f"torch_fl was imported from {getattr(loaded, '__file__', '')} "
                f"instead of the installed package in {root}"
            )
    return Path(root) / _PACKAGE


def _finder_for_stub():
    """Let an explicit stub installed by a test find its own modules."""
    for entry in sys.path[:4]:
        if not entry:
            continue
        package = Path(entry) / _PACKAGE
        if not (package / "__init__.py").is_file():
            continue
        try:
            if not package.resolve().is_relative_to(Path(_installed_root()).resolve()):
                return "stub"
        except (OSError, RuntimeError, ValueError):
            continue
    return None


class _InstalledTorchFlFinder:
    """Import the wheel's package even after pytest prepends the checkout."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        selected = _finder_for_stub()
        if selected == "stub":
            return None
        if name == _PACKAGE:
            return importlib.machinery.PathFinder.find_spec(
                name, [_installed_root()], target
            )
        if name.startswith(_PACKAGE + "."):
            loaded = sys.modules.get(name.split(".", 1)[0])
            if loaded is not None and not _contains_installed_module(_PACKAGE, loaded):
                return None
            package = Path(_installed_root()) / _PACKAGE
            return importlib.machinery.PathFinder.find_spec(
                name,
                [str(package)],
                target,
            )
        return None


def _install_import_hook():
    """Install both the installed-package and explicit-stub selectors."""
    global _PATH_HOOK_INSTALLED
    if _PATH_HOOK_INSTALLED:
        return

    finder = _InstalledTorchFlFinder()
    finder._flagos_torch_fl = True
    sys.meta_path.insert(0, finder)
    if not any(getattr(item, _SELECTION_HOOK_TAG, False) for item in sys.meta_path):
        selection = importlib.machinery.PathFinder
        selection._flagos_torch_fl_host = True
    _PATH_HOOK_INSTALLED = True
