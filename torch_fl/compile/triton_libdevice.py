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
Make ``libdevice.*`` in an inductor-generated kernel reach a real implementation.

Inductor emits transcendentals as ``libdevice.rsqrt(x)`` and friends
(`torch/_inductor/codegen/triton.py`), having imported ``libdevice`` from
``triton.language.extra`` -- the *generic* module, whose 197 functions are all
bare ``...`` stubs. Triton's design is that each vendor backend substitutes its
own implementation module for that name via ``CompilerBackend.get_module_map()``,
which the AST walker consults for every module-typed global
(`triton/compiler/code_generator.py`, `self.gscope[k] = module_map.get(...)`).

triton-ascend ships the implementations (``triton.language.extra.ascend.libdevice``,
46 functions) but returns ``{}`` from ``AscendBackend.get_module_map()``, so
nothing connects the two. Tracing a generated kernel then walks into a stub and
fails inside Triton's own semantics layer, where the diagnostic no longer names
the op::

    tmp21 = libdevice.rsqrt(tmp20)
            ^
    AttributeError("'NoneType' object has no attribute 'type'")

Filling the map in is the whole fix. It is what the NVIDIA backend does, and it
is per-backend state, so nothing outside Triton's Ascend compile path changes.
FlagGems' handwritten kernels already import the ascend module directly and are
unaffected either way.

Coverage is Ascend's, not ours: of the 41 names inductor can emit, 38 have an
Ascend implementation. ``erfc``, ``llrint`` and ``fast_tanhf`` have none -- not in
``triton.language`` or ``triton.language.math`` either -- so a graph needing one
still fails, now with an ``AttributeError`` naming the missing function instead of
a stub's internals. That is the honest report of a toolchain gap; substituting an
approximation would be a silent accuracy change.
"""

from typing import Any, Dict


_MODULE_MAP_FLAG = "_flagos_libdevice_module_map"


def _generic_libdevice_name() -> str:
    """Module name inductor's generated kernels bind ``libdevice`` to."""
    from triton.language.extra import libdevice

    return libdevice.__name__


def patch_triton_libdevice_module_map() -> None:
    """Route the generic ``libdevice`` module to the vendor implementation.

    Only for non-CUDA-like builds, and only when the vendor backend has left its
    module map empty -- a backend that already fills it knows better than we do.
    Idempotent: the flag is set on the backend class it patched.
    """
    from torch_fl.compile.platform_profile import platform_profile

    profile = platform_profile()
    if profile.is_cuda_like:
        return

    try:
        import triton
    except ImportError:
        return

    backend = triton.backends.backends.get(profile.triton_backend_key)
    if backend is None:
        return

    compiler = backend.compiler
    if getattr(compiler, _MODULE_MAP_FLAG, False):
        return

    vendor = _vendor_libdevice(profile.triton_backend_key)
    if vendor is None:
        return

    original_get_module_map = compiler.get_module_map
    generic_name = _generic_libdevice_name()

    def get_module_map(self: Any) -> Dict[str, Any]:
        module_map = dict(original_get_module_map(self))
        module_map.setdefault(generic_name, vendor)
        return module_map

    compiler.get_module_map = get_module_map
    setattr(compiler, _MODULE_MAP_FLAG, True)


def _vendor_libdevice(backend_key: str) -> Any:
    """The backend's own libdevice module, or None if it ships none.

    Vendor backends that provide implementations put them in
    ``triton.language.extra.<key>.libdevice``, mirroring upstream's
    ``extra.cuda.libdevice``.
    """
    import importlib

    try:
        return importlib.import_module(f"triton.language.extra.{backend_key}.libdevice")
    except ImportError:
        return None
