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

"""Carry FlagGems' ``ext`` alias into the kernels Inductor re-emits.

FlagGems kernels call ``ext.program_id(axis)`` and friends, where ``ext`` is a
module-level import (``from flag_gems.utils import triton_lang_extension as
ext``) in 390 of their files. When ``torch.compile`` traces one of those
kernels, Dynamo captures it and Inductor re-emits its source into a wrapper
that imports only ``triton`` and ``tl``. The module-level ``ext`` import is not
carried over, so compilation fails at the first use::

    def repeat_interleave_tensor_kernel(...):
        pid = ext.program_id(0)
              ^
    NameError('ext is not defined')

Adding the import to Inductor's common Triton prelude fixes every such kernel at
once. ``ext.program_id`` is ``tl.program_id(axis).to(tl.int64)``; the other
helpers in that module are equally small wrappers, so importing the module is
what the kernel meant in the first place.

The patch is inert when FlagGems is not installed, and idempotent across
recompiles.
"""

from typing import Any


_PRELUDE_FLAG = "_flagos_flaggems_ext_prelude"

_EXT_IMPORT = "\nfrom flag_gems.utils import triton_lang_extension as ext\n"

# Modules that bind the generator by value at import time; patching only the
# defining module would leave these pointing at the original function.
_BY_VALUE_IMPORTERS = (
    "torch._inductor.codegen.triton_combo_kernel",
    "torch._inductor.select_algorithm",
)


def patch_flaggems_kernel_prelude() -> None:
    """Append the ``ext`` import to Inductor's common Triton prelude."""
    import importlib
    import importlib.util
    import functools

    if importlib.util.find_spec("flag_gems") is None:
        return

    try:
        from torch._inductor.codegen import triton as triton_codegen
    except ImportError:
        return

    original = triton_codegen.gen_common_triton_imports
    if getattr(original, _PRELUDE_FLAG, False):
        return

    @functools.lru_cache(None)
    def gen_common_triton_imports() -> str:
        return original() + _EXT_IMPORT

    setattr(gen_common_triton_imports, _PRELUDE_FLAG, True)
    triton_codegen.gen_common_triton_imports = gen_common_triton_imports

    for name in _BY_VALUE_IMPORTERS:
        try:
            module: Any = importlib.import_module(name)
        except ImportError:
            continue
        if getattr(module, "gen_common_triton_imports", None) is original:
            module.gen_common_triton_imports = gen_common_triton_imports
