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

"""FlagGems' ``ext`` alias must reach the kernels Inductor re-emits.

FlagGems kernels use ``ext.program_id(axis)`` after a module-level
``from flag_gems.utils import triton_lang_extension as ext``. Dynamo captures
such a kernel, and Inductor re-emits its source into a wrapper that imports
only ``triton``/``tl``, so the alias is undefined at compile time:

    NameError('ext is not defined')

These cases pin the prelude patch that carries the import.
"""

import pytest

import torch_fl  # noqa: F401  -- must precede torch: relinks the vendor libtorch

pytest.importorskip("flag_gems")

from torch._inductor.codegen import triton as triton_codegen

from torch_fl.compile import triton_flaggems_kernel_imports

EXT_IMPORT = "from flag_gems.utils import triton_lang_extension as ext"


def _unpatched_prelude():
    """The prelude generator before this module wrapped it."""
    generator = triton_codegen.gen_common_triton_imports
    while hasattr(generator, "__wrapped__"):
        generator = generator.__wrapped__
    return generator


def test_prelude_carries_the_ext_import(monkeypatch):
    monkeypatch.setattr(
        triton_codegen, "gen_common_triton_imports", _unpatched_prelude()
    )

    triton_flaggems_kernel_imports.patch_flaggems_kernel_prelude()

    assert EXT_IMPORT in triton_codegen.gen_common_triton_imports()


def test_patch_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        triton_codegen, "gen_common_triton_imports", _unpatched_prelude()
    )

    triton_flaggems_kernel_imports.patch_flaggems_kernel_prelude()
    patched = triton_codegen.gen_common_triton_imports
    triton_flaggems_kernel_imports.patch_flaggems_kernel_prelude()

    assert triton_codegen.gen_common_triton_imports is patched
