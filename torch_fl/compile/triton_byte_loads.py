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
Work around a triton-ascend miscompile of masked 2-D loads of byte dtypes.

**This is a correctness workaround, not a feature.** triton-ascend 3.2.0 silently
returns wrong data for a masked, strided 2-D load of an 8-bit dtype: only the
first two elements of each row are fetched, then repeated across the row. No
error is raised, so the kernel runs and the numbers are simply wrong.

Reduced on a 910 (`Ascend910_9382`, CANN 9.0.0), reading an int8 `[32, 128]`
laid out row-major and indexed `xindex[:, None] + 128 * r[None, :]`::

    expected row0[:16]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
    got      row0[:16]: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  0, ...]

The trigger is all three of: 8-bit element type, a 2-D strided index, and a mask
on the load. Dropping the mask, using a contiguous index, or widening to int16 or
wider all return correct data. The mask value is irrelevant -- the reduction above
has an all-true mask. `num_warps`, `num_stages`, `multibuffer`,
`enable_select_analysis`, `enable_nd2nz_on_vector`, `enable_persistent`,
`optimize_epilogue` and `parallel_mode` make no difference.

This matters because inductor emits exactly that shape for any reduction over a
boolean mask, which is what `relu` backward is: `threshold_backward` loads the
`le` mask as `*i1` and reduces over it. So `torch.compile` of a plain
`Linear -> ReLU` produced silently wrong gradients (measured: 4090 of 4096
elements wrong, max abs error 3.05) rather than failing.

Two fixes are needed, and both are required -- either alone still returns wrong
data:

1. ``enable_linearize=True`` in the vendor compile options. This fixes the 8-bit
   case for ``*i8`` pointers. Verified not to change results for float kernels or
   for ragged shapes where the mask actually masks.
2. Declaring boolean tensors as ``*i8`` instead of ``*i1`` in the kernel
   signature. ``enable_linearize`` does *not* fix the ``*i1`` pointer type, but the
   identical bytes read through an ``*i8`` pointer are correct. This is sound
   because torch stores `bool` as one byte, and inductor's generated code already
   assumes a byte load: it appends ``.to(tl.int1)`` after loading from an int1
   pointer, with the comment "tl.load returns int8 when loading from pointer to
   int1" (`torch/_inductor/codegen/triton.py`). Only the *declared* pointer type
   changes; the loaded value and every use of it are untouched.

Both are scoped to non-CUDA-like builds. Remove them once triton-ascend fixes the
underlying miscompile; the reduction above is the regression test.
"""

from typing import Any

from torch_fl.compile.platform_profile import platform_profile


_SIGNATURE_FLAG = "_flagos_bool_as_i8"
_OPTIONS_FLAG = "_flagos_enable_linearize"


def patch_triton_byte_load_workarounds() -> None:
    """Apply both halves of the masked byte-load workaround. Idempotent."""
    if platform_profile().is_cuda_like:
        return

    _patch_bool_signature()
    _patch_linearize_option()


def _patch_bool_signature() -> None:
    """Declare boolean tensor args as ``*i8`` rather than ``*i1``.

    Patched at `triton_utils.signature_of`, which every signature path funnels
    through (`signature_to_meta` looks it up in module globals at call time), so
    the generated `triton_meta` and the signature handed to `triton.compile`
    change together and cannot disagree.

    Only the pointer type ``*i1`` is rewritten. A bare ``i1`` is a scalar bool
    `SizeArg`, not a pointer, and is left alone.
    """
    from torch._inductor.codegen import triton_utils

    if getattr(triton_utils.signature_of, _SIGNATURE_FLAG, False):
        return

    original_signature_of = triton_utils.signature_of

    def signature_of(arg: Any, *, size_dtype: Any) -> str:
        signature = original_signature_of(arg, size_dtype=size_dtype)
        return "*i8" if signature == "*i1" else signature

    setattr(signature_of, _SIGNATURE_FLAG, True)
    triton_utils.signature_of = signature_of


def _patch_linearize_option() -> None:
    """Add ``enable_linearize=True`` to the vendor compile options.

    Inductor builds its option dict in `CachingAutotuner._create_compile_options`
    and only ever adds vendor keys for cuda and hip, so there is no inductor-side
    hook for an Ascend-only option; this wraps `triton.compile` instead. An
    explicit caller-supplied value wins, so a kernel can still opt out.

    The idempotency flag lives on the `triton` module rather than on the wrapper,
    because `triton_resource_limits` wraps the same function: a flag on the
    function object is invisible once the other patch has wrapped over it, and
    registration runs before every compile_fx, so the wrappers would nest without
    bound.
    """
    import triton

    if getattr(triton, _OPTIONS_FLAG, False):
        return

    original_compile = triton.compile

    def compile(*args: Any, **kwargs: Any) -> Any:
        options = dict(kwargs.get("options") or {})
        options.setdefault("enable_linearize", True)
        kwargs["options"] = options
        return original_compile(*args, **kwargs)

    triton.compile = compile
    setattr(triton, _OPTIONS_FLAG, True)
