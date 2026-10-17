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
Declare boolean tensor arguments as ``*i8`` rather than ``*i1`` on Ascend.

**This is a correctness workaround, not a feature.** A masked, strided 2-D load
of 8-bit data goes through a wrong path in the Ascend Triton codegen when the
pointer is declared ``*i1``: the kernel runs, no error is raised, and the numbers
are simply wrong. Reading the identical bytes through an ``*i8`` pointer returns
correct data.

This matters because inductor emits exactly that shape for any reduction over a
boolean mask, which is what `relu` backward is: `threshold_backward` loads the
`le` mask as ``*i1`` and reduces over it. So `torch.compile` of a plain
`Linear -> ReLU` produced silently wrong gradients rather than failing -- there
is no error to catch, only a gradient that disagrees with eager.

Measured twice, on the two Ascend CI runs that bracket our own edits:

* On triton-ascend 3.2.0 the same miscompile produced 4090 of 4096 gradient
  elements wrong, max abs difference 3.05.
* On FlagTree ascend3.5, with this rewrite removed and nothing else changed, the
  Ascend compile job of run `38763731890` (job `116522683281`) fails
  `test_ascend_compile_backward_stays_on_flagos` with 4094 of 4096 elements wrong
  and max abs difference 3.824, and
  `test_ascend_oversized_config_does_not_fail_compile` with `-inf` at every
  element. Both are the `Linear -> ReLU` backward above.

The other half of the original workaround -- `enable_linearize=True` in the
vendor compile options -- is gone, and should stay gone: FlagTree ascend3.5
returns correct data for the same access pattern with no options set, and
`enable_linearize` no longer exists in `NPUOptions`, so `parse_options` had been
dropping it. The signature rewrite is the half that is still load-bearing, which
is why it survives on its own here.

The rewrite is sound because torch stores `bool` as one byte and inductor's
generated code already assumes a byte load: it appends ``.to(tl.int1)`` after
loading from an int1 pointer, with the comment "tl.load returns int8 when loading
from pointer to int1" (`torch/_inductor/codegen/triton.py`). Only the *declared*
pointer type changes; the loaded value and every use of it are untouched.

It is scoped to non-CUDA-like builds, as it was before, and a vendor fix makes it
dead rather than wrong: `test_ascend_bool_tensors_declared_as_byte_pointers`
asserts the rewrite is installed, and `test_ascend_compile_backward_stays_on_flagos`
is the end-to-end guard that fails with wrong numbers if the rewrite is dropped
while the miscompile is still there.
"""

from typing import Any

from torch_fl.compile.platform_profile import platform_profile


_SIGNATURE_FLAG = "_flagos_bool_as_i8"


def patch_triton_bool_pointer_signatures() -> None:
    """Rewrite ``*i1`` kernel-argument pointers to ``*i8``. Idempotent."""
    if platform_profile().is_cuda_like:
        return

    _patch_bool_signature()


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
