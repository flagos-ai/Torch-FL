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
Refuse an Inductor kernel the GCU300 target cannot compile, instead of crashing.

The GCU300 target has no 64-bit support at all, and both ways of reaching it
end badly:

* A 64-bit *element type* in the kernel signature (an int64 or float64 tensor)
  is rejected by the compiler, which surfaces as an ``InductorError`` wrapping
  ``RuntimeError: Pipeline run failed: PassManager execution failed``. The
  interpreter survives.
* A 64-bit *promotion of size arithmetic* keeps an int32 signature -- the i64
  only reaches ``tl.full(..., tl.int64)`` constants and the ``arith.extsi``
  feeding them -- and there the vendor pass manager does not fail politely.
  ``arith.extsi`` is marked illegal in the GCU pipeline, and legalizing it
  aborts the process: SIGSEGV, exit code 139, no traceback.

The second one is why this guard has to exist, and why it sits where it does.
The crash happens inside ``triton.compile``, i.e. in the vendor MLIR pass
manager; Inductor's own ``try/except`` around that call in
``CachingAutotuner._precompile_config`` (torch/_inductor/runtime/
triton_heuristics.py) is *after* the crash point, so nothing on Inductor's
exception path ever runs and no Python-level handler can see it. The only place
a guard can stand is before ``triton.compile`` is entered, and
``_precompile_config`` is that place.

Inductor reaches the promotion on its own, without any int64 input:
``SIMDScheduling.select_index_dtype`` falls back to ``torch.int64`` whenever
``can_use_32bit_indexing`` declines the buffer set, and every size-derived value
(``tl.full([[XBLOCK, R0_BLOCK]], 0, tl.int64)`` and friends) then carries the
kernel's index dtype. No Inductor config knob selects the index dtype, so this
cannot be fixed by a config patch. Flex attention's ``create_block_mask``
reduction is a real instance: it is int32 throughout and still crashes.

So the predicate below scans the generated source, plus the signature, for a
64-bit token, and raises an ``InductorError`` naming the kernel before the
compiler is invoked. The failure becomes a diagnosable Python exception that
``FLAGOS_COMPILE_FALLBACK_EAGER=1`` can turn into an eager run.

What this does *not* fix: the vendor pass manager still aborts the process on a
64-bit kernel it is handed directly (someone calling the flagged Triton code
themselves, or a build where this guard is not installed). Turning that fault
into a raised exception belongs on the vendor side.
"""

import re
from typing import Any, List, Optional, Tuple

_PATCH_FLAG = "_flagos_gcu_64bit_guard"

# "int64", "uint64", "float64", "fp64" and the Triton spellings "tl.int64",
# "tl.uint64", "tl.float64". Matched as whole words so an identifier that merely
# contains the digits cannot trip it.
_WIDE_TYPE = re.compile(r"\b(?:u?int64|float64|fp64)\b")

# Signature entries Inductor writes for a 64-bit operand: "*i64", "i64", "*u64",
# "*fp64", ... This catches the element-type case directly -- the generated
# source of a pointwise int64 kernel need not mention a type at all.
_WIDE_SIGNATURE = re.compile(r"^\*?(?:i64|u64|fp64)$")

_ISSUE = "#400"

# How much of each offending source line to keep in the message.
_SNIPPET = 120


def gcu_64bit_unsupported() -> bool:
    """True when the active build targets a device without 64-bit kernel support.

    A public helper for callers that can avoid the kernel instead of being
    refused by it -- ``torch.nn.attention.flex_attention.create_block_mask``
    generates the crashing reduction on any input dtype, so a GCU caller can
    pass ``_compile=False`` (or wrap the factory in ``torch._dynamo.disable``)
    rather than pay a failed compile.
    """
    from torch_fl.compile.platform_profile import platform_profile

    return platform_profile().vendor == "gcu"


def patch_triton_64bit_guard() -> None:
    """Raise instead of segfaulting when a kernel needs 64-bit values on GCU.

    Only for GCU builds: on CUDA-like targets 64-bit kernels compile and this
    patch changes nothing. Idempotent.
    """
    if not gcu_64bit_unsupported():
        return

    try:
        from torch._inductor.runtime import triton_heuristics
    except ImportError:
        return

    # Flag on the module, not on the function: triton_byte_loads and the
    # resource-limit patch both wrap shared entry points, so a flag stored on
    # the function object can end up under another wrapper and read as absent.
    if getattr(triton_heuristics, _PATCH_FLAG, False):
        return

    original_precompile_config = triton_heuristics.CachingAutotuner._precompile_config

    def _precompile_config(self: Any, cfg: Any) -> Any:
        wide = wide_uses(
            getattr(self.fn, "src", None),
            (self.inductor_meta or {}).get("signature"),
        )
        if wide:
            raise _unsupported_error(
                (self.inductor_meta or {}).get("kernel_name"), wide
            )
        return original_precompile_config(self, cfg)

    triton_heuristics.CachingAutotuner._precompile_config = _precompile_config
    setattr(triton_heuristics, _PATCH_FLAG, True)


def wide_uses(src: Optional[str], signature: Optional[Any]) -> List[Tuple[str, Any]]:
    """64-bit uses of a generated kernel, as ``(where, detail)`` pairs.

    ``where`` is ``"signature"`` for a 64-bit operand type and ``"line N"`` for
    a source line naming a 64-bit type. Empty when the kernel is 32-bit
    throughout, which is the case the guard must not touch.

    Split out from the patch so the predicate can be tested without the vendor
    compiler -- it is a pure function of what Inductor generated.
    """
    found: List[Tuple[str, Any]] = []

    if signature:
        items = signature.items() if hasattr(signature, "items") else signature
        for name, value in items:
            if isinstance(value, str) and _WIDE_SIGNATURE.match(value):
                found.append(("signature", f"{name}: {value}"))

    for number, line in enumerate((src or "").splitlines(), 1):
        if _WIDE_TYPE.search(line):
            found.append((f"line {number}", line.strip()[:_SNIPPET]))

    return found


def _unsupported_error(kernel_name: Optional[str], wide: List[Tuple[str, Any]]) -> Any:
    """The ``InductorError`` a refused kernel raises, with its own diagnosis.

    Typed like the compile failure it replaces, so a caller already catching
    ``InductorError`` -- or ``Exception`` -- keeps working across this change.
    """
    from torch._inductor.exc import InductorError

    name = kernel_name or "<unnamed triton kernel>"
    detail = "\n".join(f"  {where}: {what}" for where, what in wide[:8])
    if len(wide) > 8:
        detail += f"\n  ... and {len(wide) - 8} more"

    message = (
        f"The GCU300 target has no 64-bit support, and kernel '{name}' needs "
        f"64-bit values:\n{detail}\n\n"
        "The vendor compiler cannot legalize these and aborts the pass pipeline "
        "-- for a promoted int64 index that is a SIGSEGV that kills the "
        "interpreter, which is why the kernel was refused before compilation.\n"
        "To run it anyway:\n"
        "  1. Keep the values 32-bit: avoid int64/float64 tensors and "
        "size-derived indices this large.\n"
        "  2. Skip compiling the kernel that generates it. Flex attention's "
        "create_block_mask does this on any input dtype, so pass "
        "_compile=False or wrap the factory in torch._dynamo.disable.\n"
        "  3. Set FLAGOS_COMPILE_FALLBACK_EAGER=1 to run the model eagerly "
        "instead of failing the compile.\n"
        f"See issue {_ISSUE} for the current state of 64-bit support."
    )
    return InductorError(RuntimeError(message), None)
