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
Let inductor discard a Triton config the vendor compiler cannot fit on chip.

Inductor precompiles several autotune candidates per kernel and expects some to
be too big for the hardware: `CachingAutotuner._precompile_worker` catches
``OutOfResources`` per config, keeps the survivors, and only fails if *none*
compiled (`torch/_inductor/runtime/triton_heuristics.py`). NVIDIA reaches that
path because ptxas reports excess shared memory as ``OutOfResources``.

triton-ascend reports the equivalent condition -- the tile does not fit in Unified
Buffer -- as a generic ``MLIRCompilationError`` out of the BiShengHIR pipeline::

    error: ub overflow, requires 4194816 bits while 1572864 bits available!

That type is not in the tolerated set, so the first oversized candidate aborts the
whole compile even when smaller ones would have worked. Measured on a 910 for the
persistent reduction in a `Linear -> ReLU -> Linear` backward (x=128, r0_=32,
inductor tries XBLOCK 1, 8, 32, 128): XBLOCK 128 needs 512 KB of UB and 64 needs
256 KB against 192 KB available, while 32 and below compile and run. Three usable
configs were being thrown away by the one that overflowed.

So this narrows the vendor error to the one condition it describes and re-raises
it as ``OutOfResources``, which is what it is -- required and available are in the
message. Inductor then does exactly what it does on NVIDIA: drop that config,
autotune over the rest. Anything else from the compiler propagates untouched; a
kernel whose smallest config still overflows still fails, and reports UB numbers.
"""

import re
from typing import Any


_PATCH_FLAG = "_flagos_resource_limit_translation"

# "ub overflow, requires 4194816 bits while 1572864 bits available!" -- the
# BiShengHIR wording for a tile that exceeds Unified Buffer. Matched narrowly on
# purpose: a message this specific is a tile-size problem, and everything else
# from the MLIR pipeline is a real compile error that must keep its own type.
_UB_OVERFLOW = re.compile(
    r"(\w+) overflow, requires (\d+) bits while (\d+) bits available"
)


def patch_triton_resource_limit_errors() -> None:
    """Translate vendor tile-too-large errors into Triton's ``OutOfResources``.

    Only for non-CUDA-like builds, where the vendor compiler does not raise
    Triton's own resource error. Idempotent.
    """
    from torch_fl.compile.platform_profile import platform_profile

    if platform_profile().is_cuda_like:
        return

    try:
        import triton
        from triton.runtime.errors import OutOfResources
    except ImportError:
        return

    # Flag on the module, not on the function: triton_byte_loads wraps the same
    # function, so a flag on the wrapper is invisible once the other patch sits on
    # top, and registration runs before every compile_fx.
    if getattr(triton, _PATCH_FLAG, False):
        return

    original_compile = triton.compile

    def compile(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_compile(*args, **kwargs)
        except Exception as exc:
            limit = _resource_limit(exc)
            if limit is None:
                raise
            name, required, available = limit
            raise OutOfResources(required, available, f"{name} (bits)") from exc

    triton.compile = compile
    setattr(triton, _PATCH_FLAG, True)


def _resource_limit(exc: BaseException) -> Any:
    """``(name, required, available)`` if ``exc`` is an on-chip capacity error.

    None for anything else, including a resource error nested inside an
    unrelated failure -- the message is only trusted when the compiler raised it
    as the error itself.
    """
    match = _UB_OVERFLOW.search(str(exc))
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))
