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

"""The recognized ``GEMS_VENDOR`` set and the fail-loud explicit-value check.

``torch_fl/__init__.py`` resolves FlagGems' vendor through a ladder of branches
(``_patch_flaggems_codegen_config``). Two failure modes used to be silent:

* an explicitly exported ``GEMS_VENDOR`` value torch_fl cannot configure was
  left in place (``_env.set_foreign`` never overrides an explicit export) and
  handed to FlagGems and the comm layer, whose profiles did not know it either;
* when no branch matched -- a CUDA wheel on a host with no reachable GPU, a
  MetaX wheel with no card -- the ladder's final ``else`` selected ``ascend``,
  so the wrong vendor's codegen config was used and the failure surfaced far
  from its cause.

This module is the single chokepoint for the first: the recognized set and the
validation of an explicitly requested value live here, so the ladder can only
proceed with a value it can actually configure. The second is handled in
``__init__.py`` next to the ladder, which now raises instead of defaulting.

Stdlib-only by design: tests load it by path, the way they load ``_env.py``, so
they need no torch install and never execute ``torch_fl/__init__.py``.
"""

from __future__ import annotations

#: Vendors torch_fl and its comm layer (``comm/process_group._VENDOR_PROFILES``)
#: can route. Not the same as "every value FlagGems understands": an explicit
#: value outside this set is one torch_fl would silently mismatch, so it is
#: rejected rather than passed through. Kept in sync with the comm profiles by
#: tests/unit/test_vendor_resolution.py.
KNOWN_VENDORS = frozenset(
    {
        "nvidia",
        "cuda",
        "metax",
        "hygon",
        "ascend",
        "mthreads",
        "musa",
        "enflame",
        "iluvatar",
        "kunlunxin",
        "du",
        "thead",
        "cambricon",
    }
)


def validate_explicit(value: str | None) -> str | None:
    """Normalize an explicit ``GEMS_VENDOR``, or raise if it is not recognized.

    Returns the lowercased, stripped value, or ``None`` when it is unset or
    blank (an empty export is how a user says "not this vendor"). Raises
    ``RuntimeError`` for a value torch_fl cannot configure, naming the set it
    can, so the mistake surfaces at ``import torch_fl`` instead of as a
    wrong-vendor failure inside a kernel.
    """
    if value is None:
        return None
    name = value.strip().lower()
    if not name:
        return None
    if name not in KNOWN_VENDORS:
        raise RuntimeError(
            f"GEMS_VENDOR={value!r} not recognized; valid: "
            f"{sorted(KNOWN_VENDORS)}; unset it to let torch_fl detect the vendor"
        )
    return name
