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

"""The software low-precision prelude's contract, checked without a device.

The generated wrappers open with a prelude that returns FP8/FP4 matrix calls
into ``csrc/aten/backends/soft_lowp`` on a ``FLAGOS_SOFT_LOWP`` build. The
prelude cannot tell an FP8 operand from an ordinary one, so the software path
receives every mm/addmm/bmm on such a build and computes the ordinary ones
itself (the decode is a no-op). Two things must therefore stay true, and both
are properties of the source rather than of a run:

* the software path must not hand a call back to its own dispatcher by name --
  the generated wrapper is the only place the dispatch log names ``bmm``, and a
  by-name re-entry from inside the path is what segfaulted on hardware; and
* the wrapper's prelude must be off while the software path dispatches into
  another backend, or (``FLAGOS_FORCE_BACKEND=flaggems``) a fused ``addmm`` sent
  to its flaggems arm would come straight back here.

Usage:
    pytest tests/unit/test_soft_lowp_prelude_contract.py -v
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOFT_LOWP = REPO_ROOT / "csrc" / "aten" / "backends" / "soft_lowp" / "ops.cc"
DISPATCHER = REPO_ROOT / "csrc" / "aten" / "dispatcher.h"


def test_soft_lowp_does_not_redispatch_through_the_generated_wrappers():
    """No by-name re-entry: the software path either computes or throws.

    A re-entry (``bmm_dispatcher.DispatchAs(...)`` from BmmImpl, say) lands in
    the generated wrapper, which runs the prelude again unless the path has
    raised the suppression guard -- and the guard only wraps the fused addmm.
    The MetaX wheel with the prelude compiled segfaulted on exactly that shape
    of call (CI run 40195692591), so the helper is gone rather than guarded.
    """
    source = SOFT_LOWP.read_text(encoding="utf-8")
    reentries = re.findall(r"\b(\w+_dispatcher)\.DispatchConfigured\(", source)
    assert not reentries, f"soft_lowp re-dispatches by name: {reentries}"


def test_the_prelude_has_a_suppression_guard():
    """The fused addmm's FlagGems call is not re-captured by the prelude."""
    source = SOFT_LOWP.read_text(encoding="utf-8")
    addmm_impl = source.split("at::Tensor AddmmImpl(", 1)[1].split("\n}\n", 1)[0]
    assert "DispatchSuppressionGuard" in addmm_impl, (
        "AddmmImpl dispatches into another backend without raising the "
        "suppression guard; the wrapper's prelude would send the call back in"
    )
    dispatcher = DISPATCHER.read_text(encoding="utf-8")
    assert "const char* OpName() const" in dispatcher


def test_the_software_path_accepts_ordinary_dtypes():
    """The prelude sends every dtype here, so the lowp-only check must be gone."""
    source = SOFT_LOWP.read_text(encoding="utf-8")
    assert "requires a low-precision input" not in source, (
        "soft_lowp still rejects ordinary dtypes, but the generated prelude "
        "sends them here on every FLAGOS_SOFT_LOWP build"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
