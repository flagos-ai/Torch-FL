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

"""The `Backend` enum names routes, not runtimes.

`Backend::kCuda` used to name the boxed slot while MetaX, Hygon DCU and the
T-Head PPU all dispatched through it, so the name said "NVIDIA" about a route
three other vendors also take. It is `kBoxing` now: the route re-dispatches
PrivateUse1 to CUDA through whatever vendor libtorch the wheel bundles.

A reintroduction is cheap (one codegen line) and silent (the enum compiles and
the slot works), so it is pinned here: no tracked source may carry
`Backend::kCuda`, the generator must emit the new name, and the conf spelling
"cuda" must stay accepted as an alias because confs and `FLAGOS_OP_*` overrides
in the field still use it.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SOURCES = (
    "csrc/aten/common.h",
    "csrc/aten/common.cc",
    "csrc/aten/dispatcher.h",
    "csrc/aten/generated/cuda_kernels.cc",
    "csrc/aten/backends/soft_lowp/ops.cc",
    "scripts/codegen/codegen_ops.py",
)


def test_no_source_names_the_boxed_slot_kcuda():
    offenders = [
        str(path)
        for rel in _SOURCES
        if (path := REPO_ROOT / rel).exists()
        and re.search(r"\bkCuda\b", path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"Backend::kCuda is back in {offenders}; the slot is Backend::kBoxing "
        "(the CUDA-ABI platforms all route through it, so naming it after the "
        "runtime misdescribes it). Rename, do not re-add."
    )


def test_cuda_codegen_emits_the_boxing_slot():
    generated = (REPO_ROOT / "csrc/aten/generated/cuda_kernels.cc").read_text(
        encoding="utf-8"
    )
    registered = re.findall(
        r"REGISTER_IMPL_TO_DISPATCHER\([^)]*Backend::(\w+)", generated
    )
    assert registered, "no REGISTER_IMPL_TO_DISPATCHER lines in cuda_kernels.cc"
    assert set(registered) == {"kBoxing"}, sorted(set(registered))
    generator = (REPO_ROOT / "scripts/codegen/codegen_ops.py").read_text(
        encoding="utf-8"
    )
    assert "Backend::kBoxing" in generator, (
        "codegen_ops.py does not emit Backend::kBoxing, so the next regeneration "
        "would put kCuda back"
    )


def test_cuda_spelling_stays_an_accepted_alias():
    """Field confs and FLAGOS_OP_<name>=cuda predate the rename."""
    common = (REPO_ROOT / "csrc/aten/common.cc").read_text(encoding="utf-8")
    assert "Backend::kCuda" not in common
    parse = common[common.index("bool ParseBackendName") :]
    parse = parse[: parse.index("const char* BackendName")]
    assert '"cuda"' in parse and '"boxing"' in parse, (
        "ParseBackendName must accept both 'boxing' (canonical) and 'cuda' "
        "(legacy alias)"
    )
    # The canonical name is what BackendName reports back.
    name_fn = common[common.index("const char* BackendName") :]
    name_fn = name_fn[: name_fn.index("// Parse one conf file")]
    assert 'case Backend::kBoxing:       return "boxing";' in name_fn
