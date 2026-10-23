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

"""A conf route key and a Backend enum value name the same vendor.

The enum carried `kMusa`, `kGcu` and `kMps` -- the SDK each was first built
against -- while the confs had already moved to `mthreads`, `enflame` and
`apple`. A reader had to translate between the two, and nothing failed when
they disagreed because both spellings resolved to the right slot.

`ParseBackendName` still accepts the older spelling (an out-of-tree conf or a
`FLAGOS_OP_*` override may carry it), which is exactly why the *emitted* names
are pinned here: the aliases make drift silent.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_CC = REPO_ROOT / "csrc" / "aten" / "common.cc"
COMMON_H = REPO_ROOT / "csrc" / "aten" / "common.h"

#: The values a conf may route to that are not a vendor's kernel.
#: `boxing` is the day-0 files' spelling of the same kernel: those are not
#: vendor confs (they exist only to box), so they name the route, not a vendor.
NON_VENDOR_KEYS = {
    "flaggems",
    "flaggems_cpp",
    "flagos_python",
    "tileops",
    "boxing",
    "none",
}


def _platforms() -> dict:
    return json.loads(
        (REPO_ROOT / "cmake" / "flagos_platforms.json").read_text(encoding="utf-8")
    )["accelerators"]


def _backend_name_table() -> dict:
    """`BackendName()`: enumerator -> the string it reports."""
    text = COMMON_CC.read_text(encoding="utf-8")
    body = text[text.index("const char* BackendName") :]
    body = body[: body.index("// Parse one conf file")]
    return dict(re.findall(r'case Backend::(\w+):\s+return "([a-z_]+)";', body))


def test_every_vendor_has_a_backend_enumerator_named_after_it():
    enum = COMMON_H.read_text(encoding="utf-8")
    enum = enum[enum.index("enum class Backend {") : enum.index("};")]
    enumerators = set(re.findall(r"^\s+(k\w+),", enum, flags=re.M))
    names = _backend_name_table()
    for vendor in sorted({row["vendor"] for row in _platforms().values()}):
        expected = "k" + vendor[0].upper() + vendor[1:]
        if expected in enumerators:
            assert names.get(expected) == vendor, (
                f"{expected} reports {names.get(expected)!r}, not {vendor!r}"
            )
        # A vendor with no kernel of its own (nvidia here routes through kBoxing;
        # hygon and thead have no slot) is not expected to have one.


def test_no_enumerator_still_names_an_sdk():
    """kMusa / kGcu / kMps were the SDK spellings; the vendor name replaced them."""
    enum = COMMON_H.read_text(encoding="utf-8")
    enum = enum[enum.index("enum class Backend {") : enum.index("};")]
    for dead in ("kMusa", "kGcu", "kMps"):
        assert dead not in enum, f"{dead} is back; the enum names vendors (AGENTS.md)"
    names = _backend_name_table()
    for value in names.values():
        assert value not in {"gcu", "musa", "mps"}, (
            f"BackendName still reports the SDK spelling {value!r}"
        )


def test_shipped_confs_route_to_vendor_keys_only():
    vendors = {row["vendor"] for row in _platforms().values()}
    allowed = vendors | NON_VENDOR_KEYS
    offenders = {}
    for conf in sorted((REPO_ROOT / "torch_fl" / "configs").glob("backends_*.conf")):
        for line in conf.read_text(encoding="utf-8").splitlines():
            body = line.split("#", 1)[0].strip()
            if "=" not in body:
                continue
            _op, _, value = body.partition("=")
            key = value.strip()
            if key and key not in allowed:
                offenders.setdefault(conf.name, set()).add(key)
    assert not offenders, (
        f"conf route keys outside the vendor names: {offenders}. `cuda` is the "
        "pre-rename spelling of the boxing key and is still parsed, but nothing "
        "generated may emit it."
    )


def test_the_old_spellings_still_parse():
    """An out-of-tree conf or FLAGOS_OP_* written before the rename."""
    text = COMMON_CC.read_text(encoding="utf-8")
    parse = text[text.index("bool ParseBackendName") :]
    parse = parse[: parse.index("const char* BackendName")]
    for legacy in ('"cuda"', '"gcu"', '"musa"', '"mps"', '"flagos_python"'):
        assert legacy in parse, f"{legacy} must stay accepted"
    # `nvidia` is the emitted key the other tests require; the parser has to
    # accept it too, or the conf is read as an unknown backend and every op it
    # routes that way silently takes the FlagGems default. A vendor no conf
    # routes to (drobotics) needs no arm of its own.
    shipped = set()
    for conf in (REPO_ROOT / "torch_fl" / "configs").glob("backends_*.conf"):
        for line in conf.read_text(encoding="utf-8").splitlines():
            body = line.split("#", 1)[0]
            if "=" in body:
                shipped.add(body.partition("=")[2].strip())
    for vendor in sorted({row["vendor"] for row in _platforms().values()}):
        if vendor in shipped:
            assert f'"{vendor}"' in parse, f"{vendor} must stay accepted"
