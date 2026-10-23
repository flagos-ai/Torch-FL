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

"""A conf route a test can assert against names the backend the log prints.

The integration tests assert ``[flagos dispatch] <op> -> <name>`` against the
route their platform's conf lists, through ``_LOG_NAME`` in
``tests/integration/ops/backend_conf.py``: ``routed_backend()`` resolves the
conf value, and an unmapped one falls through to its own spelling. A conf row
in a vocabulary ``LogDispatch`` does not print therefore fails those tests on
that platform alone -- and they only run on its hardware.

Both sides are read here: every value the confs route to, and the name the log
prints for the backend ``ParseBackendName`` reads that value as. The table is
deliberate rather than derived from the SDK-era aliases in the parser, because
the aliases are accepted for old confs and `nvidia`/`nvidia`-era keys do not all
print their own spelling.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "torch_fl" / "configs"
INTEGRATION_CONF = ROOT / "tests" / "integration" / "ops" / "backend_conf.py"
DISPATCHER = ROOT / "csrc" / "aten" / "dispatcher.h"
COMMON_CC = ROOT / "csrc" / "aten" / "common.cc"

#: conf value -> the name LogDispatch prints for the backend it parses to.
#: `nvidia` is the NVIDIA kernel set's key; the kernel logs as `boxing`.
LOGGED_AS = {
    "flaggems": "flagos_python",
    "flaggems_python": "flagos_python",
    "flagos_python": "flagos_python",
    "flaggems_cpp": "flagos",
    "flagos": "flagos",
    "apple": "mps",
    "enflame": "gcu",
    "mthreads": "musa",
    "nvidia": "boxing",
    "boxing": "boxing",
    "ascend": "ascend",
    "metax": "metax",
    "tsingmicro": "tsingmicro",
    "kunlunxin": "kunlunxin",
    "iluvatar": "iluvatar",
    "tileops": "tileops",
    "none": "none",
}


def _conf_values():
    values = set()
    for conf in CONF_DIR.glob("backends_*.conf"):
        for line in conf.read_text(encoding="utf-8").splitlines():
            _name, sep, value = line.split("#")[0].partition("=")
            if sep:
                values.add(value.strip())
    return values


def _source_section(path, start, end):
    text = path.read_text(encoding="utf-8")
    return text[text.index(start) : text.index(end)]


def _parser_accepts(value):
    """Whether ``ParseBackendName`` reads ``value`` as a backend."""
    parse = _source_section(
        COMMON_CC, "bool ParseBackendName", "const char* BackendName"
    )
    return f'"{value}"' in parse


def _logged_names():
    """The names ``LogDispatch`` can print: its ``name = "..."`` assignments."""
    log = _source_section(DISPATCHER, "static void LogDispatch", "const char* op_name_")
    return set(re.findall(r'name = "(\w+)"', log))


def _integration_mapping():
    """The ``_LOG_NAME`` dict literal, read without importing torch_fl."""
    tree = ast.parse(INTEGRATION_CONF.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.Assign)
        and any(getattr(target, "id", "") == "_LOG_NAME" for target in item.targets)
    )
    return ast.literal_eval(node.value)


def test_every_conf_route_is_a_backend_the_log_names():
    logged = _logged_names()
    assert logged, "no dispatch-log names found in dispatcher.h"
    problems = {}
    for value in sorted(_conf_values()):
        if not _parser_accepts(value):
            problems[value] = "ParseBackendName reads it as an unknown backend"
        elif LOGGED_AS.get(value) not in logged:
            problems[value] = f"the log has no {LOGGED_AS.get(value)!r}"
    assert not problems, problems


def test_the_integration_tests_map_onto_the_logged_name():
    logged = _logged_names()
    mapping = _integration_mapping()
    for value, name in mapping.items():
        assert name == LOGGED_AS.get(value, value), (value, name)
        assert name in logged, (value, name)


def test_the_parser_accepts_every_key_the_log_side_knows():
    parse = _source_section(
        COMMON_CC, "bool ParseBackendName", "const char* BackendName"
    )
    for value in LOGGED_AS:
        if value in {"flagos_python", "flagos"}:
            continue  # log-side spellings the parser accepts differently
        assert f'"{value}"' in parse, value
    # The SDK-era aliases still read the same backend as the vendor key.
    assert '"enflame" || val == "gcu"' in parse
    assert '"mthreads" || val == "musa"' in parse
    assert '"apple" || val == "mps"' in parse
