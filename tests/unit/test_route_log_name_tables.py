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

The integration suites resolve an op's conf route into the name that
``[flagos dispatch] <op> -> <name>`` prints, through a ``_LOG_NAME`` table
owned by their helper module (``tests/integration/ops/backend_conf.py``,
``runtime/runtime_caps.py``, ``flaggems/flaggems_caps.py``). An unmapped conf
value falls through to its own spelling, so a conf row in a different
vocabulary than the log (the Apple rows say ``apple``; ``LogDispatch`` prints
``mps``) fails those suites on that platform alone -- and they only run on its
hardware.

This reads every suite's table, and checks each one against the values the
shipped confs route to and the names ``ParseBackendName``/``LogDispatch`` give
them. It is why the tables can stay separate without drifting apart.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "torch_fl" / "configs"
TABLES = (
    ROOT / "tests" / "integration" / "ops" / "backend_conf.py",
    ROOT / "tests" / "integration" / "runtime" / "runtime_caps.py",
    ROOT / "tests" / "integration" / "flaggems" / "flaggems_caps.py",
)
DISPATCHER = ROOT / "csrc" / "aten" / "dispatcher.h"

#: conf value -> the name LogDispatch prints for the backend it parses to.
#: `nvidia` is the NVIDIA kernel set's key; the kernel logs as `boxing`.
LOGGED_AS = {
    "flaggems": "flagos_python",
    "flaggems_python": "flagos_python",
    "flagos_python": "flagos_python",
    "flaggems_cpp": "flagos",
    "flagos": "flagos",
    "apple": "mps",
    "mps": "mps",
    "drobotics": "drobotics",
    "bpu": "drobotics",
    "enflame": "gcu",
    "gcu": "gcu",
    "mthreads": "musa",
    "musa": "musa",
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


def _logged_names():
    text = DISPATCHER.read_text(encoding="utf-8")
    log = text[text.index("LogDispatch") :]
    return set(re.findall(r'name = "(\w+)"', log))


def _table(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.Assign)
        and any(getattr(target, "id", "") == "_LOG_NAME" for target in item.targets)
    )
    return ast.literal_eval(node.value)


def _resolve(table, value):
    return table.get(value, value)


def test_every_conf_value_resolves_to_a_logged_name():
    logged = _logged_names()
    assert logged, "no dispatch-log names found in dispatcher.h"
    assert set(_conf_values()) <= set(LOGGED_AS), sorted(
        _conf_values() - set(LOGGED_AS)
    )
    table = _table(TABLES[0])
    problems = {}
    for value in sorted(_conf_values()):
        if _resolve(table, value) not in logged:
            problems[value] = _resolve(table, value)
    assert not problems, problems


def test_the_three_tables_agree():
    """Each resolves every mapped value to the same logged name.

    One table may hold an entry another does not: a suite only reads the routes
    it asserts. Two tables holding the same value with different answers is the
    drift this catches.
    """
    tables = {path.relative_to(ROOT): _table(path) for path in TABLES}
    resolved = {}
    for name, table in tables.items():
        for value, logged in table.items():
            known = resolved.setdefault(value, logged)
            assert known == logged, (name, value, logged, known)


def test_every_table_entry_names_a_logged_backend():
    logged = _logged_names()
    for path in TABLES:
        for value, name in _table(path).items():
            assert name in logged, (path.relative_to(ROOT), value, name)
            assert LOGGED_AS.get(value) == name, (path.relative_to(ROOT), value, name)
