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

"""Which wheels omit the autoload hook must stay the list the tests believe.

tests/unit/test_startup_subprocess.py runs in every lane's `autoload_contract`
group, on the wheel's installed files. There it cannot read ``setup.py`` (the
wheel does not ship one), so it carries ``_NO_AUTOLOAD`` -- the accelerators
whose wheel deliberately has no ``torch.backends`` entry point. This pins that
list to the branch in ``setup.py`` that produces it, in the checkout where
``setup.py`` does exist.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP = REPO_ROOT / "setup.py"
STARTUP_TEST = REPO_ROOT / "tests" / "unit" / "test_startup_subprocess.py"


def _omitting_accelerators() -> set[str]:
    """Accelerators whose ``setup()`` kwargs set ``kwargs["entry_points"] = {}``.

    The branch is read from the assignment back to the condition that guards
    it, so it does not have to know where each block ends.
    """
    text = SETUP.read_text(encoding="utf-8")
    omitting = set()
    for match in re.finditer(r'kwargs\["entry_points"\] = \{\}', text):
        before = text[: match.start()]
        conditions = list(re.finditer(r'FLAGOS_ACCELERATOR == "([a-z0-9_]+)"', before))
        if conditions:
            omitting.add(conditions[-1].group(1))
    return omitting


def _declared_omissions() -> set[str]:
    text = STARTUP_TEST.read_text(encoding="utf-8")
    block = text[text.index("_NO_AUTOLOAD = {") :]
    block = block[: block.index("}")]
    return set(re.findall(r'"([a-z0-9_]+)":', block))


def test_the_startup_contract_omits_exactly_what_setup_omits():
    assert _declared_omissions() == _omitting_accelerators(), (
        "test_startup_subprocess._NO_AUTOLOAD disagrees with setup.py's "
        'kwargs["entry_points"] = {} branch'
    )
