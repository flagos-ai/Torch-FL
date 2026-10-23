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
    """Accelerators whose ``setup()`` can set ``kwargs["entry_points"] = {}``.

    Read from the helper that makes the decision, in the shape of build that
    drops ``scripts=``; a build that keeps it keeps the hook too, which a test
    in ``test_backend_autoload_contract.py`` pins.
    """
    text = SETUP.read_text(encoding="utf-8")
    body = text[text.index("def _wheel_omits_entry_points") :]
    body = body[: body.index("\ndef ")]
    conditions = re.findall(r'if FLAGOS_ACCELERATOR != "([a-z0-9_]+)"', body)
    assert 'kwargs["entry_points"] = {}' in text, (
        "setup.py has no branch that omits the entry-points table"
    )
    return set(conditions)


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
