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

"""Unit coverage for torch_fl/_vendor.py, the GEMS_VENDOR chokepoint.

``_patch_flaggems_codegen_config`` in ``torch_fl/__init__.py`` used to leave an
explicitly exported ``GEMS_VENDOR`` that torch_fl cannot configure in place, and
its final ``else`` silently selected ``ascend`` when no vendor branch matched.
Both are now fail-loud: an unknown explicit value raises here, and the ladder
raises on detection failure instead of defaulting. These tests pin the pure
validation half; the ladder half needs a built torch_fl and is exercised by the
import smoke test.

The module is loaded by path rather than imported: ``torch_fl/_vendor.py`` is
stdlib-only by design so this file needs no torch install, and importing the
package would execute ``torch_fl/__init__.py``.
"""

import importlib.machinery
import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_vendor():
    loader = importlib.machinery.SourceFileLoader(
        "torch_fl_vendor_under_test", str(REPO_ROOT / "torch_fl" / "_vendor.py")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_unknown_value_raises_and_lists_valid_vendors():
    vendor = _load_vendor()
    with pytest.raises(RuntimeError) as excinfo:
        vendor.validate_explicit("not-a-vendor")
    message = str(excinfo.value)
    assert "not-a-vendor" in message
    assert "not recognized" in message
    # The message must name the set so the user can fix the typo without reading
    # the source.
    for known in ("nvidia", "ascend", "metax"):
        assert known in message


def test_known_values_are_normalized():
    vendor = _load_vendor()
    assert vendor.validate_explicit("NVIDIA") == "nvidia"
    assert vendor.validate_explicit("  Metax  ") == "metax"
    assert vendor.validate_explicit("cuda") == "cuda"


def test_unset_or_blank_is_none():
    vendor = _load_vendor()
    # None, unset and blank all mean "detect for me"; an empty export is how a
    # user says "not this vendor".
    assert vendor.validate_explicit(None) is None
    assert vendor.validate_explicit("") is None
    assert vendor.validate_explicit("   ") is None


def test_known_vendors_covers_the_comm_profiles():
    """Every vendor the comm layer can route is one torch_fl accepts.

    ``comm/process_group._VENDOR_PROFILES`` is the other consumer of
    GEMS_VENDOR; a vendor it lists but ``KNOWN_VENDORS`` rejects would be
    impossible to select explicitly. Parsed as text so this needs no torch.
    """
    vendor = _load_vendor()
    text = (REPO_ROOT / "torch_fl" / "comm" / "process_group.py").read_text(
        encoding="utf-8"
    )
    profiles = set(re.findall(r'^\s*"([a-z0-9_]+)":\s*_VendorProfile\(', text, re.M))
    assert profiles, "no _VENDOR_PROFILES entries parsed -- did the format change?"
    assert profiles <= vendor.KNOWN_VENDORS, profiles - vendor.KNOWN_VENDORS
