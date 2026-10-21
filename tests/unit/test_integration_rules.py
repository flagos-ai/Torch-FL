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

"""Keep the static device-import gate narrow enough to trust."""

import importlib.util
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "integration_rules",
    ROOT / "tests/integration/integration_rules.py",
)
assert SPEC is not None and SPEC.loader is not None
RULES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RULES)


def test_shared_initializer_allows_a_torch_submodule_import():
    source = textwrap.dedent(
        """
        from device_init import torch, torch_fl
        import torch.nn.functional as F

        F = torch.nn.functional
        """
    )
    assert RULES.checked_source(source)


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("from device_init import torch\n", None),
        ("from device_init import torch_fl, torch\n", None),
        (
            "from device_init import torch as selected\nimport torch.nn as nn\n",
            "before torch loads",
        ),
    ],
)
def test_only_a_torch_initializer_counts_as_complete(source, error):
    if error is None:
        assert RULES.checked_source(source)
        return
    with pytest.raises(AssertionError, match=error):
        RULES.checked_source(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import torch\n", "before torch loads"),
        (
            "import torch\nfrom device_init import torch, torch_fl\n",
            "follow the shared initializer",
        ),
        (
            "from device_init import torch, torch_fl\ndef check():\n    import torch\n",
            "function-local",
        ),
        (
            "from device_init import torch, torch_fl\nimport torch_fl\n",
            "rebound directly",
        ),
        (
            "from device_init import torch\nimport torch\n",
            "rebound directly",
        ),
        (
            "from device_init import torch, torch_fl\n"
            "from device_init import torch, torch_fl\n",
            "initialize the device once",
        ),
        ("import torch\ndef broken(\n", "cannot parse source"),
    ],
)
def test_invalid_device_import_order_fails(source, message):
    with pytest.raises(AssertionError, match=message):
        RULES.checked_source(source)


def test_checker_reports_the_source_path(tmp_path):
    source = tmp_path / "bad_import.py"
    source.write_text("import torch\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=str(source.resolve())):
        RULES.check_source(source)
