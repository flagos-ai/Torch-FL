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

"""Gate BPU unit tests to the D-Robotics BPU build."""

from __future__ import annotations

from pathlib import Path

import pytest

import torch_fl


# BPU unit tests exercise the board-specific runtime/compiler contract. A CPU,
# CUDA, or other vendor build must not collect them as ordinary unit tests.
_ROOT = Path(__file__).resolve().parents[3]
_REASON = "BPU unit tests require a D-Robotics S600 build (FLAGOS_ACCELERATOR=bpu)"


def pytest_ignore_collect(collection_path, config):
    """Do not import dependencies from another platform's test sources."""
    del config
    path = Path(str(collection_path)).resolve()
    return (
        path.is_relative_to(_ROOT / "tests/unit/bpu")
        and path.name.startswith("test_")
        and path.suffix == ".py"
        and torch_fl._build_accelerator() != "bpu"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Guard any BPU module collected through a direct test path."""
    del config
    for item in items:
        if item.path.resolve().is_relative_to(_ROOT / "tests/unit/bpu"):
            item.add_marker(pytest.mark.skip(reason=_REASON))
