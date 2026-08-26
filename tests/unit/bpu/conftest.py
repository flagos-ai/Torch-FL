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

import pytest

import torch_fl


# BPU unit tests exercise the board-specific runtime/compiler contract. A CPU,
# CUDA, or other vendor build must not collect them as ordinary unit tests.
requires_bpu = pytest.mark.skipif(
    torch_fl._build_accelerator() != "bpu",
    reason="BPU unit tests require a D-Robotics S600 build (ACCELERATOR=bpu)",
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the BPU test directory unless the build targets the BPU."""
    del config
    for item in items:
        if item.path.parent.name == "bpu":
            item.add_marker(requires_bpu)
