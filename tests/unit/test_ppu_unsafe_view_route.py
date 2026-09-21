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

"""Pin PPU routes whose value-only checks miss a contract failure."""

import ast
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_GENERATOR = _REPO / "scripts/codegen/gen_vendor_confs.py"
_CONFIG = _REPO / "torch_fl/configs/backends_ppu.conf"


def test_ppu_contract_failures_are_regenerable_boxing_exceptions():
    tree = ast.parse(_GENERATOR.read_text())
    gaps = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "BOXING_TRITON_GAPS"
            for target in node.targets
        ):
            continue
        gaps = ast.literal_eval(node.value)
        break

    assert gaps is not None
    assert {"_unsafe_view", "slice.Tensor"} <= gaps["ppu"]

    routes = {}
    for raw in _CONFIG.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            op, backend = (part.strip() for part in line.split("=", 1))
            routes[op] = backend
    assert routes["_unsafe_view"] == "cuda"
    assert routes["slice.Tensor"] == "cuda"
