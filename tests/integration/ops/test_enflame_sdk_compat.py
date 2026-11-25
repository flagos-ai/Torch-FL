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

import os
import subprocess
import sys

import pytest
import torch

from torch_fl import flagos

pytestmark = pytest.mark.gcu


def test_enflame_device_name():
    name = flagos.get_device_properties("flagos:0").name
    assert name and not name.startswith("Ascend NPU"), name


def test_pinned_host_copy():
    source = torch.arange(64, dtype=torch.float32).pin_memory()
    assert source.is_pinned()
    result = source.to("flagos:0", non_blocking=True)
    flagos.synchronize()
    assert result.device == torch.device("flagos:0")
    torch.testing.assert_close(result.cpu(), source, atol=0, rtol=0)


def test_flagtree_jit_with_an_empty_cache(tmp_path):
    from torch_fl.accelerator.enflame._gcu_compat import (
        is_flagtree_enflame_available,
    )

    assert is_flagtree_enflame_available(), "requires the FlagTree Enflame backend"

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
import torch_fl

source = torch.linspace(-1, 1, 4096, dtype=torch.float32)
result = torch.exp2(source.to("flagos:0"))
torch_fl.flagos.synchronize()
assert result.device == torch.device("flagos:0")
torch.testing.assert_close(result.cpu(), source.exp2(), atol=1e-6, rtol=1e-5)
""",
        ],
        env={
            **os.environ,
            "FLAGOS_LOG": "dispatch,fallback",
            "TRITON_CACHE_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "[flagos dispatch] exp2 -> flagos_python" in process.stderr
    assert "[flagos cpu_fallback]" not in process.stderr
