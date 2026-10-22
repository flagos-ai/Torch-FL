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

"""The .hbm cache key.

Weights are compiled *into* the artifact, so the key has to cover them. Hashing
only the graph structure makes two same-shaped models collide, and the second
one silently runs the first one's weights -- a wrong answer with no error. None
of these tests needs hbdk4 or the device.
"""

from __future__ import annotations

import pytest
import torch

torch_fl = pytest.importorskip("torch_fl")

from torch_fl.accelerator.bpu.compiler import graph_key  # noqa: E402


def _traced(seed: int, scale: float = 1.0) -> torch.fx.GraphModule:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
    ).eval()
    with torch.no_grad():
        model[0].weight.mul_(scale)
    return torch.fx.symbolic_trace(model)


EX = [torch.randn(1, 3, 16, 16)]


def test_same_model_gives_a_stable_key():
    gm = _traced(0)
    assert graph_key(gm, EX, "nash-p") == graph_key(gm, EX, "nash-p")


def test_different_weights_give_different_keys():
    """The bug this file exists for: same architecture, different weights."""
    a, b = _traced(1), _traced(2)
    assert graph_key(a, EX, "nash-p") != graph_key(b, EX, "nash-p")


def test_a_small_weight_change_changes_the_key():
    """Retraining a model must not silently reuse the old artifact."""
    base = _traced(3)
    tweaked = _traced(3)
    with torch.no_grad():
        tweaked.get_parameter("0.weight")[0, 0, 0, 0] += 1.0
    assert graph_key(base, EX, "nash-p") != graph_key(tweaked, EX, "nash-p")


def test_key_still_separates_march_and_input_shape():
    gm = _traced(4)
    assert graph_key(gm, EX, "nash-p") != graph_key(gm, EX, "nash-e")
    other = [torch.randn(1, 3, 32, 32)]
    assert graph_key(gm, EX, "nash-p") != graph_key(gm, other, "nash-p")


def test_large_weights_are_still_separated():
    """Big tensors are sampled rather than hashed whole; that must still work."""
    torch.manual_seed(5)
    big_a = torch.fx.symbolic_trace(
        torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, padding=1)).eval()
    )
    big_b = torch.fx.symbolic_trace(
        torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, padding=1)).eval()
    )
    ex = [torch.randn(1, 64, 8, 8)]
    assert graph_key(big_a, ex, "nash-p") != graph_key(big_b, ex, "nash-p")
