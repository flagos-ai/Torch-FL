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

"""The `cuda` -> flagos device alias must not break FX or Dynamo.

`_alias_cuda_to_flagos` rebinds `torch.device` to a Python wrapper class so that
`torch.device("cuda")` lands on the accelerator that is actually present. Two
torch registries compare against that attribute by *identity* rather than
structurally, so the swap silently changed their behaviour:

  * `torch.fx.graph.add_global` special-cases `obj != torch.device` to emit a
    device constant as the bare name `device(type='cpu')`. Once the attribute
    was a different object the branch fell through to the qualified-name path
    for custom ops, the name was never added to the generated module's globals,
    and running the graph raised `NameError: name 'device' is not defined`.
  * `torch._dynamo.utils.common_constant_types` is membership-tested with
    `type(obj) in ...`, so reading `tensor.device` during tracing asserted with
    "Cannot construct `ConstantVariable` for value of type `torch.device`".

Neither surfaces without an accelerator alias in place, and together they took
out every model that calls `torch.arange(..., device=...)` -- which is how HF
builds position ids, i.e. every transformer.
"""

from __future__ import annotations

import pytest
import torch

torch_fl = pytest.importorskip("torch_fl")


def _aliased() -> bool:
    """Whether this build actually installed the alias."""
    return torch.device is not torch._C.device


requires_alias = pytest.mark.skipif(
    not _aliased(), reason="build has real CUDA, so the alias is a no-op"
)


def test_device_constructor_still_returns_a_real_device():
    d = torch.device("cpu")
    assert type(d) is torch._C.device
    assert isinstance(d, torch.device)
    assert d.type == "cpu"


@requires_alias
def test_repr_round_trips_through_the_shim():
    """FX codegen emits `repr(device)` verbatim and resolves it as a call."""
    for d in (torch.device("cpu"), torch.device("cpu", 0)):
        back = eval(repr(d), {"device": torch.device})  # noqa: S307
        assert back == d
        assert type(back) is torch._C.device


def test_fx_custom_builtin_matches_the_live_attribute():
    """`add_global`'s carve-out compares against `torch.device` by identity."""
    from torch.fx.graph import _custom_builtins

    assert _custom_builtins["device"].obj is torch.device


def test_dynamo_treats_a_device_as_a_constant():
    from torch._dynamo.utils import common_constant_types, is_safe_constant

    assert torch._C.device in common_constant_types
    assert is_safe_constant(torch.device("cpu"))


def test_graph_with_a_device_constant_executes():
    """The regression: this raised NameError once the alias was installed."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(x):
        return x + torch.arange(4, device=x.device)

    gm = make_fx(f, tracing_mode="fake")(torch.randn(4))
    assert "device(type=" in gm.code, "expected a bare device constant in codegen"
    assert "device" in gm.forward.__globals__

    x = torch.randn(4)
    torch.testing.assert_close(gm(x), f(x))


def test_reading_tensor_device_while_tracing_does_not_assert():
    """The Dynamo half: `.device` is wrapped as a ConstantVariable."""
    seen = []

    def backend(gm, example_inputs):
        seen.append(gm)
        return gm.forward

    @torch.compile(backend=backend, fullgraph=True)
    def f(x):
        return x + torch.arange(x.shape[0], device=x.device)

    x = torch.randn(4)
    torch.testing.assert_close(f(x), x + torch.arange(4))
    assert seen, "backend never ran"
