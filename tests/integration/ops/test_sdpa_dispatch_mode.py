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

"""
scaled_dot_product_attention keeps its autograd graph under a TorchDispatchMode.

A mode is claimed at the Python dispatch key, which sits below the autograd keys
and above every backend key, and the region a mode redispatches into is below
autograd. An op whose grad path lives inside a *device-key* kernel body therefore
runs there with the autograd key excluded, and everything that body dispatches
loses it too: the composite flagos SDPA re-runs for the grad case built no graph
at all, so the output came back detached and its backward raised "element 0 of
tensors does not require grad and does not have a grad_fn". The fix is a second
registration of the same body on AutogradPrivateUse1, above the mode
(csrc/aten/sdp_choice_stub.cc; issue flagos-ai/Torch-FL#383).

A mode observes dispatch, so it must not change the computation. These tests
assert exactly that, because the defect is invisible to every check that does not
look at gradients: the forward is unaffected, so a test that only reads values or
`isfinite(loss)` cannot see a dropped attention gradient. No other test in this
directory installs a mode, which is why the registration position went unnoticed
through every earlier SDPA and autograd-registration fix.

Usage:
    pytest tests/integration/ops/test_sdpa_dispatch_mode.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode


DEVICE = "flagos:0"

# Scoring matches the neighbouring op tests: relative to the same-shape
# no-mode result taken on the device, so a precision difference is not mistaken
# for a lost graph.
TOL = 2e-3

# (q/k/v shape, id) -- batches x heads x sequence x head_dim.
SHAPE_CASES = [
    ((1, 4, 8, 8), "single_batch"),
    ((2, 8, 16, 16), "batched_multi_head"),
    ((2, 4, 8, 16), "head_dim_16"),
]


class Passthrough(TorchDispatchMode):
    """The smallest mode that still dispatches: dispatch, call through, return."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def _grad_fn_name(tensor: torch.Tensor) -> str:
    return type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else "None"


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error relative to the reference's magnitude."""
    return (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)


def _sdpa(shape, seed=0):
    """q/k/v on the device, all requiring grad, as a single attention call."""
    torch.manual_seed(seed)
    q = torch.randn(shape, device=DEVICE).requires_grad_(True)
    k = torch.randn(shape, device=DEVICE).requires_grad_(True)
    v = torch.randn(shape, device=DEVICE).requires_grad_(True)
    return q, k, v


def _forward_backward(shape, mode=None):
    """One forward+backward, with `mode` installed around both if given.

    Wrapping the backward as well as the forward is the shape that matters for
    a real training step (tests/integration/test_fallback_trace_train.py holds a
    mode across forward, backward and optimizer.step()), and it is also what a
    mode that only wraps the backward looks like from inside.
    """
    q, k, v = _sdpa(shape)
    if mode is None:
        out = F.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
    else:
        with mode:
            out = F.scaled_dot_product_attention(q, k, v)
            out.sum().backward()
    return out, (q.grad, k.grad, v.grad)


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "shape", [s for s, _ in SHAPE_CASES], ids=[i for _, i in SHAPE_CASES]
)
def test_sdpa_keeps_its_autograd_graph_in_a_dispatch_mode(shape):
    """The forward must build the same graph with a mode active."""
    q, k, v = _sdpa(shape)
    with Passthrough():
        out = F.scaled_dot_product_attention(q, k, v)

    assert out.requires_grad, (
        "SDPA returned a detached tensor inside a dispatch mode; "
        f"grad_fn={_grad_fn_name(out)}"
    )
    assert out.grad_fn is not None, "SDPA produced no grad_fn inside a dispatch mode"

    # The backward runs with the mode gone, which is the ordinary case: it must
    # reach all three inputs.
    out.sum().backward()
    for name, tensor in (("query", q), ("key", k), ("value", v)):
        assert tensor.grad is not None, f"backward never reached {name}"
        assert tensor.grad.abs().sum().item() > 0, (
            f"{name} received an all-zero gradient"
        )


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "shape", [s for s, _ in SHAPE_CASES], ids=[i for _, i in SHAPE_CASES]
)
def test_sdpa_mode_does_not_change_the_gradients(shape):
    """Same inputs, same step: the mode must not change the numbers."""
    out_plain, grads_plain = _forward_backward(shape, mode=None)
    out_mode, grads_mode = _forward_backward(shape, mode=Passthrough())

    assert _rel_err(out_mode.detach(), out_plain.detach()) < TOL, "forward differs"
    for name, got, ref in zip(("query", "key", "value"), grads_mode, grads_plain):
        assert got is not None, f"{name} lost its gradient inside a dispatch mode"
        assert _rel_err(got, ref) < TOL, (
            f"{name} gradient differs with a dispatch mode active "
            f"(rel err {_rel_err(got, ref):.3e})"
        )


@pytest.mark.anyplatform
def test_sdpa_gradients_survive_a_mode_across_the_training_step():
    """The checked-in harness shape: mode held across forward, backward, step."""
    torch.manual_seed(0)
    model = torch.nn.Linear(32, 32, bias=False).to(DEVICE)
    x = torch.randn(2, 4, 32, device=DEVICE)
    before = model.weight.detach().clone()

    with Passthrough():
        q = model(x)
        loss = F.scaled_dot_product_attention(q, q, q).sum()
        loss.backward()

    assert model.weight.grad is not None, "no gradient reached the projection"
    assert model.weight.grad.abs().sum().item() > 0, (
        "the projection got an all-zero gradient"
    )

    with torch.no_grad():
        model.weight -= 0.01 * model.weight.grad
    assert not torch.equal(model.weight, before), "the step did not move the weights"


@pytest.mark.anyplatform
def test_sdpa_inference_is_unaffected_by_a_dispatch_mode():
    """A mode must not change the inference route or its values.

    Nothing requires grad here, so the wrapper takes its boxing/FlagGems branch
    (issue #269's route) instead of the composite; that route has to keep
    producing the same tensor with a mode active.
    """
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 16, device=DEVICE)

    plain = F.scaled_dot_product_attention(q, q, q)
    with Passthrough():
        in_mode = F.scaled_dot_product_attention(q, q, q)

    assert not in_mode.requires_grad and in_mode.grad_fn is None
    assert in_mode.device.type == plain.device.type
    assert torch.equal(in_mode, plain), "the inference value changed with a mode active"
