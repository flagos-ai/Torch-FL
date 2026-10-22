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

"""The FlagGems device identity and guarded-route cases on every configured CI platform.

FlagGems compares a tensor's device with the name its vendor backend resolved.
torch_fl registers every accelerator as ``flagos``, so a stale name sends a routed
call down a fallback or into a vendor guard. These cases were previously gated by
a named chip and skipped on the others. They now declare the exact platform rows
that claim the route; a missing flag_gems import, device or row is a failure.
"""

from __future__ import annotations

import importlib

import pytest
from device_init import torch, torch_fl

from flaggems_caps import (
    FLAGGEMS_BACKENDS,
    accel_label,
    conf_route,
    flaggems_runtime_unavailable,
    route_reason,
)
from flaggems_probe import probe_routes

# The fixture imports torch_fl to claim the flagos device before torch caches its
# PrivateUse1 hooks. Keep those imports in the process that runs the test.

DEVICE = "flagos:0"
SEED = 1234

#: Guarded pointwise routes and their (build, input shape) on a flagos tensor.
_GUARDED_POINTWISE = {
    "i0": (torch.ops.aten.i0, (64,)),
    "i0.out": (torch.ops.aten.i0.out, (64,)),
    "special_i0e": (torch.ops.aten.special_i0e, (64,)),
    "special_i1": (torch.ops.aten.special_i1, (64,)),
    "special_scaled_modified_bessel_k1": (
        torch.ops.aten.special_scaled_modified_bessel_k1,
        (64,),
    ),
    "special_scaled_modified_bessel_k1.out": (
        torch.ops.aten.special_scaled_modified_bessel_k1.out,
        (64,),
    ),
}

#: Reflection padding takes different arities in ATen's schema.
_REFLECTION_PAD = {
    "reflection_pad2d": (torch.ops.aten.reflection_pad2d, (2, 3, 8, 8), [1, 1, 1, 1]),
    "reflection_pad3d": (
        torch.ops.aten.reflection_pad3d,
        (2, 3, 4, 8, 8),
        [1, 1, 1, 1, 1, 1],
    ),
    "reflection_pad2d.out": (
        torch.ops.aten.reflection_pad2d.out,
        (2, 3, 8, 8),
        [1, 1, 1, 1],
    ),
    "reflection_pad3d.out": (
        torch.ops.aten.reflection_pad3d.out,
        (2, 3, 4, 8, 8),
        [1, 1, 1, 1, 1, 1],
    ),
}

_GUARDED_OTHER = {
    "_embedding_bag_dense_backward": (None, (), ()),
    "eq.Tensor": (None, (), ()),
    "eq.Scalar": (None, (), ()),
    "soft_margin_loss": (torch.ops.aten.soft_margin_loss, (64,), ()),
}


def _accel() -> str:
    accel = accel_label()
    if not accel:
        pytest.fail("the active conf is not a named backends_<platform>.conf")
    return accel


def _require_runtime() -> None:
    reason = flaggems_runtime_unavailable()
    if reason:
        pytest.fail(reason)


def _flag_gems():
    import flag_gems

    return flag_gems


def _detector():
    flag_gems = _flag_gems()
    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        from flag_gems.runtime.backend.device import DeviceDetector
    return flag_gems, DeviceDetector()


def _registered_device_name():
    torch_fl.flagos.init()
    return torch._C._get_privateuse1_backend_name()


def _flag_gems_mul():
    entry = importlib.import_module("flag_gems.ops.mul")
    return entry if callable(entry) else entry.mul


def _assert_matches_cpu(produce, reference):
    torch.manual_seed(SEED)
    result = produce()
    torch.flagos.synchronize()
    torch.testing.assert_close(result.to("cpu"), reference, rtol=1e-4, atol=1e-4)


_GUARDED_ROUTES = (*_GUARDED_POINTWISE, *_REFLECTION_PAD, *_GUARDED_OTHER)


def _route_arm(op: str) -> str:
    """Which device path executes this guarded case.

    An unlisted operator resolves to FlagGems. An explicit ``none`` route stays
    on the CPU, so its case still checks the host comparison without claiming a
    FlagGems entry point. A configured FlagGems route must be observed by the
    probe before the parameterized case runs.
    """
    raw = conf_route(op)
    if raw == "none":
        return "cpu"
    if raw is None or raw in FLAGGEMS_BACKENDS:
        return "device"
    pytest.fail(
        f"the guarded case for {op} is selected on {accel_label()}, but "
        f"the conf routes it to {raw!r}; add its own row or fix the conf"
    )


def _assert_case_route(op: str) -> str:
    """Return the arm to execute after checking the one FlagGems route can see."""
    arm = _route_arm(op)
    if arm == "device":
        _assert_route_probe(op)
    return arm


def _assert_route_probe(*ops: str) -> None:
    """Execute a guarded entry point in a fresh child and read its dispatch."""
    report = probe_routes(list(ops))
    failures = []
    for op in ops:
        entry = report[op]
        if entry["error"] is not None:
            failures.append(f"{op}: {entry['error']}")
            continue
        backend, why = route_reason(op)
        expected_fallback = (
            entry["dispatched_backend"] is None and entry["cpu_fallback"]
        )
        if (
            backend is not None
            and entry["dispatched_backend"] == backend
            and not entry["cpu_fallback"]
        ) or (backend is None and expected_fallback):
            continue
        failures.append(
            f"{op} ({why}) dispatched to {entry['dispatched_backend']!r} "
            f"(cpu_fallback={entry['cpu_fallback']}); the "
            f"{accel_label()} route resolves it to {backend!r}"
        )
    if failures:
        pytest.fail(
            "the configured guarded routes did not execute:\n" + "\n".join(failures)
        )


def _require_configured_guarded_routes() -> None:
    """Execute every guarded entry point before this module's identity checks.

    Each parameterized case runs its own entry point as well. This probe keeps
    a claimed FlagGems route from being represented only by a test id. An
    explicitly disabled route still runs on CPU, so no guarded case disappears.
    """
    _assert_route_probe(
        *(name for name in _GUARDED_ROUTES if _route_arm(name) == "device")
    )


def _guarded_input(shape, scale=1.0):
    torch.manual_seed(SEED)
    return torch.rand(shape) * scale + 0.5


@pytest.mark.main_ops
@pytest.mark.flaggems_python
def test_detector_name_matches_registered_backend():
    _require_runtime()
    _accel()
    _require_configured_guarded_routes()
    flag_gems, detector = _detector()
    registered = _registered_device_name()
    assert detector.name == registered
    assert flag_gems.device == registered


@pytest.mark.main_ops
@pytest.mark.flaggems_python
def test_op_module_copies_are_realigned():
    _require_runtime()
    _accel()
    _require_configured_guarded_routes()
    _detector()
    registered = _registered_device_name()
    cumsum_mod = importlib.import_module("flag_gems.ops.cumsum")
    mul_mod = importlib.import_module("flag_gems.ops.mul")
    assert cumsum_mod.device == registered
    assert mul_mod._DEVICE_NAME == registered


@pytest.mark.main_ops
@pytest.mark.flaggems_python
def test_flag_gems_mul_accepts_a_python_scalar():
    _require_runtime()
    _accel()
    _require_configured_guarded_routes()
    _detector()
    torch.manual_seed(SEED)
    value = torch.randn((64, 64), device=DEVICE)
    reference = value.to("cpu") * 1.3333333333333333
    result = _flag_gems_mul()(value, 1.3333333333333333)
    assert torch.allclose(result.to("cpu"), reference)


@pytest.mark.main_ops
@pytest.mark.flaggems_python
def test_python_float_multiply_matches_cpu():
    _require_runtime()
    _accel()
    _assert_case_route("mul.Tensor")
    torch.manual_seed(SEED)
    value = torch.randn((64, 64), device=DEVICE)
    reference = value.to("cpu") * 1.3333333333333333
    assert torch.allclose((value * 1.3333333333333333).to("cpu"), reference)


@pytest.mark.main_ops
@pytest.mark.flaggems_python
@pytest.mark.parametrize("name", sorted(_GUARDED_POINTWISE))
def test_guarded_pointwise_routes_execute(name):
    _require_runtime()
    _accel()
    arm = _assert_case_route(name)
    _detector()
    op, shape = _GUARDED_POINTWISE[name]
    value = _guarded_input(shape, 20.0)
    expected = (
        op(value, out=torch.empty_like(value)) if name.endswith(".out") else op(value)
    )
    if arm == "cpu":
        actual = expected
    elif name.endswith(".out"):
        actual = op(value.to(DEVICE), out=torch.empty_like(value, device=DEVICE))
    else:
        actual = op(value.to(DEVICE))
    _assert_matches_cpu(lambda: actual, expected)


@pytest.mark.main_ops
@pytest.mark.flaggems_python
def test_guarded_soft_margin_loss_route_executes():
    _require_runtime()
    _accel()
    arm = _assert_case_route("soft_margin_loss")
    op, shape, _ = _GUARDED_OTHER["soft_margin_loss"]
    value = _guarded_input(shape, 20.0)
    other = _guarded_input(shape, 20.0)
    expected = op(value, other)
    actual = op(value.to(DEVICE), other.to(DEVICE)) if arm == "device" else expected
    _assert_matches_cpu(lambda: actual, expected)


@pytest.mark.main_ops
@pytest.mark.flaggems_python
@pytest.mark.parametrize("name", sorted(_REFLECTION_PAD))
def test_guarded_reflection_pad_routes_execute(name):
    _require_runtime()
    _accel()
    arm = _assert_case_route(name)
    _detector()
    op, shape, padding = _REFLECTION_PAD[name]
    torch.manual_seed(SEED)
    value = torch.randn(shape)
    padded = len(padding) // 2
    out_shape = tuple(shape[:-padded]) + tuple(s + 2 for s in shape[-padded:])
    expected = (
        op(value, padding, out=torch.empty(out_shape))
        if name.endswith(".out")
        else op(value, padding)
    )
    if arm == "cpu":
        actual = expected
    elif name.endswith(".out"):
        actual = op(
            value.to(DEVICE), padding, out=torch.empty(out_shape, device=DEVICE)
        )
    else:
        actual = op(value.to(DEVICE), padding)
    _assert_matches_cpu(lambda: actual, expected)


@pytest.mark.main_ops
@pytest.mark.flaggems_python
@pytest.mark.parametrize("name", sorted(_GUARDED_OTHER))
def test_other_guarded_routes_execute(name):
    _require_runtime()
    _accel()
    arm = _assert_case_route(name)
    if name == "_embedding_bag_dense_backward":
        _embedding_bag_case(arm)
        return
    if name.startswith("eq."):
        _eq_case(name, arm)
        return
    op, shape, _ = _GUARDED_OTHER[name]
    value = _guarded_input(shape, 20.0)
    other = _guarded_input(shape, 20.0)
    expected = op(value, other)
    actual = op(value.to(DEVICE), other.to(DEVICE)) if arm == "device" else expected
    _assert_matches_cpu(lambda: actual, expected)


def _embedding_bag_case(arm: str) -> None:
    _detector()
    torch.manual_seed(SEED)
    num_weights, dim, num_bags, num_samples = 16, 4, 3, 8
    grad = torch.randn(num_bags, dim)
    indices = torch.randint(0, num_weights, (num_samples,), dtype=torch.int64)
    offset2bag = torch.zeros(num_samples, dtype=torch.int64)
    bag_size = torch.full((num_bags,), num_samples, dtype=torch.int64)
    maximum_indices = torch.zeros(num_bags, dtype=torch.int64)
    call = torch.ops.aten._embedding_bag_dense_backward
    host = (grad, indices, offset2bag, bag_size, maximum_indices)
    expected = call(*host, num_weights, False, 0, None)
    actual = (
        call(*(t.to(DEVICE) for t in host), num_weights, False, 0, None)
        if arm == "device"
        else expected
    )
    torch.testing.assert_close(actual.cpu().float(), expected.float())


def _eq_case(name: str, arm: str) -> None:
    _detector()
    torch.manual_seed(SEED)
    value = torch.rand((64,))
    other = torch.rand((64,))
    if name == "eq.Tensor":
        expected = torch.ops.aten.eq.Tensor(value, other)
        actual = (
            torch.ops.aten.eq.Tensor(value.to(DEVICE), other.to(DEVICE))
            if arm == "device"
            else expected
        )
    else:
        expected = torch.ops.aten.eq.Scalar(value, 0.5)
        actual = (
            torch.ops.aten.eq.Scalar(value.to(DEVICE), 0.5)
            if arm == "device"
            else expected
        )
    _assert_matches_cpu(lambda: actual, expected)


@pytest.mark.main_ops
def test_is_cuda_is_false_on_a_flagos_tensor():
    _accel()
    value = torch.randn((64,), device=DEVICE)
    assert not value.is_cuda


@pytest.mark.main_ops
@pytest.mark.parametrize(
    "specifier",
    ["flagos", "flagos:0", torch.device("flagos"), torch.device("flagos", 0)],
    ids=["name", "name-index", "device", "device-index"],
)
def test_device_properties_accepts_the_registered_name(specifier):
    _accel()
    reference = torch.cuda.get_device_properties(0)
    resolved = torch.cuda.get_device_properties(specifier)
    assert resolved.name == reference.name
    assert resolved.multi_processor_count == reference.multi_processor_count


@pytest.mark.main_ops
@pytest.mark.flaggems_python
def test_masked_select_past_the_single_pass_cutoff():
    _require_runtime()
    _accel()
    _detector()
    torch.manual_seed(SEED)
    value = torch.randn(4097, device=DEVICE)
    mask = value > 0.0
    _assert_matches_cpu(
        lambda: torch.masked_select(value, mask),
        torch.masked_select(value.to("cpu"), mask.to("cpu")),
    )
