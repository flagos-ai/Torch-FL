# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Device and thread isolation of the published vendor precision adapters."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch_fl

pytestmark = [
    pytest.mark.nvidia,
    pytest.mark.native_source_host,
    pytest.mark.native_vendor_precision,
    pytest.mark.flaggems,
    pytest.mark.main_ops,
]


@pytest.fixture(autouse=True)
def require_native_vendor():
    assert torch_fl._build_config.NATIVE_VENDOR


def exercise(device):
    value = torch.full((3, 8), 4.0, device=device)
    matrix = torch.ones((3, 8), dtype=torch.bfloat16, device=device)
    other = torch.ones((8, 5), dtype=torch.bfloat16, device=device)
    results = (
        (value.pow(2), torch.full((3, 8), 16.0)),
        (value.mean(-1), torch.full((3,), 4.0)),
        (value.rsqrt(), torch.full((3, 8), 0.5)),
        (value.softmax(-1), torch.full((3, 8), 0.125)),
        (matrix @ other, torch.full((3, 5), 8.0, dtype=torch.bfloat16)),
        (
            torch.bmm(matrix.expand(2, 3, 8), other.expand(2, 8, 5)),
            torch.full((2, 3, 5), 8.0, dtype=torch.bfloat16),
        ),
    )
    for actual, expected in results:
        assert actual.device == torch.device(device)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.min_devices(2)
def test_vendor_precision_preserves_ambient_device_and_stream():
    initial = torch_fl.flagos.current_device()
    other = (initial + 1) % torch_fl.flagos.device_count()
    stream = torch.Stream(device=f"flagos:{other}")
    with stream, torch.no_grad():
        torch_fl.flagos.set_device(initial)
        exercise(f"flagos:{other}")
        assert torch_fl.flagos.current_device() == initial
        assert torch.accelerator.current_stream(other).stream_id == stream.stream_id
    assert torch_fl.flagos.current_device() == initial


def test_vendor_precision_handles_are_thread_and_stream_local():
    def worker():
        stream = torch.Stream(device="flagos:0")
        with stream, torch.no_grad():
            for _ in range(3):
                exercise("flagos:0")
            return stream.stream_id

    with ThreadPoolExecutor(max_workers=3) as pool:
        streams = list(pool.map(lambda _: worker(), range(3)))
    assert len(set(streams)) == 3


def test_batched_optimizer_calls_use_native_vendor_on_the_current_stream():
    from torch_fl.native_host import vendor

    def update(weights, moments, variances, grads):
        torch._foreach_mul_(weights, 0.5)
        torch._foreach_lerp_(moments, grads, 0.5)
        torch._foreach_addcmul_(variances, grads, grads, value=0.5)
        denom = torch._foreach_sqrt(variances)
        torch._foreach_div_(denom, [0.5, 1.0])
        torch._foreach_add_(denom, 0.5)
        torch._foreach_addcdiv_(weights, moments, denom, [-0.5, -0.5])
        return weights, moments, variances, denom

    shapes = ((3, 8), (129,))
    sources = [
        [torch.full(shape, fill, dtype=torch.bfloat16) for shape in shapes]
        for fill in (4.0, 0.25, 0.5, 1.0)
    ]
    stream = torch.Stream(device="flagos:0")
    with stream, torch.no_grad():
        inputs = [[value.to("flagos:0") for value in group] for group in sources]
        before = vendor.audit()["hits"]
        actual = update(*inputs)
    stream.synchronize()
    expected = update(*sources)
    after = vendor.audit()["hits"]
    foreach_counts = {
        op: after[op] - count
        for op, count in before.items()
        if op.startswith("_foreach_")
    }
    assert foreach_counts == {
        "_foreach_add_.Scalar": 1,
        "_foreach_addcdiv_.ScalarList": 1,
        "_foreach_addcmul_.Scalar": 1,
        "_foreach_div_.ScalarList": 1,
        "_foreach_lerp_.Scalar": 1,
        "_foreach_mul_.Scalar": 1,
        "_foreach_sqrt": 1,
    }
    for group, reference in zip(actual, expected):
        for value, target in zip(group, reference):
            torch.testing.assert_close(value.cpu(), target, atol=0, rtol=0)

    noncontiguous = torch.ones((3, 8), dtype=torch.bfloat16, device="flagos:0").T
    torch._foreach_mul_([noncontiguous], 2.0)
    assert (
        vendor.audit()["hits"]["_foreach_mul_.Scalar"] == after["_foreach_mul_.Scalar"]
    )
    torch.testing.assert_close(
        noncontiguous.cpu(), torch.full((8, 3), 2.0, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_sum_uses_zero_identity_and_preserves_gradient_reduction(dtype):
    from torch_fl.native_host import vendor

    cpu = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
    original = cpu.to("flagos:0").transpose(0, 1)
    empty = original[..., :0]
    before = vendor.audit()["hits"]["sum.dim_IntList"]
    summed = original.sum(dim=(0, 1), keepdim=True)
    zeros = empty.sum(dim=-1)
    assert vendor.audit()["hits"]["sum.dim_IntList"] == before + 2
    torch.testing.assert_close(
        summed.cpu(), cpu.transpose(0, 1).sum(dim=(0, 1), keepdim=True)
    )
    torch.testing.assert_close(zeros.cpu(), torch.zeros((3, 2), dtype=dtype))


@torch.no_grad()
@pytest.mark.parametrize("layout", ["contiguous", "strided", "empty"])
def test_native_softmax_backward_writes_the_supplied_output(layout):
    from torch_fl.native_host import vendor

    gradient = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
    probability = torch.full_like(gradient, 0.125)
    op = torch.ops.aten._softmax_backward_data
    expected = op.default(gradient, probability, -1, torch.float32)
    stream = torch.Stream(device="flagos:0")
    with stream:
        grad = gradient.to("flagos:0")
        prob = probability.to("flagos:0")
        if layout == "contiguous":
            out = torch.empty_like(grad)
        elif layout == "strided":
            out = torch.empty(2, 3, 16, device=grad.device)[..., ::2]
        else:
            out = torch.empty(0, device=grad.device)
        pointer, stride = out.data_ptr(), out.stride()
        before = vendor.audit()["hits"]["_softmax_backward_data.out"]
        actual = op.out(grad, prob, -1, torch.float32, grad_input=out)
    stream.synchronize()
    assert actual is out
    if layout != "empty":
        assert actual.data_ptr() == pointer
        assert actual.stride() == stride
    assert vendor.audit()["hits"]["_softmax_backward_data.out"] == before + 1
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize(
    "name", ["_softmax_backward_data", "_log_softmax_backward_data"]
)
def test_native_softmax_backward_functional_uses_one_vendor_entry(name):
    from torch_fl.native_host import vendor

    grad = (torch.arange(48, dtype=torch.float32) / 64).reshape(2, 3, 8).transpose(0, 1)
    value = torch.full_like(grad, 0.125)
    if name.startswith("_log"):
        value = value.log()
    op = getattr(torch.ops.aten, name).default
    expected = op(grad, value, -1, torch.float32)

    stream = torch.Stream(device="flagos:0")
    with stream:
        before = vendor.audit()["hits"]
        actual = op(grad.to(stream.device), value.to(stream.device), -1, torch.float32)
    stream.synchronize()
    after = vendor.audit()["hits"]
    assert after[name] == before[name] + 1
    assert after[f"{name}.out"] == before[f"{name}.out"]
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)
