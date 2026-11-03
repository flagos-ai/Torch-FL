"""Device arithmetic preserves FP64 when the transport only accepts integers."""

from types import SimpleNamespace
import sys

import pytest
import torch
import torch.distributed as dist

pg = pytest.importorskip("torch_fl.comm.process_group")


@pytest.mark.parametrize("explicit", [None, "torch_npu"])
def test_ascend_flagcx_backend_selection_preserves_user_choice(monkeypatch, explicit):
    monkeypatch.delenv("FLAGCX_TORCH_BACKEND", raising=False)
    if explicit is not None:
        monkeypatch.setenv("FLAGCX_TORCH_BACKEND", explicit)
    pg._configure_flagcx_torch_backend("ascend")
    assert pg.os.environ["FLAGCX_TORCH_BACKEND"] == (explicit or "flagos")


def test_ascend_collective_guard_uses_operand_device_and_restores_caller(monkeypatch):
    monkeypatch.setenv("GEMS_VENDOR", "ascend")
    selected = [0]
    runtime = SimpleNamespace(
        _get_device=lambda: selected[0],
        _set_device=lambda index: selected.__setitem__(0, index),
    )
    monkeypatch.setitem(sys.modules, "torch_fl._C", runtime)
    monkeypatch.setattr(sys.modules["torch_fl"], "_C", runtime, raising=False)
    guarded = pg._device_scoped_guard(lambda group, tensor: selected[0])
    group = SimpleNamespace()

    class IndexedTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("privateuseone:1")

    tensor = torch.empty(1).as_subclass(IndexedTensor)
    assert guarded(group, tensor) == 1
    assert selected[0] == 0


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("rank", range(3))
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize(
    "operation", [dist.ReduceOp.SUM, dist.ReduceOp.AVG, dist.ReduceOp.PRODUCT]
)
def test_device_reduction_retains_sub_fp32_components(
    monkeypatch, rank, dtype, operation, wrapped
):
    values = torch.tensor(
        [[1.00000000001, -2.25], [1.00000000002, 3.75], [1.00000000003, 4.125]],
        dtype=dtype,
    )
    if dtype.is_complex:
        values += 0.00000000001j
    original = values[rank].clone()
    output = original.clone()
    roots = []
    stream_events = []

    def transport_view(tensor):
        return (torch.view_as_real(tensor) if tensor.is_complex() else tensor).view(
            torch.int64
        )

    def broadcast(tensors, options):
        root = options.rootRank
        assert stream_events[-1] == "synchronize"
        stream_events.append("broadcast")
        roots.append(root)
        assert tensors[0].dtype == torch.int64
        if root == rank:
            torch.testing.assert_close(
                tensors[0], transport_view(original), atol=0, rtol=0
            )
        tensors[0].copy_(transport_view(values[root]))
        return SimpleNamespace(wait=lambda: True)

    group = SimpleNamespace(
        _inner=SimpleNamespace(broadcast=broadcast),
        _transport_view=transport_view,
        _ascend_double_reduction_warned=True,
        rank=lambda: rank,
        size=lambda: len(values),
    )
    monkeypatch.setattr(
        torch,
        "flagos",
        SimpleNamespace(synchronize=lambda device: stream_events.append("synchronize")),
        raising=False,
    )
    options = pg._c10d.AllreduceOptions()
    options.reduceOp = operation
    passed_operation = options.reduceOp if wrapped else operation
    work = pg.ProcessGroupFlagOS._allreduce_ascend_composite(
        group, output, passed_operation
    )
    assert work.wait()
    completed = work.get_future().wait()
    assert isinstance(completed, list) and len(completed) == 1
    assert completed[0].dtype == output.dtype
    expected = values.prod(0) if operation == dist.ReduceOp.PRODUCT else values.sum(0)
    if operation == dist.ReduceOp.AVG:
        expected /= len(values)
    torch.testing.assert_close(output, expected, atol=1e-14, rtol=1e-14)
    assert roots == list(range(len(values)))
    assert stream_events == ["synchronize", "broadcast"] * len(values) + ["synchronize"]


@pytest.mark.parametrize("operation", [dist.ReduceOp.MIN, dist.ReduceOp.MAX])
def test_complex_ordered_reduction_is_refused_before_transport(operation):
    with pytest.raises(RuntimeError, match="do not support complex"):
        pg.ProcessGroupFlagOS._allreduce_ascend_composite(
            None, torch.tensor([1j], dtype=torch.complex128), operation
        )


def test_bitwise_floating_reduction_is_refused_before_transport():
    with pytest.raises(RuntimeError, match="unsupported floating-point"):
        pg.ProcessGroupFlagOS._allreduce_ascend_composite(
            None, torch.ones(2, dtype=torch.float64), dist.ReduceOp.BAND
        )


def test_fp32_average_uses_sum_and_device_division(monkeypatch):
    original = torch.tensor([1.25, -3.5])
    actual = original.clone()

    def allreduce(tensors, options):
        assert getattr(options.reduceOp, "op", options.reduceOp) == dist.ReduceOp.SUM
        tensors[0].mul_(3)
        return SimpleNamespace(wait=lambda: True)

    group = SimpleNamespace(
        _inner=SimpleNamespace(allreduce=allreduce),
        _reduction_view=lambda tensor: tensor,
        size=lambda: 3,
    )
    monkeypatch.setattr(
        torch, "flagos", SimpleNamespace(synchronize=lambda device: None), raising=False
    )
    work = pg.ProcessGroupFlagOS._allreduce_ascend_composite(
        group, actual, dist.ReduceOp.AVG
    )
    assert work.wait()
    torch.testing.assert_close(actual, original, atol=0, rtol=0)


def test_integer_average_is_refused_before_transport():
    with pytest.raises(RuntimeError, match="AVG requires floating-point"):
        pg.ProcessGroupFlagOS._allreduce_ascend_composite(
            None, torch.ones(2, dtype=torch.int64), dist.ReduceOp.AVG
        )


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_transport_view_preserves_every_storage_bit(dtype):
    values = torch.tensor([1.00000000001, float("nan"), -0.0], dtype=dtype)
    group = SimpleNamespace(_ascend_double_transport=lambda tensor: True)
    actual = pg.ProcessGroupFlagOS._transport_view(group, values)
    expected = values.view(torch.float64).view(torch.int64)
    torch.testing.assert_close(actual.reshape(-1), expected.reshape(-1), atol=0, rtol=0)
    assert actual.data_ptr() == values.data_ptr()
