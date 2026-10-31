"""Krylov inner products and trace preserve complex views and gradients."""

import pytest
import torch_fl  # noqa: F401
import torch

pytestmark = pytest.mark.ascend
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [0, 1, 7])
def test_vdot_strided_lazy_conjugates(dtype, length):
    generator = torch.Generator().manual_seed(97)
    left = torch.randn(2 * length, dtype=dtype, generator=generator)[::2].conj()
    right = torch.randn(2 * length, dtype=dtype, generator=generator)[::2]
    actual = torch.vdot(left.to("flagos:0").conj().conj(), right.to("flagos:0"))
    tolerance = 3e-5 if dtype in {torch.float32, torch.complex64} else 1e-11
    torch.testing.assert_close(
        actual.cpu(), torch.vdot(left, right), atol=tolerance, rtol=tolerance
    )
    output = torch.empty((), dtype=dtype, device=actual.device)
    torch.vdot(left.to(output.device), right.to(output.device), out=output)
    torch.testing.assert_close(output.cpu(), actual.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vdot_gradient_matches_cpu(dtype):
    reference = torch.randn(
        5, dtype=dtype, generator=torch.Generator().manual_seed(101), requires_grad=True
    )
    tensor = reference.detach().to("flagos:0").requires_grad_()
    torch.vdot(reference, reference).real.backward()
    torch.vdot(tensor, tensor).real.backward()
    torch.testing.assert_close(tensor.grad.cpu(), reference.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [*DTYPES, torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(3, 4), (0, 3)])
def test_trace_and_out_match_cpu(dtype, shape):
    reference = torch.arange(shape[0] * shape[1]).reshape(shape).to(dtype).T
    actual = torch.trace(reference.to("flagos:0"))
    expected = torch.trace(reference)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    output = torch.empty((), dtype=expected.dtype, device=actual.device)
    torch.ops.aten.trace.out(reference.to(output.device), out=output)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


def test_reductions_reject_bool():
    tensor = torch.ones(2, dtype=torch.bool, device="flagos:0")
    with pytest.raises(RuntimeError):
        torch.vdot(tensor, tensor)
    with pytest.raises(RuntimeError):
        torch.trace(tensor.expand(2, 2))
