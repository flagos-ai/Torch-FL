# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check the Hopper FA3 wheel against a reference from stock CUDA PyTorch."""

from contextlib import nullcontext
from itertools import product
import os
from pathlib import Path
import sys

import pytest
import torch


_DTYPES = (torch.float16, torch.bfloat16)


def _provider():
    from flash_attn_3_nv import flash_attn_interface

    return flash_attn_interface


def _check_no_cuda_torch():
    mappings = Path("/proc/self/maps").read_text()
    for library in ("libtorch_cuda.so", "libc10_cuda.so"):
        assert library not in mappings


@pytest.fixture(scope="module")
def reference():
    path = os.environ.get("FLAGOS_FA3_CUDA_REFERENCE")
    if not path:
        pytest.skip("Set FLAGOS_FA3_CUDA_REFERENCE to a CUDA reference from this file")
    import torch_fl  # noqa: F401

    assert not torch._C._has_cuda
    assert not torch.ops.flagos.cpu_fallback_allowed()
    _check_no_cuda_torch()
    yield torch.load(path, weights_only=True)
    _check_no_cuda_torch()


def _forward(device, varlen, dtype, causal, side_stream):
    rng = torch.Generator().manual_seed(42)
    inputs = [
        torch.randn((37, 4, 64), dtype=dtype, generator=rng),
        torch.randn((42, 2, 64), dtype=dtype, generator=rng),
        torch.randn((42, 2, 64), dtype=dtype, generator=rng),
    ]
    provider = _provider()
    options = dict(causal=causal, return_attn_probs=True)
    with torch.Stream(device=device) if side_stream else nullcontext():
        if varlen:
            attention = provider.flash_attn_varlen_func
            options.update(
                cu_seqlens_q=torch.tensor([0, 17, 37], dtype=torch.int32).to(device),
                cu_seqlens_k=torch.tensor([0, 13, 42], dtype=torch.int32).to(device),
                max_seqlen_q=20,
                max_seqlen_k=29,
            )
        else:
            attention = provider.flash_attn_func
            inputs = [tensor.unsqueeze(0) for tensor in inputs]
        result = attention(*(tensor.to(device) for tensor in inputs), **options)
        assert all(tensor.device == device for tensor in result)
        return [(tensor.stride(), tensor.cpu()) for tensor in result]


def _assert_exact(actual, expected):
    assert len(actual) == len(expected)
    for (stride, value), (reference_stride, reference) in zip(actual, expected):
        assert stride == reference_stride
        assert value.dtype == reference.dtype
        assert value.shape == reference.shape
        assert torch.equal(
            value.contiguous().view(torch.uint8),
            reference.contiguous().view(torch.uint8),
        )


@pytest.mark.parametrize("side_stream", [False, True], ids=["default", "side-stream"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("varlen", [False, True], ids=["dense", "packed-varlen"])
@torch.inference_mode()
def test_flash_attention_3_forward(reference, varlen, dtype, causal, side_stream):
    device = torch.device("flagos:0")
    result = _forward(device, varlen, dtype, causal, side_stream)
    _assert_exact(result, reference[(varlen, str(dtype), causal, side_stream)])


def _lifetime_inputs():
    rng = torch.Generator().manual_seed(91)
    return [
        torch.randn((1, 128, 4, 64), dtype=torch.bfloat16, generator=rng)
        for _ in range(3)
    ]


@torch.inference_mode()
def test_flash_attention_3_keeps_inputs_alive_across_streams(reference):
    device = torch.device("flagos:0")
    main = torch.accelerator.current_stream(0)
    side = torch.Stream(device=device)
    inputs = [tensor.to(device) for tensor in _lifetime_inputs()]
    scratch = torch.ones((4096, 4096), device=device, dtype=torch.bfloat16)
    side.wait_stream(main)
    with side:
        delayed = scratch
        for _ in range(4):
            delayed = scratch @ delayed
        actual = _provider().flash_attn_func(*inputs)
    del inputs
    replacements = [
        torch.zeros((1, 128, 4, 64), dtype=torch.bfloat16, device=device)
        for _ in range(3)
    ]
    main.wait_stream(side)
    _assert_exact([(actual.stride(), actual.cpu())], reference["lifetime"])
    assert all(not tensor.cpu().any() for tensor in replacements)


@torch.inference_mode()
def _write_reference(output):
    assert torch._C._has_cuda and torch.cuda.get_device_capability(0)[0] == 9
    device = torch.device("cuda:0")
    values = {
        (varlen, str(dtype), causal, side): _forward(
            device, varlen, dtype, causal, side
        )
        for varlen, dtype, causal, side in product(
            (False, True), _DTYPES, (False, True), (False, True)
        )
    }
    lifetime = _provider().flash_attn_func(
        *(tensor.to(device) for tensor in _lifetime_inputs())
    )
    values["lifetime"] = [(lifetime.stride(), lifetime.cpu())]
    torch.save(values, output)


if __name__ == "__main__":
    _write_reference(Path(sys.argv[1]))
