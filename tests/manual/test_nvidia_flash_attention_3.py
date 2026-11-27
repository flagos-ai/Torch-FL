# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare the optional FlashAttention 3 forward route with native CUDA."""

from contextlib import nullcontext
from importlib import import_module
from importlib.util import find_spec
from inspect import signature

import pytest
import torch
import torch_fl  # noqa: F401


@pytest.fixture(scope="module")
def flash_attention():
    if not torch.version.cuda or not torch.cuda.is_available():
        pytest.skip("FlashAttention 3 requires CUDA PyTorch on NVIDIA")
    if torch.cuda.get_device_capability(0)[0] != 9:
        pytest.skip("The FlashAttention 3 parity cases use Hopper")
    for package in ("flash_attn_3_nv", "flash_attn_3"):
        if find_spec(package) is not None:
            return import_module(f"{package}.flash_attn_interface")
    if find_spec("flash_attn_interface") is not None:
        return import_module("flash_attn_interface")
    pytest.skip("Install FlashAttention 3 to run CUDA parity")


@pytest.mark.parametrize("side_stream", [False, True], ids=["default", "side-stream"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("varlen", [False, True], ids=["dense", "packed-varlen"])
@torch.inference_mode()
def test_flash_attention_3_forward(flash_attention, varlen, dtype, causal, side_stream):
    rng = torch.Generator().manual_seed(42)
    inputs = [
        torch.randn((37, 4, 64), dtype=dtype, generator=rng),
        torch.randn((42, 2, 64), dtype=dtype, generator=rng),
        torch.randn((42, 2, 64), dtype=dtype, generator=rng),
    ]
    if varlen:
        attention = flash_attention.flash_attn_varlen_func
        cu_q = torch.tensor([0, 17, 37], dtype=torch.int32)
        cu_k = torch.tensor([0, 13, 42], dtype=torch.int32)
    else:
        attention = flash_attention.flash_attn_func
        inputs = [tensor.unsqueeze(0) for tensor in inputs]

    def run(device):
        options = {"causal": causal}
        if varlen:
            options.update(
                cu_seqlens_q=cu_q.to(device),
                cu_seqlens_k=cu_k.to(device),
                max_seqlen_q=20,
                max_seqlen_k=29,
            )
        if "return_attn_probs" in signature(attention).parameters:
            options["return_attn_probs"] = True
        return attention(*(tensor.to(device) for tensor in inputs), **options)

    expected = run("cuda:0")
    cuda_stream_before = torch._C._cuda_getCurrentStream(0)
    flagos = torch.device("flagos:0")
    with torch.Stream(device=flagos) if side_stream else nullcontext():
        actual = run(flagos)

    assert torch._C._cuda_getCurrentStream(0) == cuda_stream_before
    output, lse = actual
    for result, reference in zip((output, lse), expected):
        assert result.device == flagos
        assert result.stride() == reference.stride()
        assert result.dtype == reference.dtype
        assert torch.equal(
            result.cpu().contiguous().view(torch.uint8),
            reference.cpu().contiguous().view(torch.uint8),
        )


@torch.inference_mode()
def test_flash_attention_3_keeps_inputs_alive_across_streams(flash_attention):
    from torch_fl.flagos._streams import get_raw_stream

    attention = flash_attention.flash_attn_func
    shape = (1, 128, 4, 64)
    rng = torch.Generator().manual_seed(91)
    inputs = [torch.randn(shape, dtype=torch.bfloat16, generator=rng) for _ in range(3)]
    expected = attention(*(tensor.cuda() for tensor in inputs)).cpu()
    flagos = torch.device("flagos:0")
    main = torch.accelerator.current_stream(0)
    side = torch.Stream(device=flagos)
    transferred = [tensor.to(flagos) for tensor in inputs]
    side.wait_stream(main)
    with side:
        previous = torch._C._cuda_getCurrentStream(0)
        torch.cuda.set_stream(torch.cuda.ExternalStream(get_raw_stream(0), device=0))
        try:
            torch.cuda._sleep(20_000_000)
        finally:
            torch.cuda._set_stream_by_id(*previous)
        actual = attention(*transferred)
    del transferred
    replacements = [
        torch.zeros(shape, dtype=torch.bfloat16, device=flagos) for _ in inputs
    ]
    main.wait_stream(side)
    assert torch.equal(actual.cpu(), expected)
    assert all(not tensor.cpu().any() for tensor in replacements)
