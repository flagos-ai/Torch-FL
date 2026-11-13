# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the independent CoreX runtime after the common environment gate."""


def test_iluvatar_triton_and_flagos_add():
    # The CPU core must load before torch_fl's independent CoreX device runtime.
    import torch
    import torch_fl
    import triton
    import triton.language as tl

    assert torch_fl.flagos.is_available(), "CoreX device is unavailable"

    @triton.jit
    def add_kernel(x, y, output, size: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < size
        tl.store(
            output + offsets,
            tl.load(x + offsets, mask=mask, other=0)
            + tl.load(y + offsets, mask=mask, other=0),
            mask=mask,
        )

    x_cpu = torch.arange(1025, dtype=torch.float32)
    x = x_cpu.to("flagos")
    output = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 256),)](x, x, output, x.numel(), 256)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), x_cpu + x_cpu)
    value = torch.tensor([1.0, 2.0, 3.0], device="flagos")
    torch.testing.assert_close((value + value).cpu(), torch.tensor([2.0, 4.0, 6.0]))
