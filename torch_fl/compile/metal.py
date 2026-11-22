# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Inductor's Metal scheduling with FlagOS buffers and stream submission."""

from functools import wraps
from pathlib import Path

import torch
from torch._inductor.codegen.common import get_scheduling_for_device
from torch._inductor.codegen.mps import MetalScheduling as UpstreamMetalScheduling
from torch._inductor.virtualized import V
from torch.utils._cpp_embed_headers import _embed_headers

from torch_fl import _C


def compile_shader(source):
    include = Path(torch.__file__).resolve().parent / "include"
    return _C._MetalKernel(
        _embed_headers(source.splitlines(keepends=True), [include], set())
    )


class MetalScheduling(UpstreamMetalScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        if (wrapper := V.graph.wrapper_code) is not None:
            wrapper.header.writeline(
                "from torch_fl.compile.metal import compile_shader as compile_flagos_shader"
            )

    def define_kernel(self, src_code, node_schedule, kernel):
        prefix = "compile_mps_shader("
        if not src_code.startswith(prefix):
            raise RuntimeError("Unsupported Inductor Metal kernel wrapper")
        source = "compile_flagos_shader(" + src_code.removeprefix(prefix)
        return super().define_kernel(source, node_schedule, kernel)


def install_scheduler():
    from torch._inductor.scheduler import Scheduler

    original = Scheduler.create_backend
    if getattr(original, "_flagos_metal", False):
        return

    @wraps(original)
    def create_backend(scheduler, device):
        # Inductor 2.10's Triton requirement exempts its built-in Metal device.
        # Apply the same requirement to the registered FlagOS Metal scheduler.
        if (
            device.type == "flagos"
            and get_scheduling_for_device("flagos") is MetalScheduling
        ):
            assert device.index is not None, (
                f"Expected a normalized device, got {device}"
            )
            V.graph.add_device_info(device)
            return MetalScheduling(scheduler)
        return original(scheduler, device)

    create_backend._flagos_metal = True
    Scheduler.create_backend = create_backend
