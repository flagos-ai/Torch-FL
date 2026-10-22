# Copyright (c) 2026, BAAI. All rights reserved.
#
# Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
# with "openreg" device name replaced with "flagos" and torch_openreg._C replaced with torch_fl._C.
# Below is the original copyright:
# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

from .. import _C  # type: ignore[import-untyped]

from . import _lazy_init, current_device, device_count


__all__ = [
    "get_rng_state",
    "set_rng_state",
    "manual_seed",
    "manual_seed_all",
    "initial_seed",
    "_is_in_bad_fork",
]


def get_rng_state(device="flagos"):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("flagos", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = _C._get_default_generator(idx)
    return default_generator.get_state()


def set_rng_state(new_state, device="flagos"):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("flagos", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = _C._get_default_generator(idx)
    default_generator.set_state(new_state)


def initial_seed() -> int:
    _lazy_init()
    idx = current_device()
    default_generator = _C._get_default_generator(idx)
    return default_generator.initial_seed()


def manual_seed(seed: int) -> None:
    seed = int(seed)

    idx = current_device()
    default_generator = _C._get_default_generator(idx)
    default_generator.manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    seed = int(seed)

    for idx in range(device_count()):
        default_generator = _C._get_default_generator(idx)
        default_generator.manual_seed(seed)


def _is_in_bad_fork() -> bool:
    """Required, alongside `manual_seed_all`, by torch.random._seed_custom_device.

    Without it `torch.manual_seed()` skips this device module entirely and warns
    "Set seed for `flagos` device does not take effect" on every call -- which is
    misleading twice over: the flagos RNG *is* seeded (the flaggems and native
    CUDA paths both go through torch.cuda.manual_seed_all, patched by the vendor
    shim), and the PrivateUse1 generator this module owns was simply never
    reached. Defining it wires torch.manual_seed to that generator too, so
    flagos.get_rng_state()/initial_seed() track the global seed.

    flagos generators live in host memory and carry no fork-unsafe device
    context (unlike CUDA's, which is why torch.cuda has a real check), so there
    is no bad-fork state to detect.
    """
    return False
