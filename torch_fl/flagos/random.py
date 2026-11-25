# Copyright (c) 2026, BAAI. All rights reserved.
#
# Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
# with "openreg" device name replaced with "flagos" and torch_openreg._C replaced with torch_fl._C.
# Below is the original copyright:
# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from collections.abc import Iterable
import sys

import torch

from .. import _C  # type: ignore[import-untyped]

from . import _lazy_init, current_device, device_count


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
    "_is_in_bad_fork",
]


def _resolve_device_index(device: int | str | torch.device) -> int:
    if isinstance(device, int) and not isinstance(device, bool):
        if device < 0:
            raise RuntimeError(f"Device index must not be negative, got {device}")
        return device
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("flagos", device)
    if device.type not in ("flagos", "privateuseone"):
        raise ValueError(f"Expected a 'flagos' device, but got {device}")
    if device.index is None:
        return current_device()
    return device.index


def get_rng_state(device: int | str | torch.device = "flagos") -> torch.Tensor:
    idx = _resolve_device_index(device)
    default_generator = _C._get_default_generator(idx)
    return default_generator.get_state()


def get_rng_state_all() -> list[torch.Tensor]:
    flagos_mod = sys.modules.get(__package__)
    getter = getattr(flagos_mod, "get_rng_state", get_rng_state)
    return [getter(idx) for idx in range(device_count())]


def set_rng_state(
    new_state: torch.Tensor, device: int | str | torch.device = "flagos"
) -> None:
    idx = _resolve_device_index(device)
    default_generator = _C._get_default_generator(idx)
    default_generator.set_state(new_state)


def set_rng_state_all(new_states: Iterable[torch.Tensor]) -> None:
    flagos_mod = sys.modules.get(__package__)
    setter = getattr(flagos_mod, "set_rng_state", set_rng_state)
    for idx, state in enumerate(new_states):
        setter(state, idx)


def initial_seed() -> int:
    if device_count() > 0:
        _lazy_init()
    idx = current_device()
    default_generator = _C._get_default_generator(idx)
    return default_generator.initial_seed()


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current FlagOS device.

    It is safe to call this function if no FlagOS device is available; in that
    case, it is silently ignored.
    """
    seed = int(seed)
    if device_count() == 0:
        return

    idx = current_device()
    default_generator = _C._get_default_generator(idx)
    default_generator.manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all FlagOS devices.

    It is safe to call this function if no FlagOS device is available; in that
    case, it is silently ignored.
    """
    seed = int(seed)
    for idx in range(device_count()):
        default_generator = _C._get_default_generator(idx)
        default_generator.manual_seed(seed)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number on the current FlagOS device.

    It is safe to call this function if no FlagOS device is available; in that
    case, it is silently ignored.
    """
    if device_count() == 0:
        return

    idx = current_device()
    default_generator = _C._get_default_generator(idx)
    random_seed = default_generator.seed()
    flagos_mod = sys.modules.get(__package__)
    if (
        flagos_mod is not None
        and getattr(flagos_mod, "manual_seed", None) is not manual_seed
    ):
        flagos_mod.manual_seed(random_seed)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all FlagOS devices.

    It is safe to call this function if no FlagOS device is available; in that
    case, it is silently ignored.
    """
    count = device_count()
    if count == 0:
        return

    random_seed = 0
    seeded = False
    for idx in range(count):
        default_generator = _C._get_default_generator(idx)
        if not seeded:
            random_seed = default_generator.seed()
            seeded = True
        else:
            default_generator.manual_seed(random_seed)
    flagos_mod = sys.modules.get(__package__)
    if (
        flagos_mod is not None
        and getattr(flagos_mod, "manual_seed_all", None) is not manual_seed_all
    ):
        flagos_mod.manual_seed_all(random_seed)


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
