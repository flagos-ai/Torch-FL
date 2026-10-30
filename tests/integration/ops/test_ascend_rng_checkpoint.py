"""RNG checkpoints retain native and FlagGems streams together."""

import pytest
import torch_fl
import torch

pytestmark = pytest.mark.ascend


@pytest.mark.parametrize("device_index", [0, 1])
def test_rng_state_restores_native_and_flaggems_draws(device_index):
    if device_index >= torch.flagos.device_count():
        pytest.skip("second device unavailable")
    device = f"flagos:{device_index}"
    torch.flagos.manual_seed_all(113)
    state = torch.flagos.get_rng_state(device)
    probabilities = torch.tensor([0.1, 0.2, 0, 0.7], device=device)

    def draws():
        return torch.multinomial(probabilities, 37, True).cpu(), torch.rand(
            37, device=device
        ).cpu()

    expected = draws()
    torch.flagos.set_rng_state(state, device)
    actual = draws()
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("operation", ["uniform", "normal", "rand", "randn"])
def test_explicit_rng_generator_is_reproducible(dtype, operation):
    def draw():
        generator = torch.Generator(device="flagos:0").manual_seed(127)
        if operation in {"rand", "randn"}:
            return getattr(torch, operation)(
                37, device="flagos:0", dtype=dtype, generator=generator
            )
        result = torch.empty(37, dtype=dtype, device="flagos:0")
        return getattr(result, operation + "_")(generator=generator)

    first = draw().cpu()
    torch.flagos.manual_seed_all(131)
    second = draw().cpu()
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_legacy_native_rng_state_is_still_accepted():
    generator = torch_fl._C._get_default_generator(0)
    state = generator.get_state()
    torch.flagos.set_rng_state(state, 0)
    torch.testing.assert_close(generator.get_state(), state, atol=0, rtol=0)
