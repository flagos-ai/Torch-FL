"""Unit coverage for the DCU CUDA/flagos RNG state bridge."""

import torch

import torch_fl
from torch_fl.accelerator.dcu import _dcu_compat


def test_dcu_rng_bridge_forwards_seed_and_state(monkeypatch):
    calls = []
    states = [torch.tensor([11, 0], dtype=torch.int64).view(torch.uint8)]

    class Generator:
        def get_state(self):
            return states[0]

        def set_state(self, state):
            states[0] = state

    generators = [Generator()]

    def cuda_manual_seed(seed):
        calls.append(("seed", seed))

    def cuda_manual_seed_all(seed):
        calls.append(("all", seed))

    monkeypatch.setattr(torch.cuda, "default_generators", generators)
    monkeypatch.setattr(torch.cuda, "manual_seed", cuda_manual_seed)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_manual_seed_all)
    monkeypatch.setattr(_dcu_compat, "_patched", False)
    monkeypatch.setattr(
        torch_fl.flagos, "default_generators", torch_fl.flagos.default_generators
    )
    monkeypatch.setattr(torch_fl.flagos, "get_rng_state", torch_fl.flagos.get_rng_state)
    monkeypatch.setattr(torch_fl.flagos, "set_rng_state", torch_fl.flagos.set_rng_state)
    monkeypatch.setattr(
        torch_fl.flagos, "manual_seed", lambda seed: calls.append(("native", seed))
    )
    monkeypatch.setattr(
        torch_fl.flagos,
        "manual_seed_all",
        lambda seed: calls.append(("native_all", seed)),
    )

    assert _dcu_compat.install_dcu_rng_bridge()
    torch_fl.flagos.manual_seed(17)
    torch_fl.flagos.manual_seed_all(23)
    state = torch_fl.flagos.get_rng_state()
    replacement = torch.tensor([29, 4], dtype=torch.int64).view(torch.uint8)
    torch_fl.flagos.set_rng_state(replacement)

    assert calls == [("native", 17), ("seed", 17), ("native_all", 23), ("all", 23)]
    assert torch.equal(
        state, torch.tensor([11, 0], dtype=torch.int64).view(torch.uint8)
    )
    assert torch.equal(torch_fl.flagos.get_rng_state(), replacement)
    assert torch_fl.flagos.default_generators is generators


def test_dcu_rng_bridge_accepts_device_forms(monkeypatch):
    class Generator:
        def get_state(self):
            return torch.tensor([7, 0], dtype=torch.int64).view(torch.uint8)

        def set_state(self, state):
            self.state = state

    generators = [Generator(), Generator()]
    monkeypatch.setattr(torch.cuda, "default_generators", generators)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(_dcu_compat, "_patched", False)
    monkeypatch.setattr(
        torch_fl.flagos, "default_generators", torch_fl.flagos.default_generators
    )
    monkeypatch.setattr(torch_fl.flagos, "get_rng_state", torch_fl.flagos.get_rng_state)
    monkeypatch.setattr(torch_fl.flagos, "set_rng_state", torch_fl.flagos.set_rng_state)
    monkeypatch.setattr(torch_fl.flagos, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch_fl.flagos, "manual_seed_all", lambda seed: None)

    assert _dcu_compat.install_dcu_rng_bridge()
    assert torch.equal(
        torch_fl.flagos.get_rng_state("flagos:1"),
        torch.tensor([7, 0], dtype=torch.int64).view(torch.uint8),
    )
    assert torch.equal(
        torch_fl.flagos.get_rng_state(torch.device("flagos:1")),
        torch.tensor([7, 0], dtype=torch.int64).view(torch.uint8),
    )
