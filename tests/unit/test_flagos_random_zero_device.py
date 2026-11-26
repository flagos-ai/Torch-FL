# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for zero-device FlagOS RNG seeding and generator bounds (#893)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types

import pytest


def _require_built_torch_fl():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required")
    try:
        import torch_fl  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Built torch_fl extension is required: {exc}")


def test_random_module_zero_device_unit_contract(monkeypatch):
    _require_built_torch_fl()
    import torch
    from torch_fl.flagos import random as flagos_random

    lazy_init_calls = []
    gen_calls = []

    monkeypatch.setattr(flagos_random, "device_count", lambda: 0)
    monkeypatch.setattr(flagos_random, "current_device", lambda: -1)
    monkeypatch.setattr(flagos_random, "_lazy_init", lambda: lazy_init_calls.append(1))

    def fake_get_default_generator(idx):
        gen_calls.append(idx)
        raise RuntimeError(
            "Cannot get default flagos generator when no flagos devices are available."
        )

    monkeypatch.setattr(
        flagos_random._C, "_get_default_generator", fake_get_default_generator
    )

    # Seeding APIs must be safe silent no-ops when 0 devices are visible.
    assert flagos_random.manual_seed(1) is None
    assert flagos_random.manual_seed_all(1) is None
    assert flagos_random.seed() is None
    assert flagos_random.seed_all() is None
    assert flagos_random.get_rng_state_all() == []
    assert flagos_random.set_rng_state_all([]) is None
    assert gen_calls == []
    assert lazy_init_calls == []

    # Invalid seed types still fail fast even when 0 devices are visible.
    with pytest.raises(ValueError):
        flagos_random.manual_seed("not-an-int")  # type: ignore[arg-type]

    # Querying state or initial_seed with 0 devices must not call _lazy_init
    # and must raise a clean RuntimeError instead of crashing.
    with pytest.raises(RuntimeError, match="no flagos devices are available"):
        flagos_random.initial_seed()
    assert lazy_init_calls == []
    assert gen_calls == [-1]

    with pytest.raises(RuntimeError, match="no flagos devices are available"):
        flagos_random.get_rng_state()
    with pytest.raises(RuntimeError, match="no flagos devices are available"):
        flagos_random.get_rng_state(0)
    with pytest.raises(RuntimeError, match="no flagos devices are available"):
        flagos_random.set_rng_state(torch.empty(0, dtype=torch.uint8))

    # Explicit negative int indices and foreign device types are rejected cleanly.
    with pytest.raises(RuntimeError, match="Device index must not be negative"):
        flagos_random.get_rng_state(-1)
    with pytest.raises(RuntimeError, match="Device index must not be negative"):
        flagos_random.set_rng_state(torch.empty(0, dtype=torch.uint8), -1)
    with pytest.raises(ValueError, match="Expected a 'flagos' device"):
        flagos_random.get_rng_state("cpu")


def test_random_module_multi_device_and_bridge_propagation(monkeypatch):
    _require_built_torch_fl()
    import torch
    import torch_fl
    from torch_fl.flagos import random as flagos_random

    class FakeGen:
        def __init__(self, idx: int):
            self.idx = idx
            self.last_seed: int | None = None
            self.state = torch.tensor([idx, 0], dtype=torch.uint8)

        def manual_seed(self, s: int):
            self.last_seed = s
            return self

        def seed(self) -> int:
            self.last_seed = 7000 + self.idx
            return self.last_seed

        def initial_seed(self) -> int:
            return self.last_seed or 0

        def get_state(self) -> torch.Tensor:
            return self.state.clone()

        def set_state(self, new_state: torch.Tensor) -> None:
            self.state = new_state.clone()

    gens = [FakeGen(0), FakeGen(1)]
    bridged_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(flagos_random, "device_count", lambda: 2)
    monkeypatch.setattr(flagos_random, "current_device", lambda: 1)
    monkeypatch.setattr(flagos_random, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        flagos_random._C, "_get_default_generator", lambda idx: gens[idx]
    )
    monkeypatch.setattr(
        torch_fl.flagos,
        "manual_seed",
        lambda s: bridged_calls.append(("seed", s)),
    )
    monkeypatch.setattr(
        torch_fl.flagos,
        "manual_seed_all",
        lambda s: bridged_calls.append(("seed_all", s)),
    )
    monkeypatch.setattr(torch_fl.flagos, "get_rng_state", flagos_random.get_rng_state)
    monkeypatch.setattr(torch_fl.flagos, "set_rng_state", flagos_random.set_rng_state)

    flagos_random.manual_seed(42)
    assert gens[0].last_seed is None
    assert gens[1].last_seed == 42

    flagos_random.manual_seed_all(99)
    assert [g.last_seed for g in gens] == [99, 99]

    flagos_random.seed()
    assert gens[1].last_seed == 7001
    assert bridged_calls[-1] == ("seed", 7001)

    flagos_random.seed_all()
    assert [g.last_seed for g in gens] == [7000, 7000]
    assert bridged_calls[-1] == ("seed_all", 7000)

    states = flagos_random.get_rng_state_all()
    assert len(states) == 2
    replacement = [
        torch.tensor([10, 1], dtype=torch.uint8),
        torch.tensor([20, 2], dtype=torch.uint8),
    ]
    flagos_random.set_rng_state_all(replacement)
    assert [g.state.tolist() for g in gens] == [[10, 1], [20, 2]]
    assert isinstance(flagos_random._is_in_bad_fork(), bool)
    assert isinstance(types.SimpleNamespace(), object)


def test_cpp_default_generator_bounds_check():
    _require_built_torch_fl()
    import torch_fl

    count = torch_fl.flagos.device_count()
    for bad_idx in (count, count + 5, -2):
        with pytest.raises(RuntimeError):
            torch_fl._C._get_default_generator(bad_idx)


def test_vendor_rng_device_index_bounds_and_zero_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required")
    from torch_fl.accelerator.hygon import _dcu_compat
    from torch_fl.accelerator.metax import _metax_compat
    from torch_fl.accelerator.nvidia import _cuda_compat

    for mod in (_cuda_compat, _metax_compat, _dcu_compat):
        helper = mod._resolve_rng_device_index
        assert helper("flagos", lambda: 1, lambda: 2) == 1
        assert helper(None, lambda: 1, lambda: 2) == 1
        assert helper("flagos:0", lambda: 1, lambda: 2) == 0
        assert helper("cuda:1", lambda: 0, lambda: 2) == 1
        assert helper(1, lambda: 0, lambda: 2) == 1

        with pytest.raises(RuntimeError, match="Invalid device index 2 for 2"):
            helper(2, lambda: 0, lambda: 2)
        with pytest.raises(RuntimeError, match="Device index must not be negative"):
            helper(-1, lambda: 0, lambda: 2)
        with pytest.raises(ValueError, match="Expected a 'flagos' device"):
            helper("cpu", lambda: 0, lambda: 2)
        with pytest.raises(RuntimeError, match="no flagos devices are available"):
            helper("flagos", lambda: -1, lambda: 0)
        with pytest.raises(RuntimeError, match="no flagos devices are available"):
            helper(0, lambda: -1, lambda: 0)


def test_nvidia_and_metax_patch_flagos_rng_state_bounds(monkeypatch):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required")
    import ctypes
    import torch
    from torch_fl.accelerator.metax import _metax_compat
    from torch_fl.accelerator.nvidia import _cuda_compat

    # 1. Verify NVIDIA patch_torch_cuda_for_flagos() enforces bounds & current_device
    class FakeCudaLib:
        def cuDeviceGetCount(self, ptr):
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = 2
            return 0

    fake_flagos_nv = types.SimpleNamespace(
        device_count=lambda: 2,
        current_device=lambda: 1,
        set_device=lambda idx: None,
        Event=None,
        Stream=None,
        memory_allocated=lambda idx=0: 0,
        memory_reserved=lambda idx=0: 0,
        memory_stats=lambda idx=0: {},
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda idx=0: None,
        manual_seed=lambda s: None,
        manual_seed_all=lambda s: None,
    )
    fake_cuda_nv = types.SimpleNamespace(
        device=type("FakeDeviceCtx", (), {"__init__": lambda self, d: None}),
        default_generators=(),
    )
    fake_torch_nv = types.SimpleNamespace(
        cuda=fake_cuda_nv,
        flagos=fake_flagos_nv,
        _C=types.SimpleNamespace(),
        device=torch.device,
        tensor=torch.tensor,
        int64=torch.int64,
        uint8=torch.uint8,
        initial_seed=lambda: 12345,
    )
    monkeypatch.setattr(_cuda_compat, "torch", fake_torch_nv)
    monkeypatch.setattr(_cuda_compat, "_patched", False)
    monkeypatch.setattr(_cuda_compat, "_cuda_generators", {})
    monkeypatch.setattr(_cuda_compat, "_boxing_built", lambda: False)
    monkeypatch.setattr(_cuda_compat, "is_nvidia_cuda_available", lambda: True)
    monkeypatch.setattr(_cuda_compat, "_load_libcuda", lambda: FakeCudaLib())
    monkeypatch.setattr(_cuda_compat, "_patch_triton_do_bench", lambda: None)

    assert _cuda_compat.patch_torch_cuda_for_flagos()
    fake_flagos_nv.manual_seed_all(100)
    fake_flagos_nv.manual_seed(202)
    assert not torch.equal(
        fake_flagos_nv.get_rng_state(0), fake_flagos_nv.get_rng_state()
    )
    with pytest.raises(RuntimeError, match="Invalid device index 2 for 2"):
        fake_flagos_nv.get_rng_state(2)
    with pytest.raises(RuntimeError, match="Device index must not be negative"):
        fake_flagos_nv.get_rng_state(-1)
    with pytest.raises(RuntimeError, match="Invalid device index 2 for 2"):
        fake_flagos_nv.set_rng_state(fake_flagos_nv.get_rng_state(0), 2)

    # 2. Verify MetaX patch_torch_cuda_for_metax() enforces bounds & zero-device safety
    dev_count = [2]
    fake_flagos_mx = types.SimpleNamespace(
        device_count=lambda: dev_count[0],
        current_device=lambda: 1 if dev_count[0] > 0 else -1,
        set_device=lambda idx: None,
        Event=object(),
        manual_seed=lambda s: None,
        manual_seed_all=lambda s: None,
    )
    fake_cuda_mx = types.SimpleNamespace(device_count=lambda: 0, default_generators=())
    fake_torch_mx = types.SimpleNamespace(
        cuda=fake_cuda_mx,
        flagos=fake_flagos_mx,
        _C=types.SimpleNamespace(),
        device=torch.device,
    )
    monkeypatch.setattr(_metax_compat, "torch", fake_torch_mx)
    monkeypatch.setattr(_metax_compat, "_patched", False)
    monkeypatch.setattr(_metax_compat, "_cuda_generators", {})
    monkeypatch.setattr(
        _metax_compat,
        "_get_cuda_generator",
        lambda idx: _cuda_compat._CudaPhiloxGenerator(10 + idx),
    )
    monkeypatch.setattr(_metax_compat, "_find_metax_path", lambda: "/maca")
    monkeypatch.setattr(_metax_compat, "_load_mcruntime", lambda p: object())
    monkeypatch.setattr(_metax_compat, "_patch_cuda_tensor_ops", lambda p: None)
    monkeypatch.setattr(_metax_compat, "_patch_triton_do_bench", lambda: None)

    assert _metax_compat.patch_torch_cuda_for_metax()
    with pytest.raises(RuntimeError, match="Invalid device index 2 for 2"):
        fake_flagos_mx.get_rng_state(2)
    with pytest.raises(RuntimeError, match="Device index must not be negative"):
        fake_flagos_mx.get_rng_state(-1)
    dev_count[0] = 0
    assert fake_flagos_mx.manual_seed(7) is None
    assert fake_flagos_mx.manual_seed_all(7) is None
    with pytest.raises(RuntimeError, match="no flagos devices are available"):
        fake_flagos_mx.get_rng_state()


def test_zero_visible_devices_subprocess_no_segfault():
    _require_built_torch_fl()
    env = os.environ.copy()
    for var in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "MACA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "MUSA_VISIBLE_DEVICES",
        "MTHREADS_VISIBLE_DEVICES",
        "COREX_VISIBLE_DEVICES",
        "TOPS_VISIBLE_DEVICES",
        "PPU_VISIBLE_DEVICES",
        "ALIXPU_VISIBLE_DEVICES",
        "MLU_VISIBLE_DEVICES",
        "XPURT_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
    ):
        env[var] = ""
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    code = """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import torch_fl
import torch

count = torch.flagos.device_count()
if count != 0:
    print(f"SKIP_MASK_UNSUPPORTED:{count}")
    raise SystemExit(0)

# Seeding & empty collection operations must succeed cleanly without SIGSEGV.
torch.flagos.manual_seed(1)
torch.flagos.manual_seed_all(1)
torch.flagos.seed()
torch.flagos.seed_all()
torch.manual_seed(1)
assert torch.flagos.get_rng_state_all() == []
torch.flagos.set_rng_state_all([])
assert len(torch.flagos.default_generators) == 0
assert list(torch.flagos.default_generators) == []
with torch.random.fork_rng(device_type="flagos"):
    pass

# Direct generator/state queries with 0 devices must raise RuntimeError, not SIGSEGV.
for fn in (
    lambda: torch_fl._C._get_default_generator(0),
    lambda: torch_fl._C._get_default_generator(-1),
    lambda: torch.flagos.get_rng_state(),
    lambda: torch.flagos.get_rng_state(0),
    lambda: torch.flagos.set_rng_state(torch.empty(0, dtype=torch.uint8)),
    lambda: torch.flagos.initial_seed(),
):
    try:
        fn()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError when 0 flagos devices are visible")

try:
    _ = torch.flagos.default_generators[0]
except IndexError:
    pass
else:
    raise AssertionError("Expected IndexError from empty default_generators[0]")

print("ZERO_DEVICE_RNG_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    if result.stdout.strip().startswith("SKIP_MASK_UNSUPPORTED:"):
        pytest.skip(
            f"Runtime does not mask devices via env vars ({result.stdout.strip()})"
        )
    assert "ZERO_DEVICE_RNG_OK" in result.stdout
