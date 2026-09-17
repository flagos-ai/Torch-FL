"""Real MTT S5000 coverage for the MThreads FlagGems hybrid path."""

import importlib

import pytest
import torch
import torch_fl


pytestmark = pytest.mark.musa
DEVICE = torch.device("flagos:0")


def _require_flaggems_mthreads():
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("MUSA device is unavailable")
    # The wheel decides whether the hybrid path exists, not the shell it runs
    # in: this used to be an exported FLAGOS_USE_FLAGGEMS, which nothing read
    # once that switch was retired -- so the gate never opened and the file
    # silently measured nothing. The build record cannot disagree with the
    # wheel it is inside.
    from torch_fl import _env

    if "flaggems" not in _env.build_kernels():
        pytest.skip(
            "this wheel was not built with the FlagGems kernels (FLAGOS_BUILD_FLAGGEMS=ON)"
        )
    try:
        import triton
        import flag_gems
    except Exception as exc:
        pytest.skip(f"FlagGems MThreads runtime is unavailable: {exc}")
    if "mthreads" not in triton.backends.backends:
        pytest.skip("the installed Triton does not provide the MThreads backend")
    flag_gems.enable()


def test_selected_flaggems_routes_execute_on_s5000(monkeypatch):
    _require_flaggems_mthreads()
    calls = []

    def track(qualname):
        """Patch the entry point the generated kernel actually calls.

        The codegen freezes the *package-level* name -- ``flag_gems.<fn>``, see
        ``_normalize_flaggems_qualname`` in scripts/codegen_ops.py -- and
        ``PythonOpCache::GetFunc`` resolves it by importing the prefix and taking
        the attribute. So the observable call site is ``flag_gems.<fn>``.
        Patching ``flag_gems.ops.<module>.<fn>`` instead writes to a different
        attribute slot that happens to hold the same object, and the wrapper is
        never entered.
        """
        module_path, _, function_name = qualname.rpartition(".")
        module = importlib.import_module(module_path)
        original = getattr(module, function_name)

        def wrapper(*args, **kwargs):
            calls.append(function_name)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, function_name, wrapper)

    track("flag_gems.all")
    track("flag_gems.all_dims")
    track("flag_gems.any")
    track("flag_gems.any_dims")
    track("flag_gems.repeat_interleave_tensor")
    track("flag_gems.index_add")
    track("flag_gems.index_add_")

    values = torch.tensor([[1, 0, 1], [1, 1, 1]], device=DEVICE)
    assert torch.equal(torch.all(values).cpu(), torch.all(values.cpu()))
    assert torch.equal(
        torch.all(values, dim=(1,)).cpu(), torch.all(values.cpu(), dim=(1,))
    )
    assert torch.equal(torch.any(values).cpu(), torch.any(values.cpu()))
    assert torch.equal(
        torch.any(values, dim=(1,)).cpu(), torch.any(values.cpu(), dim=(1,))
    )

    repeats = torch.tensor([1, 2, 1], device=DEVICE)
    assert torch.equal(
        torch.repeat_interleave(repeats).cpu(),
        torch.repeat_interleave(repeats.cpu()),
    )

    base = torch.zeros(4, 3, device=DEVICE)
    index = torch.tensor([1, 1, 3], device=DEVICE)
    source = torch.ones(3, 3, device=DEVICE)
    expected = torch.index_add(base.cpu(), 0, index.cpu(), source.cpu())
    assert torch.equal(torch.index_add(base, 0, index, source).cpu(), expected)

    inplace = torch.zeros(4, 3, device=DEVICE)
    inplace.index_add_(0, index, source)
    assert torch.equal(inplace.cpu(), expected)
    assert calls == [
        "all",
        "all_dims",
        "any",
        "any_dims",
        "repeat_interleave_tensor",
        "index_add",
        "index_add_",
    ]


def test_flaggems_randn_shares_native_generator_reservations():
    _require_flaggems_mthreads()
    import flag_gems

    # The package-level name, not ``flag_gems.ops.randn``: that is what the
    # generated kernel resolves to, and on MUSA ``SpecOpRegistrar`` has rebound
    # it to the vendor override ``_mthreads.ops.randn``. Driving the generic
    # module here would exercise a callable the dispatch never reaches.
    flaggems_randn = flag_gems.randn

    def run(seed):
        torch.flagos.manual_seed(seed)
        native_before = torch.rand(64, device=DEVICE)
        flaggems = flaggems_randn((64,), device=DEVICE)
        assert flaggems.device == DEVICE
        native_after = torch.rand(64, device=DEVICE)
        torch.flagos.synchronize()
        return native_before.cpu(), flaggems.cpu(), native_after.cpu()

    first = run(20260817)
    second = run(20260817)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.isfinite(first[1]).all()
    assert first[1].device.type == "cpu"

    torch.flagos.manual_seed(20260817)
    initial_state = torch.flagos.get_rng_state()
    torch.rand(64, device=DEVICE)
    flaggems_randn((64,), device=DEVICE)
    mixed_state = torch.flagos.get_rng_state()

    torch.flagos.set_rng_state(initial_state)
    torch_fl._C._reserve_rng_seed(0)
    torch_fl._C._reserve_rng_seed(0)
    expected_state = torch.flagos.get_rng_state()
    assert torch.equal(mixed_state, expected_state)
