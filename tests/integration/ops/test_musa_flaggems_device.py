"""MUSA hardware behaviour of the FlagGems route.

The device half of the MThreads FlagGems coverage that used to live in
``test_musa_flaggems.py``. That file's route-execution case asserted the FlagGems
integration itself and is now expressed once for every platform by
``tests/integration/flaggems/``; what is left here needs an MTT S5000 and cannot
be restated generically -- the float64 ``mm`` tune-space limit, the shared
generator the FlagGems and native RNG entry points both draw from.

The file carries the same ``flaggems``/``main_ops`` marks as the platform-neutral
FlagGems group, so it is collected by the identical ``-m "flaggems and main_ops"``
step on every platform and skips itself off MUSA, rather than needing a step of
its own.
"""

import pytest
import torch
import torch_fl


pytestmark = [pytest.mark.musa, pytest.mark.flaggems, pytest.mark.main_ops]
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


def test_float64_mm_is_exact_across_the_tile_boundary():
    """The MUSA tune-space regression: float64 ``mm`` on this hardware.

    The mthreads ``mm`` entry declares two 64x64x64 tiles at ``num_stages`` 5 and
    4, which at 8 bytes per element ask 262144 B and 196608 B against the
    196608 B static shared-memory limit, so before the fit every float64 shape
    whose M, K and N all exceed 32 failed -- the first configuration with
    ``OutOfResources: Required: 262144, Hardware limit: 196608`` from inside the
    benchmark, the second with ``Triton Error [MUSA]: invalid argument`` from the
    final launch. The same space on float32 asks half as much and was never
    affected, which is what makes the defect look shape- and dtype-dependent.

    32 in every dimension is the boundary, and both sides of it are asserted:
    FlagTree's AABS narrows the declared tile to the extent before the compiler
    sizes it, so a small shape stayed inside the limit by accident. Exactness is
    the assertion rather than "it runs", because the failure this guards against
    is a configuration that is silently wrong, not only one that raises.
    """
    _require_flaggems_mthreads()

    for m, k, n in [
        (31, 32, 32),
        (32, 32, 32),
        (33, 32, 32),
        (33, 33, 33),  # first shape the declared space cannot serve on float64
        (64, 64, 64),  # both declared configurations out of limit on float64
        (32, 64, 64),
        (64, 64, 32),
        (512, 512, 512),
    ]:
        a = torch.randn(m, k, dtype=torch.float64)
        b = torch.randn(k, n, dtype=torch.float64)
        got = torch.mm(a.to(DEVICE), b.to(DEVICE)).cpu()
        torch.testing.assert_close(
            got,
            torch.mm(a, b),
            rtol=1e-12,
            atol=1e-12,
            msg=f"float64 mm at {m}x{k}x{n}",
        )


def test_the_space_the_tuner_is_given_fits_the_device():
    """Every configuration in the space the tuner ends up launching is inside the
    shared-memory limit, measured on this device through the installed patch.

    The fit is spied on rather than ``LibTuner.run``: the wrapper hands the
    declared list back before it returns, so a repaired space is visible only on
    the way in, and the declared list on the way in is what says whether the
    refusal this shape raises was computed to be repairable. The arithmetic is
    written out here rather than called from the module under test, so "fits" is
    not being asserted in the fix's own terms.
    """
    _require_flaggems_mthreads()
    limit = torch_fl.flagos._musa_shared_memory_limit()
    if limit is None:
        pytest.skip("this backend does not report a static shared-memory limit")

    seen = []
    fit = torch_fl.flagos._fitted_tune_space

    def spy(configs, args, kwargs, limit_bytes):
        fitted = fit(configs, args, kwargs, limit_bytes)
        seen.append((list(configs), fitted, limit_bytes))
        return fitted

    torch_fl.flagos._fitted_tune_space = spy
    try:
        # A shape no other test in the session uses, so the ConfigCache cannot
        # answer with a configuration tuned before the spy was installed.
        a = torch.randn(97, 95, dtype=torch.float64)
        b = torch.randn(95, 99, dtype=torch.float64)
        got = torch.mm(a.to(DEVICE), b.to(DEVICE)).cpu()
    finally:
        torch_fl.flagos._fitted_tune_space = fit

    torch.testing.assert_close(got, torch.mm(a, b), rtol=1e-12, atol=1e-12)

    # float64 operands throughout, hence 8 bytes per element.
    def ask(config):
        return (
            (config.num_stages - 1)
            * (
                config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_K"]
                + config.kwargs["BLOCK_K"] * config.kwargs["BLOCK_N"]
            )
            * 8
        )

    assert seen, "the mm route never sized a tune space"

    for declared, fitted, limit_bytes in seen:
        assert limit_bytes == limit
        if any(ask(config) >= limit_bytes for config in declared):
            assert fitted is not None, (
                "a declared configuration asks more than the device can give and"
                f" was handed over unchanged: {[ask(c) for c in declared]}"
            )
        for config in declared if fitted is None else fitted:
            assert ask(config) < limit_bytes, (
                f"num_stages={config.num_stages} on {config.kwargs} asks"
                f" {ask(config)} B against a {limit_bytes} B limit"
            )
