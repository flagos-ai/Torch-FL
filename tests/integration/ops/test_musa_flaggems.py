"""Real MTT S5000 coverage for the MThreads FlagGems hybrid path."""

import importlib

import pytest
import torch
import torch_fl


pytestmark = pytest.mark.musa
DEVICE = torch.device("flagos:0")


def _require_flaggems_mthreads():
    """Require the runtime without replacing torch_fl's dispatch kernels."""
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
        import flag_gems  # noqa: F401 -- verify availability without registering ops
    except Exception as exc:
        pytest.skip(f"FlagGems MThreads runtime is unavailable: {exc}")
    if "mthreads" not in triton.backends.backends:
        pytest.skip("the installed Triton does not provide the MThreads backend")
    # torch_fl already routes each overload through its generated C++ kernels.
    # Global FlagGems registration would bypass that dispatcher and register
    # lift_fresh, whose vendor-device check rejects flagos tensors.


def _flag_gems_call_site(qualname):
    """The module and attribute the generated kernel's qualname resolves to.

    The codegen freezes the *package-level* name -- ``flag_gems.<fn>``, see
    ``_normalize_flaggems_qualname`` in scripts/codegen_ops.py -- and
    ``PythonOpCache::GetFunc`` resolves it by importing the prefix and taking
    the attribute, so ``flag_gems.<fn>`` is the call site.

    Vendors override some ops by rebinding that package attribute, and the
    keyword-argument caller (``GetFuncBare``) goes to ``flag_gems.ops.<fn>``
    instead; ``flag_gems.ops.<op>.<fn>`` also exists but is a different slot
    holding the same object when nothing overrides it. Both entry points the
    caller can reach are patched, so the test counts the call whichever one the
    dispatcher takes.
    """
    _, _, function_name = qualname.rpartition(".")
    return [
        (importlib.import_module("flag_gems"), function_name),
        (importlib.import_module("flag_gems.ops"), function_name),
    ]


def _track(monkeypatch, qualname, calls):
    """Count every call to ``qualname`` through the entry point it reaches."""
    function_name = qualname.rpartition(".")[2]
    for module, attribute in _flag_gems_call_site(qualname):
        if not hasattr(module, attribute):
            continue
        original = getattr(module, attribute)
        if getattr(original, "_musa_counts_calls", False):
            continue

        def wrapper(*args, _original=original, _name=function_name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        wrapper._musa_counts_calls = True
        monkeypatch.setattr(module, attribute, wrapper)


#: The generated kernels' qualname for each op this test drives, in call order.
ROUTED_OPS = [
    "flag_gems.all",
    "flag_gems.all_dims",
    "flag_gems.any",
    "flag_gems.any_dims",
    "flag_gems.repeat_interleave_tensor",
    "flag_gems.index_add",
    "flag_gems.index_add_",
]


def test_selected_flaggems_routes_execute_on_s5000(monkeypatch):
    """Each routed op runs its FlagGems kernel, not the CPU fallback.

    The counted calls are an assertion on the route and the CPU comparison
    after each one is an assertion on the result, so a route that runs the
    wrong kernel fails here rather than in a downstream model.
    """
    _require_flaggems_mthreads()
    calls = []
    for qualname in ROUTED_OPS:
        _track(monkeypatch, qualname, calls)

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

    assert calls == [name.rpartition(".")[2] for name in ROUTED_OPS]


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

    The metric op is warmed up at a shape this test does not use first, and the
    spy is installed after that. Two things make the warm-up load-bearing:

    * ``LibEntry.run`` caches the compiled launch per *specialized* key
      (``align32`` on M, N and K for ``mm``), so 97x95x99 and 33x33x33 share the
      entry 64x64x64 was compiled under -- a second float64 shape never reaches
      its tuner at all once any float64 ``mm`` has run in the process.
    * a repaired launch hands the declared space back, so a repair later in the
      session cannot be seen through the tuner from the outside.

    Before that warm-up, the arithmetic still reproduces the refused shape at a
    cache-miss key: it lands on the narrow tile 33x32x32, where 64x64x64 at five
    stages asks 262144 B. A fixed shape would then be testing what the
    ConfigCache happens to hold rather than the fit.
    """
    _require_flaggems_mthreads()
    limit = torch_fl.flagos._musa_shared_memory_limit()
    if limit is None:
        pytest.skip("this backend does not report a static shared-memory limit")

    # Float64 ``mm`` is the only space this platform declares tighter than the
    # device allows, and it runs before the spy goes in (see the docstring).
    torch.mm(
        torch.randn(48, 48, dtype=torch.float64).to(DEVICE),
        torch.randn(48, 48, dtype=torch.float64).to(DEVICE),
    )

    seen = []
    fit = torch_fl.flagos._fitted_tune_space

    def spy(configs, args, kwargs, limit_bytes):
        fitted = fit(configs, args, kwargs, limit_bytes)
        seen.append((list(configs), fitted, limit_bytes))
        return fitted

    torch_fl.flagos._fitted_tune_space = spy
    try:
        # (M, K, N), also the tuner's specialized key. 97x95x99 is the one shape
        # an empty ConfigCache cannot have answered for; 33x32x32 is the first
        # key whose 64x64x64 tile cannot be narrowed below the limit, so its fit
        # is looked at rather than the repaired launch that follows it.
        for m, k, n in ((33, 32, 32), (97, 95, 99)):
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
    finally:
        torch_fl.flagos._fitted_tune_space = fit

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

    over_limit = False
    for declared, fitted, limit_bytes in seen:
        assert limit_bytes == limit
        refused = [ask(config) >= limit_bytes for config in declared]
        if any(refused):
            over_limit = True
            assert fitted is not None, (
                "a declared configuration asks more than the device can give and"
                f" was handed over unchanged: {[ask(c) for c in declared]}"
            )
        for config in declared if fitted is None else fitted:
            assert ask(config) < limit_bytes, (
                f"num_stages={config.num_stages} on {config.kwargs} asks"
                f" {ask(config)} B against a {limit_bytes} B limit"
            )
    assert over_limit, "no declared space reached the limit, so the fit was untested"
