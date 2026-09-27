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
    if flag_gems.runtime.device.vendor_name != "mthreads":
        pytest.skip(
            "FlagGems resolved a different vendor: "
            f"{flag_gems.runtime.device.vendor_name!r}"
        )
    # Deliberately *not* ``flag_gems.enable()``. That build has no
    # ``_mthreads/enable_configs.yaml``, so enabling the patch set registers
    # every FlagGems op -- including the ones ``backends_musa.conf`` routes to
    # the native MUSA kernels -- and on this vendor build the generic
    # implementations cannot serve what those routes can. Measured with the
    # registry enabled: ``tensor * 2.5`` raises ``RuntimeError: aten::mul()
    # Expected a value of type 'Tensor' for argument 'other' but instead found
    # type 'float'``, and ``torch.tensor(2.5, device="flagos:0")`` raises
    # ``ValueError: lift_fresh Triton kernel requires a musa tensor`` -- the same
    # two shapes of gap that carry ``# gcu`` markers in the FlagGems file, from a
    # registry with no vendor config to narrow it. ``torch_fl/__init__.py``
    # refuses that registry for exactly this reason ("would register a competing
    # PrivateUse1 implementation and bypass the shared dispatcher"), so these
    # tests measure the route the product ships: the conf's, reached through the
    # generated kernels below.


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


def test_float64_addmm_is_exact_through_every_entrypoint():
    """The float64 ``addmm`` repair, against a CPU float64 reference.

    ``_patch_flaggems_addmm_fp64`` serves the float64 case of all five ATen
    overloads the MUSA config routes to FlagGems (``addmm``, ``addmm.out``,
    ``addmm_``, ``addmm.dtype``, ``addmm.dtype_out``) by composing the op out of
    ``mm``, because the mthreads kernel rounds both dot operands to float32
    inside its K loop: over this matrix of cases the original measures 6e-08 to
    1.2e-07 relative, where float64 is 1e-16, while ``mm`` -- the same matmul one
    op over, with no such cast -- is exact, a single-column product aside, where
    ``mm`` rounds as well and the composition widens the operand instead.

    That the patch is installed, and that FlagGems' own callables are still
    behind it, is asserted before anything is measured: an unpatched route here
    is wrong rather than broken, so on a tree without the repair every case
    below still returns a float64 tensor and only the tolerance separates a
    float64 answer from a float32 one.
    """
    _require_flaggems_mthreads()
    import flag_gems

    for name in ("addmm", "addmm_out", "addmm_", "addmm_dtype", "addmm_dtype_out"):
        wrapper = flag_gems.__dict__.get(name)
        assert wrapper is not None, name
        assert getattr(wrapper, torch_fl.flagos._ADDMM_FP64_PATCHED, False), name
        assert callable(wrapper.__wrapped__), name
    # The float64-exact product the composition is built on, as opposed to the
    # generic module of the same name.
    assert flag_gems.mm.__module__ == "_mthreads.ops.mm"

    for m, k, n in [
        (33, 33, 33),
        (48, 64, 32),
        (64, 64, 64),
        (128, 96, 64),
        # A single-column product is the one shape ``mm`` is not float64 on: it
        # dispatches to a float32 GEMV kernel, so it is the shape the composition
        # has to widen rather than hand over (measured before the widening:
        # 2.0e-05 relative at 64x64x1, 2.1e-03 at 512x512x1).
        (64, 64, 1),
        (512, 512, 1),
        (1, 64, 1),
    ]:
        mat1 = torch.randn(m, k, dtype=torch.float64)
        # A transposed second operand: the gate tests `layout`, not contiguity,
        # and the composition's `mm` serves a strided matrix.
        mat2 = torch.randn(n, k, dtype=torch.float64).t()
        biases = [
            torch.randn(m, n, dtype=torch.float64),
            torch.randn(n, dtype=torch.float64),
            torch.randn((), dtype=torch.float64),
        ]
        for index, bias in enumerate(biases):
            for alpha, beta in [(1, 1), (2.5, 0), (0.25, -1.75)]:
                case = f"addmm at {m}x{k}x{n}, bias {index}, alpha={alpha} beta={beta}"
                got = torch.addmm(
                    bias.to(DEVICE),
                    mat1.to(DEVICE),
                    mat2.to(DEVICE),
                    beta=beta,
                    alpha=alpha,
                ).cpu()
                torch.testing.assert_close(
                    got,
                    torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha),
                    rtol=1e-12,
                    atol=1e-12,
                    msg=case,
                )


def test_float64_addmm_out_variants_write_the_same_answer():
    """The four overloads that take an ``out``, driven the way ATen spells them.

    ``addmm.out`` and ``addmm.dtype_out`` write the buffer they are given and
    return it; ``addmm_`` and ``addmm.dtype`` allocate. The in-place pair is the
    one with a real hazard -- an ``addmm_``'s bias *is* its destination, so a
    composition that wrote the product first would lose the ``beta`` term -- and
    the dtype pair carries ``out_dtype`` positionally, which is where the gate
    reads it.
    """
    _require_flaggems_mthreads()

    m, k, n = 64, 64, 64
    bias = torch.randn(m, n, dtype=torch.float64)
    mat1 = torch.randn(m, k, dtype=torch.float64)
    mat2 = torch.randn(k, n, dtype=torch.float64)
    want = torch.addmm(bias, mat1, mat2, beta=0.25, alpha=-1.75)
    tensors = [t.to(DEVICE) for t in (bias, mat1, mat2)]
    options = {"beta": 0.25, "alpha": -1.75}

    out = torch.empty(m, n, dtype=torch.float64, device=DEVICE)
    assert torch.addmm(*tensors, out=out, **options) is out
    torch.testing.assert_close(out.cpu(), want, rtol=1e-12, atol=1e-12)

    in_place = tensors[0].clone()
    assert in_place.addmm_(tensors[1], tensors[2], **options) is in_place
    torch.testing.assert_close(in_place.cpu(), want, rtol=1e-12, atol=1e-12)

    got = torch.ops.aten.addmm.dtype(*tensors, torch.float64, **options)
    assert got.dtype == torch.float64
    torch.testing.assert_close(got.cpu(), want, rtol=1e-12, atol=1e-12)

    dtype_out = torch.empty(m, n, dtype=torch.float64, device=DEVICE)
    returned = torch.ops.aten.addmm.dtype_out(
        *tensors, torch.float64, out=dtype_out, **options
    )
    assert returned is dtype_out
    torch.testing.assert_close(dtype_out.cpu(), want, rtol=1e-12, atol=1e-12)


def test_a_nan_bias_does_not_reach_a_zero_beta_result():
    """``beta == 0`` drops the bias instead of scaling it, NaN included.

    The one place the composition is not a plain ``addmm`` expression: ``bias *
    0.0`` is NaN, so the term has to be skipped rather than multiplied, or a
    float64 ``addmm`` with a zero beta returns NaN where the reference --
    ATen's CPU kernel and the FlagGems kernel it replaced, both -- returns a
    finite product.
    """
    _require_flaggems_mthreads()

    m, k, n = 64, 64, 64
    bias = torch.full((m, n), float("nan"), dtype=torch.float64).to(DEVICE)
    mat1 = torch.randn(m, k, dtype=torch.float64).to(DEVICE)
    mat2 = torch.randn(k, n, dtype=torch.float64).to(DEVICE)

    got = torch.addmm(bias, mat1, mat2, beta=0.0, alpha=2.5).cpu()
    want = torch.addmm(
        torch.full((m, n), float("nan"), dtype=torch.float64),
        mat1.cpu(),
        mat2.cpu(),
        beta=0.0,
        alpha=2.5,
    )
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    # The spelling the skip exists to avoid, and the one the kernel this patch
    # replaces would have produced had it multiplied the bias in float64.
    assert not torch.isfinite(torch.full((m, n), float("nan")) * 0.0).any()


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
