"""Real MTT S5000 coverage for the MThreads FlagGems hybrid path."""

import importlib
import sys

import pytest
import torch
import torch_fl


pytestmark = pytest.mark.musa
DEVICE = torch.device("flagos:0")

#: Attribute ``torch_fl.flagos._patch_flaggems_tune_space`` sets on every
#: ``run`` it wraps; see that function for why the repair lives behind it.
_TUNE_SPACE_FLAG = "_flagos_tune_space_fits_device"


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


def test_the_flag_gems_randn_route_reaches_its_kernel():
    """``flag_gems.randn`` is driven and leaves a finite float tensor.

    This is the route, not the RNG contract. ``torch.rand`` on this lane does
    not draw from the generator it reports -- measured through ``import
    torch_fl`` on the MTT S5000, the state after two identically seeded draws
    is identical while the values differ, on four cards -- so any assertion
    that compares values or generator states across a reseed would measure that
    defect. What is asserted here is what this file routes: the kernel runs,
    lands on the device, and produces values.
    """
    _require_flaggems_mthreads()
    import flag_gems

    # The package-level name, not ``flag_gems.ops.randn``: that is what the
    # generated kernel resolves to, and on MUSA ``SpecOpRegistrar`` has rebound
    # it to the vendor override ``_mthreads.ops.randn``. Driving the generic
    # module here would exercise a callable the dispatch never reaches.
    flaggems_randn = flag_gems.randn

    torch.flagos.manual_seed(20260817)
    native = torch.rand(64, device=DEVICE)
    drawn = flaggems_randn((64,), device=DEVICE)
    assert drawn.device == DEVICE
    torch.flagos.synchronize()
    host = drawn.cpu()
    assert host.device.type == "cpu"
    assert host.dtype == torch.float32
    assert torch.isfinite(host).all()
    assert not torch.equal(host, native.cpu())


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


def test_the_policy_tuners_carry_the_tune_space_repair():
    """FlagGems' kernels on this platform get the repaired tune space.

    The repair cannot be spied from a launch. ``LibEntry.run`` calls the
    function object it was built with, and the vendor ``mm`` module builds its
    entries at import, so on this platform nothing goes through the policy
    subclass a class-level patch could be observed on -- measured here with a
    tracehook and a ``LibTuner.run`` wrapper around float32 and float64 ``mm``
    and ``add``: zero Python frames, inside a rig whose per-launch patches
    demonstrably run. What can be asserted is the patched state itself: every
    policy tuner the loaded FlagGems kernel modules hold wraps a ``run`` that
    carries the repair, at the class and on the instance the kernel reaches.
    """
    _require_flaggems_mthreads()
    libtuner = importlib.import_module("flag_gems.utils.libentry").LibTuner

    def tuners(cls):
        for subclass in cls.__subclasses__():
            yield from tuners(subclass)
            yield subclass

    classes = list(tuners(libtuner))
    assert classes, "FlagGems registered no tuner policy"
    for subclass in classes:
        assert getattr(subclass.run, _TUNE_SPACE_FLAG, False), (
            f"{subclass.__name__}.run does not carry the tune-space repair"
        )

    import flag_gems  # noqa: F401 -- routed ops must be resolved to find entries

    entries = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "") or ""
        if not name.startswith(("flag_gems.ops", "_mthreads.ops")):
            continue
        for value in vars(module).values():
            if type(value).__name__ == "LibEntry":
                entries.append(value)
    assert entries, "no FlagGems LibEntry was loaded to check"
    for entry in entries:
        tuner = getattr(entry, "fn", None)
        if not isinstance(tuner, libtuner):
            continue
        assert getattr(tuner.run, _TUNE_SPACE_FLAG, False), (
            f"{entry.jit_function.fn.__name__} holds a tuner without the"
            " tune-space repair"
        )
