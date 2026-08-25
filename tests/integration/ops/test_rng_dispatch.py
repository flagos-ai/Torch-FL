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

"""flagos unified-RNG regression coverage.

The contract: **one seed source reaches every RNG op on the flagos device.**
`torch.manual_seed(s)` must make every draw reproducible regardless of which of
the two very different paths an op takes to get its randomness:

  * FlagGems Triton path (`flagos_python` in the conf) -- reads seed+offset from
    `torch.cuda.default_generators[device]`, a per-device philox CUDA generator
    installed by the vendor compat shim.
  * native CUDA path (`= cuda` in the conf) -- boxes the tensor to CUDA and calls
    the real ATen kernel. On a CPU-torch wheel ATen's own
    `getDefaultCUDAGenerator()` is unreachable from `torch.manual_seed` (there is
    no `torch._C._cuda_manualSeed` binding), so the generated kernels inject that
    same shared generator via `GetFlagosDefaultCudaGenerator()`.

Because both paths now converge on one generator, these tests run under *both*
backend configs -- pure vendor (`-m main_ops`) and FlagGems
(`FLAGOS_USE_FLAGGEMS=1 -m "flaggems and main_ops"`). That is deliberate: the
native-path injection is only exercised by the pure-vendor run, and a config
asymmetry in either direction is exactly the kind of regression this file exists
to catch.

Every test carries `main_ops` because CI's operator jobs select on it; a test
without that marker never runs in CI.

Usage:
    pytest tests/integration/ops/test_rng_dispatch.py -v
    FLAGOS_USE_FLAGGEMS=1 pytest tests/integration/ops/test_rng_dispatch.py -v
"""

import os

import pytest

import torch
import torch_fl  # noqa: F401


def _dropout_on_flaggems() -> bool:
    """True when `native_dropout` actually routes to the FlagGems Triton kernel.

    `FLAGOS_USE_FLAGGEMS=1` alone is not enough to answer this. An allowlist
    config (Kunlun) switches FlagGems on for a measured handful of overloads and
    leaves everything else, `native_dropout` included, on the vendor path. So
    read the selected config's route instead of inferring it from the env var:
    dropout reproducibility follows the route, not the switch.
    """
    if not _flaggems_on():
        return False
    conf = os.environ.get("FLAGOS_BACKEND_CONFIG", "")
    if not conf or not os.path.exists(conf):
        # Switch is on with no resolved config: assume the generic config, whose
        # native_dropout is flagos_python.
        return True
    with open(conf) as f:
        for line in f:
            name, _, backend = line.partition("=")
            if name.strip() == "native_dropout":
                return backend.strip() == "flagos_python"
    return False


def _flaggems_on() -> bool:
    """Mirrors conftest._flaggems_enabled, for the one xfail whose *expected*
    outcome (not merely whether it runs) depends on the active backend config."""
    return os.environ.get("FLAGOS_USE_FLAGGEMS", "0").lower() not in (
        "0",
        "",
        "off",
        "false",
    )


DEVICE = "flagos:0"
SEED = 1234
# Far enough from SEED that a collision is not plausible, but the value itself is
# arbitrary -- any different seed must produce a different draw.
OTHER_SEED = SEED + 977

aten = torch.ops.aten


def _empty(n=64, dtype=torch.float32, device=DEVICE):
    return torch.empty(n, dtype=dtype, device=device)


def _draw(op, seed):
    torch.manual_seed(seed)
    return op().float().cpu()


def _explicit_generator(seed):
    """A caller-supplied generator that this backend's RNG kernels accept.

    The device matters and is not the same everywhere. Under CUDA boxing the
    kernels take a real CUDA generator, and that is the case the injection logic
    was written for. On a vendor backend with no CUDA library (Ascend) there is
    no CUDA generator to construct -- `torch.Generator(device="cuda")` raises
    "Cannot get CUDA generator without ATen_cuda library", and a `flagos` one is
    rejected outright ("Expected a 'cpu' device type for generator"), because the
    aclnn kernels seed themselves from the default *CPU* generator.

    A `flagos` generator is not a portable fallback: it carries an mt19937
    state, while CUDA boxing needs a philox generator and aclnn consumes seeds
    from a CPU generator. Passing the incompatible device type must raise, just
    like native CPU/CUDA generator mismatches (and is covered below).

    So try CUDA first and fall back to CPU. What the two tests below assert --
    that an explicit generator is honoured and stays isolated from
    `torch.manual_seed` -- is a property of the injection logic, not of the
    generator's device, and holds either way.
    """
    try:
        gen = torch.Generator(device="cuda")
    except RuntimeError:
        gen = torch.Generator()  # cpu
    gen.manual_seed(seed)
    return gen


# --- op catalogue -----------------------------------------------------------
#
# Grouped by the mechanism each group stresses, because the injection bug this
# file guards against reappeared once per codegen template: an op whose schema
# lacks `Generator?` gets no injection and silently falls back to ATen's own
# default generator. Factory, `*_like` and out-variant overloads each needed a
# separate fix, so each is represented here.

# Elementwise / reduction RNG that FlagGems implements in Triton (and that the
# vendor config routes to the native kernels).
_GEMS_PATH_OPS = {
    "rand": lambda: torch.rand(64, device=DEVICE),
    "randn": lambda: torch.randn(64, device=DEVICE),
    "rand_like": lambda: torch.rand_like(_empty()),
    "randn_like": lambda: torch.randn_like(_empty()),
    "uniform_": lambda: _empty().uniform_(),
    "exponential_": lambda: _empty().exponential_(),
    "bernoulli_.float": lambda: _empty().bernoulli_(0.5),
    "multinomial": lambda: torch.multinomial(
        torch.ones(10, device=DEVICE), 5, replacement=True
    ),
}

# `Generator?`-carrying native kernels: the `if (!generator.has_value())` form.
_NATIVE_GENERATOR_OPS = {
    "normal_": lambda: _empty().normal_(),
    "normal_.mean_std": lambda: _empty().normal_(2.0, 3.0),
    "normal.float_float": lambda: torch.normal(0.0, 1.0, (64,), device=DEVICE),
    "normal.Tensor_float": lambda: torch.normal(torch.zeros(64, device=DEVICE), 1.0),
    "normal.Tensor_Tensor": lambda: torch.normal(
        torch.zeros(64, device=DEVICE), torch.ones(64, device=DEVICE)
    ),
    "bernoulli.Tensor": lambda: torch.bernoulli(torch.full((64,), 0.5, device=DEVICE)),
    "bernoulli_.Tensor": lambda: _empty().bernoulli_(
        torch.full((64,), 0.5, device=DEVICE)
    ),
    "random_": lambda: _empty(dtype=torch.int64).random_(),
    "random_.to": lambda: _empty(dtype=torch.int64).random_(50),
    "random_.from": lambda: _empty(dtype=torch.int64).random_(10, 50),
    "log_normal_": lambda: _empty().log_normal_(),
    "cauchy_": lambda: _empty().cauchy_(),
    "geometric_": lambda: _empty().geometric_(0.5),
    "poisson": lambda: torch.poisson(torch.full((64,), 4.0, device=DEVICE)),
    "_standard_gamma": lambda: torch._standard_gamma(
        torch.full((64,), 2.0, device=DEVICE)
    ),
    "_sample_dirichlet": lambda: torch._sample_dirichlet(
        torch.full((8, 4), 2.0, device=DEVICE)
    ),
    "binomial": lambda: torch.binomial(
        torch.full((64,), 10.0, device=DEVICE), torch.full((64,), 0.5, device=DEVICE)
    ),
}

# Generator-LESS factory / *_like overloads. `torch.randint(...)` and
# `torch.randperm(...)` dispatch to these, NOT to their `.generator` siblings, so
# they were non-reproducible until the factory and functional-pure templates
# learned to thread the shared generator in explicitly.
_GENERATOR_LESS_OPS = {
    "randint": lambda: torch.randint(0, 100, (64,), device=DEVICE),
    "randint.low": lambda: torch.randint(10, 100, (64,), device=DEVICE),
    "randint_like": lambda: torch.randint_like(_empty(dtype=torch.int64), 0, 50),
    "randperm": lambda: torch.randperm(50, device=DEVICE),
}

# Generator-LESS out-variants -- the last template to be fixed. ATen places the
# injected `optional<Generator>` before the first of names/memory_format/out, so
# a mis-ordered injection would bind the wrong overload; a silent fallback to
# ATen's default generator shows up here as non-reproducibility.
_OUT_VARIANT_OPS = {
    "rand.out": lambda: torch.rand(64, out=_empty()),
    "randn.out": lambda: torch.randn(64, out=_empty()),
    "rand.names_out": lambda: aten.rand.names_out([64], names=None, out=_empty()),
    "randn.names_out": lambda: aten.randn.names_out([64], names=None, out=_empty()),
    "rand_like.out": lambda: aten.rand_like.out(_empty(), out=_empty()),
    "randn_like.out": lambda: aten.randn_like.out(_empty(), out=_empty()),
    "randint.out": lambda: torch.randint(0, 100, (64,), out=_empty(dtype=torch.int64)),
    "randint.low_out": lambda: torch.randint(
        10, 100, (64,), out=_empty(dtype=torch.int64)
    ),
    "randint_like.out": lambda: aten.randint_like.out(
        _empty(dtype=torch.int64), 50, out=_empty(dtype=torch.int64)
    ),
    "randint_like.low_dtype_out": lambda: aten.randint_like.low_dtype_out(
        _empty(dtype=torch.int64), 5, 50, out=_empty(dtype=torch.int64)
    ),
    "randperm.out": lambda: torch.randperm(50, out=_empty(50, torch.int64)),
}

_ALL_OPS = {
    **_GEMS_PATH_OPS,
    **_NATIVE_GENERATOR_OPS,
    **_GENERATOR_LESS_OPS,
    **_OUT_VARIANT_OPS,
}


class TestRngReproducible:
    """Same seed -> identical draw, for every routed RNG op."""

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize("name", list(_ALL_OPS))
    def test_same_seed_same_draw(self, name):
        if torch.__version__.startswith("2.9") and "_like" in name:
            pytest.skip(
                "torch 2.9 _like RNG ops fall back to ATen's own default"
                " generator instead of the shared flagos generator, so they"
                " are not reproducible under torch.manual_seed. Tracked as a"
                " 2.9-downgrade follow-up."
            )
        op = _ALL_OPS[name]
        assert torch.equal(_draw(op, SEED), _draw(op, SEED)), (
            f"{name} is not reproducible under torch.manual_seed -- it is most "
            f"likely falling back to ATen's own default generator instead of the "
            f"shared flagos generator"
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize("name", list(_ALL_OPS))
    def test_different_seed_differs(self, name):
        # Reproducibility alone is satisfied by a *constant* draw, which is how
        # the original philox monkeypatch bug hid: it seeded once and ignored
        # later manual_seed calls. Seed-sensitivity is what rules that out.
        op = _ALL_OPS[name]
        assert not torch.equal(_draw(op, SEED), _draw(op, OTHER_SEED)), (
            f"{name} ignores the seed -- same draw for two different seeds"
        )


class TestRngSeedSource:
    """The seed plumbing itself, independent of any single op."""

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_default_generators_populated(self):
        # The whole scheme rests on this list being non-empty and philox-shaped
        # (2 x int64 = seed, offset). A 5056-byte mt19937 state here means the
        # shim wired up the PrivateUse1 generator by mistake, which FlagGems
        # unpacks as "too many values to unpack".
        #
        # This is a requirement of the *CUDA-shaped* seed paths only: FlagGems
        # Triton kernels read seed+offset out of this generator, and the boxed
        # native kernels inject it. A backend whose kernels seed themselves from
        # the default CPU generator (Ascend/aclnn) has no CUDA generator to
        # populate, and `torch.cuda.default_generators` is legitimately empty
        # there -- so require the philox shape only where a CUDA generator exists.
        gens = torch.cuda.default_generators
        if len(gens) == 0:
            pytest.skip(
                "no CUDA generator on this backend; RNG kernels seed from the "
                "default CPU generator instead (see torch_fl.flagos.default_generators)"
            )
        assert gens[0].get_state().numel() == 16, (
            "default generator state is not philox-shaped (2 x int64)"
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize("attr", ["cuda", "flagos"])
    def test_default_generators_iterable(self, attr):
        # Upstream types default_generators as a *tuple*, so callers iterate it,
        # slice it and list() it. Our shims are list-like proxies with no
        # __iter__, which sends Python to the legacy protocol: __getitem__(0, 1,
        # 2, ...) until IndexError. Unbounded __getitem__ therefore made
        # `for g in default_generators` an infinite loop that allocated a fresh
        # generator per step -- a hang, not an error, so nothing surfaced it.
        gens = getattr(torch if attr == "cuda" else torch_fl, attr).default_generators
        n = len(gens)
        assert len(list(gens)) == n, "iteration does not stop at device_count"
        assert len(gens[:2]) == min(2, n), "slicing is not supported"
        with pytest.raises(IndexError):
            gens[n]

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_manual_seed_reaches_flagos_module(self):
        # torch.random._seed_custom_device only seeds this device module when it
        # exposes BOTH manual_seed_all and _is_in_bad_fork; missing either one
        # makes torch.manual_seed warn "does not take effect" and skip it.
        torch.manual_seed(SEED)
        assert torch_fl.flagos.initial_seed() == SEED

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_manual_seed_emits_no_ineffective_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.manual_seed(SEED)
        assert not [w for w in caught if "does not take effect" in str(w.message)]

    @pytest.mark.skipif(
        torch.__version__.startswith("2.9"),
        reason=(
            "torch 2.9 lacks the native generator device-mismatch check added"
            " in 2.10; a flagos generator on a boxed-to-CUDA tensor"
            " infinite-redispaches to SIGSEGV instead of raising RuntimeError."
            " Tracked as a 2.9-downgrade follow-up."
        ),
    )
    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_generator_device_mismatch_raises(self):
        # A Generator contributes its dispatch key to operator selection. After
        # the boxed kernel changed its tensor to CUDA, a flagos generator used
        # to select the same flagos kernel again: infinite redispatch ending in
        # SIGSEGV. Like native device mismatches, this unsupported generator
        # combination must fail cleanly instead.
        gen = torch.Generator(device="flagos")
        with pytest.raises(RuntimeError, match="device type for generator"):
            _empty(8).normal_(generator=gen)

    @pytest.mark.skipif(
        torch.__version__.startswith("2.9"),
        reason=(
            "torch 2.9 lacks the native generator device-mismatch check added"
            " in 2.10 (see test_flagos_generator_device_mismatch_raises)."
        ),
    )
    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_generator_factory_mismatch_raises(self):
        # Factory RNG has no Tensor key to override the generator's PrivateUse1
        # key, so it is a distinct recursion entry point. Only CUDA-shaped
        # boxing backends expose this overload; aclnn uses its own factory path.
        if len(torch.cuda.default_generators) == 0:
            pytest.skip("factory generator overload requires CUDA boxing")
        gen = torch.Generator(device="flagos")
        with pytest.raises(RuntimeError, match="device type for generator"):
            torch.rand(8, device=DEVICE, generator=gen)

    @pytest.mark.skipif(
        torch.__version__.startswith("2.9"),
        reason=(
            "torch 2.9 lacks the native generator device-mismatch check added"
            " in 2.10 (see test_flagos_generator_device_mismatch_raises)."
        ),
    )
    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_generator_does_not_reroute_cpu_op(self):
        # The tensor is CPU, so a PrivateUse1 redispatch here can only have come
        # from the generator. This covers the dispatch-key root cause directly,
        # independently of CUDA boxing.
        gen = torch.Generator(device="flagos")
        with pytest.raises(RuntimeError, match="device type for generator"):
            torch.empty(8).normal_(generator=gen)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_explicit_generator_is_honoured(self):
        # Injection must only fill an *absent* generator. A caller-supplied one
        # keeps its own stream, reproducible on its own terms.
        gen = _explicit_generator(99)
        a = _empty().normal_(generator=gen).cpu()
        gen.manual_seed(99)
        b = _empty().normal_(generator=gen).cpu()
        assert torch.equal(a, b)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_explicit_generator_isolated_from_manual_seed(self):
        gen = _explicit_generator(5)
        a = _empty().normal_(generator=gen).cpu()
        torch.manual_seed(SEED)  # must not perturb `gen`
        gen.manual_seed(5)
        b = _empty().normal_(generator=gen).cpu()
        assert torch.equal(a, b)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_interleaved_paths_share_one_seed(self):
        # A model mixes both paths freely (dropout next to init). One seed must
        # drive the whole interleaved sequence, so drawing from the FlagGems path
        # and the native path under one seed has to replay identically.
        def mixed():
            gems = torch.rand(32, device=DEVICE)
            native = _empty(32).normal_()
            return torch.cat([gems, native])

        assert torch.equal(_draw(mixed, SEED), _draw(mixed, SEED))
        assert not torch.equal(_draw(mixed, SEED), _draw(mixed, OTHER_SEED))


class TestRngMultiDevice:
    """Per-device generators must be independently seeded and addressed.

    One rank per card is the normal distributed layout, so an RNG scheme that
    only works on device 0 is broken in exactly the setting that matters most.
    """

    @staticmethod
    def _second_device():
        if torch_fl.flagos.device_count() < 2:
            pytest.skip("needs at least 2 flagos devices")
        return "flagos:1"

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize("name", ["randn", "normal_", "randint", "randperm"])
    def test_reproducible_on_second_device(self, name):
        dev = self._second_device()
        ops = {
            "randn": lambda: torch.randn(64, device=dev),
            "normal_": lambda: _empty(64, device=dev).normal_(),
            "randint": lambda: torch.randint(0, 100, (64,), device=dev),
            "randperm": lambda: torch.randperm(50, device=dev),
        }
        op = ops[name]
        assert torch.equal(_draw(op, SEED), _draw(op, SEED))
        assert not torch.equal(_draw(op, SEED), _draw(op, OTHER_SEED))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_each_device_has_its_own_generator(self):
        dev = self._second_device()
        # Same empty-backend guard as test_default_generators_populated above:
        # a backend whose kernels seed from the default CPU generator
        # (Ascend/aclnn) has no per-device CUDA generator to hold distinct, and
        # `torch.cuda.default_generators` is legitimately empty there. Without
        # this the `>= 2` assert below is `assert 0 >= 2` on every such backend,
        # which contradicts the sibling test's stance within the same file.
        gens = torch.cuda.default_generators
        if len(gens) == 0:
            pytest.skip(
                "no CUDA generator on this backend; RNG kernels seed from the "
                "default CPU generator instead (see torch_fl.flagos.default_generators)"
            )
        assert len(gens) >= 2
        # Same seed on both devices: the draws are allowed to coincide (identical
        # philox inputs), but the generators must be distinct objects, or seeding
        # one device would silently reset another.
        assert gens[0] is not gens[1]
        torch.manual_seed(SEED)
        d0 = torch.randn(64, device=DEVICE).cpu()
        torch.manual_seed(SEED)
        d1 = torch.randn(64, device=dev).cpu()
        assert d0.shape == d1.shape

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    @pytest.mark.xfail(
        reason="FlagGems multinomial launches its Triton kernel against the wrong "
        "device for any index != 0 -- 'Pointer argument (at 0) cannot be accessed "
        "from Triton (cpu tensor?)' on flagos:1..7, fine on flagos:0. A device-context "
        "bug on the FlagGems Triton path rather than an RNG one: the same op on the "
        "native path is reproducible on every device. xfail (non-strict) so the "
        "eventual fix surfaces as an xpass instead of being silently assumed.",
        strict=False,
        raises=ValueError,
    )
    def test_multinomial_on_second_device(self):
        dev = self._second_device()

        def op():
            return torch.multinomial(torch.ones(10, device=dev), 5, replacement=True)

        assert torch.equal(_draw(op, SEED), _draw(op, SEED))


class TestRngDistribution:
    """Reproducibility is worthless if the numbers are wrong.

    A mis-threaded generator argument can bind a different overload and still
    look reproducible, so pin the actual distributions. Tolerances are loose
    enough for 100k samples at any seed.
    """

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_rand_uniform_0_1(self):
        torch.manual_seed(42)
        r = torch.rand(100_000, device=DEVICE).cpu()
        assert abs(r.mean().item() - 0.5) < 0.02
        assert r.min().item() >= 0.0 and r.max().item() < 1.0

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_randn_standard_normal(self):
        torch.manual_seed(42)
        n = torch.randn(100_000, device=DEVICE).cpu()
        assert abs(n.mean().item()) < 0.02
        assert abs(n.std().item() - 1.0) < 0.02

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_normal_mean_std(self):
        torch.manual_seed(42)
        n = _empty(100_000).normal_(2.0, 3.0).cpu()
        assert abs(n.mean().item() - 2.0) < 0.05
        assert abs(n.std().item() - 3.0) < 0.05

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_exponential_rate_1(self):
        torch.manual_seed(42)
        e = _empty(100_000).exponential_().cpu()
        assert abs(e.mean().item() - 1.0) < 0.05
        assert e.min().item() >= 0.0

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_uniform_range(self):
        torch.manual_seed(42)
        u = _empty(100_000).uniform_(2.0, 5.0).cpu()
        assert abs(u.mean().item() - 3.5) < 0.05
        assert u.min().item() >= 2.0 and u.max().item() <= 5.0

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_randint_range(self):
        torch.manual_seed(42)
        r = torch.randint(0, 10, (100_000,), device=DEVICE).cpu()
        assert r.min().item() == 0 and r.max().item() == 9
        assert abs(r.float().mean().item() - 4.5) < 0.05

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_randperm_is_a_permutation(self):
        torch.manual_seed(42)
        p = torch.randperm(2000, device=DEVICE).cpu()
        assert sorted(p.tolist()) == list(range(2000))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_poisson_mean_equals_variance(self):
        torch.manual_seed(42)
        p = torch.poisson(torch.full((100_000,), 4.0, device=DEVICE)).cpu()
        assert abs(p.mean().item() - 4.0) < 0.06
        assert abs(p.var().item() - 4.0) < 0.15  # Poisson: var == mean

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_bernoulli_probability(self):
        torch.manual_seed(42)
        b = _empty(100_000).bernoulli_(0.3).cpu()
        assert set(b.unique().tolist()) <= {0.0, 1.0}
        assert abs(b.mean().item() - 0.3) < 0.01

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_multinomial_respects_weights(self):
        torch.manual_seed(42)
        weights = torch.tensor([0.1, 0.9], device=DEVICE)
        picks = torch.multinomial(weights, 20_000, replacement=True).cpu()
        assert set(picks.unique().tolist()) <= {0, 1}
        assert abs((picks == 1).float().mean().item() - 0.9) < 0.02


class TestRngDropout:
    """dropout is the RNG op models actually depend on for correctness.

    It is split out because its reproducibility is config-dependent: FlagGems
    routes `native_dropout` to its own Triton kernel (seeded from the shared
    generator), while the vendor config calls the ATen composite, whose schema
    exposes no `Generator?` at all -- there is no argument to inject, so the
    randomness comes from ATen's default generator and `torch.manual_seed` cannot
    reach it. Marked `flaggems` so it asserts the contract only where the
    contract currently holds; see test_dropout_masks_are_valid for the part that
    must hold everywhere.
    """

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_dropout_masks_are_valid(self):
        # Config-independent: scaled-or-zeroed entries, roughly the right rate.
        torch.manual_seed(SEED)
        out = torch.nn.functional.dropout(
            torch.ones(100_000, device=DEVICE), p=0.5, training=True
        ).cpu()
        kept = out != 0
        assert abs(kept.float().mean().item() - 0.5) < 0.01
        # Survivors are scaled by 1/(1-p).
        assert torch.allclose(out[kept], torch.full_like(out[kept], 2.0), atol=1e-5)

    @staticmethod
    def _dropout():
        return torch.nn.functional.dropout(
            torch.ones(256, device=DEVICE), p=0.5, training=True
        )

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    @pytest.mark.skipif(
        not _dropout_on_flaggems(),
        reason="the active FlagGems config leaves native_dropout on the vendor "
        "path (allowlist config), so there is no FlagGems dropout to assert; "
        "test_dropout_reproducible covers that case as a non-strict xfail.",
    )
    def test_dropout_reproducible_on_flaggems_path(self):
        assert torch.equal(_draw(self._dropout, SEED), _draw(self._dropout, SEED))
        assert not torch.equal(
            _draw(self._dropout, SEED), _draw(self._dropout, OTHER_SEED)
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.xfail(
        condition=not _dropout_on_flaggems(),
        reason="native_dropout has no `Generator?` in its ATen schema, so there is "
        "no argument for the generated kernel to inject into -- the vendor path "
        "draws from ATen's own default CUDA generator, which torch.manual_seed "
        "cannot reach on a CPU-torch wheel. The FlagGems path routes to its own "
        "Triton kernel and is reproducible. Non-strict xfail so closing this gap "
        "(e.g. decomposing dropout onto bernoulli_) shows up as an xpass.",
        strict=False,
    )
    def test_dropout_reproducible(self):
        assert torch.equal(_draw(self._dropout, SEED), _draw(self._dropout, SEED))
